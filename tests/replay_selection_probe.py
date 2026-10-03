"""Exact control, policy restart and independent selection/exposure verification."""
import argparse
import importlib.util
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from native_experiment import NativeCommands, read, sha, write
from replay_priority_probe import journal, speech_bytes
from stage_replay_reference import MASK, Random64

_spec = importlib.util.spec_from_file_location('replay_policy_audit', Path(__file__).with_name('replay_priority_experiment.py'))
_audit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_audit)
restore_reference = _audit.restore_reference


def verify_choices(base, schedule, records, final, prioritize):
    from extend_curriculum import read_schedule
    _, stages = read_schedule(schedule)
    # This independent stepper is used for learned continuations within one
    # stage. The fixture crossing a stage is checked by exact native controls.
    docs = stages[-1]['content'].split(b'\x1e')
    first = len(stages[-2]['content'].split(b'\x1e'))
    ref = restore_reference(base, docs, first)
    for event in records:
        ref.run_until(event['observation'] - 1)
        uniform_rng = Random64(ref.random.state)
        available = [g for g in ref.groups if g.items]
        group = available[uniform_rng.below(len(available))] if len(available) > 1 else available[0]
        uniform = group.items[uniform_rng.below(len(group.items))]
        assert event['group'] == ref.groups.index(group) + 1
        candidates = event['candidates']
        assert tuple(candidates[0][k] for k in ('document', 'offset', 'length')) == uniform
        alternatives = [i for i, e in enumerate(group.items) if e != uniform and e[2] == uniform[2]]
        assert len(candidates) == (2 if alternatives else 1)
        if alternatives:
            rng = Random64((42 ^ ((event['observation'] * 0x9e3779b97f4a7c15) & MASK)) or 1)
            expected = group.items[alternatives[rng.below(len(alternatives))]]
            assert tuple(candidates[1][k] for k in ('document', 'offset', 'length')) == expected
        deltas = [c['after'] - c['before'] for c in candidates]
        chosen = int(prioritize and len(candidates) == 2 and deltas[1] > max(0, deltas[0]))
        assert event['chosen'] == chosen
        # Alternative has the same stage and length. The independent uniform
        # reference therefore has identical persisted counters/descriptors/RNG.
    from experiment_checkpoint import state_record
    ref.run_until(state_record(final)['online_updates'])
    return ref.matches(final)


def check(exe, production, out):
    import subprocess
    exe, production, out = exe.resolve(), production.resolve(), out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    native = NativeCommands(exe, out)
    fixture = out / 'source'
    fixture.mkdir()
    setup = NativeCommands(production, fixture)
    initial = [f'The {item} is in the {place}.'.encode() for item in ('key', 'cup', 'pen', 'box') for place in ('bag', 'tub')]
    lessons = [f'The {item} is in the {place}.\nAnswer: {place}.'.encode() for item in ('key', 'cup', 'pen', 'box') for place in ('bag', 'tub')]
    docs = [*initial, *lessons, b'Rain fills a stream beside a garden. The leaves turn in the wind.']
    for index, count in enumerate((8, 16, 17), 1):
        (fixture / f'{index}.dat').write_bytes(b'\x1e'.join(docs[:count]))
    schedule = fixture / 'curriculum.sg'
    schedule.write_bytes(b'SGCURRICULUM3\n24 "1.dat" 1 all 1\n64 "2.dat" .5 new 64\n160 "3.dat" .25 new 1\n')
    results = []
    for math in ('fp32', 'tf32'):
        base = fixture / f'{math}-base/latest.ckpt'
        setup('live', '--curriculum', schedule, '--out', base.parent, '--channels', 16, '--hidden', 24,
              '--layers', 2, '--cell', 'associative', '--chunk', 16, '--updates', 48,
              '--replay', 'stage', '--replay-capacity', 64, '--replay-every', 2, '--speak-every', 8,
              '--tokens', 13, '--graph', '--prompt', 'A', '--lr', .0003, '--seed', 1337,
              *(['--fast'] if math == 'tf32' else []))
        setup('live', '--resume', base, '--curriculum', schedule, '--out', fixture / f'{math}-control',
              '--updates', 112, '--prompt', 'A')
        base_hash = sha(base)
        directories = {}
        for mode in ('none', 'uniform', 'interference'):
            root = out / f'{math}-{mode}'
            native('run', '--checkpoint', base, '--curriculum', schedule, '--out', root,
                   '--updates', 112, '--mode', mode, '--prompt', 'A', '--audit-state', 1)
            split = out / f'{math}-{mode}-split'
            native('run', '--checkpoint', base, '--curriculum', schedule, '--out', split,
                   '--updates', 79, '--mode', mode, '--prompt', 'A')
            resumed = out / f'{math}-{mode}-resumed'
            native('run', '--checkpoint', base, '--resume', split / 'latest.ckpt', '--curriculum', schedule,
                   '--out', resumed, '--updates', 112, '--mode', mode, '--prompt', 'A')
            for suffix in ('latest.ckpt', 'latest.ckpt.sgpriority'):
                assert (root / suffix).read_bytes() == (resumed / suffix).read_bytes()
            assert journal(root / 'selection.jsonl') == journal(split / 'selection.jsonl') + journal(resumed / 'selection.jsonl')
            assert (root / 'speech.txt').read_bytes() == (split / 'speech.txt').read_bytes() + (resumed / 'speech.txt').read_bytes()
            if mode == 'interference':
                assert read(root / 'result.json')['overrides'] > 0
                for row in journal(root / 'selection.jsonl'):
                    values = [c['after'] - c['before'] for c in row['candidates']]
                    assert row['chosen'] == int(len(values) == 2 and values[1] > max(0, values[0]))
                    assert len({c['length'] for c in row['candidates']}) == 1
            else:
                assert sha(root / 'latest.ckpt') == sha(fixture / f'{math}-control/latest.ckpt')
                assert (root / 'speech.txt').read_bytes() == speech_bytes(fixture / f'{math}-control/transcript.txt')
            directories[mode] = root
        assert sha(base) == base_hash
        results.append(dict(math=math, complete_restart_exact=True, uniform_controls_exact=True,
                            positive_interference_overrides=read(directories['interference'] / 'result.json')['overrides']))
    base = fixture / 'fp32-base/latest.ckpt'
    timed = out / 'timed'
    native('run', '--checkpoint', base, '--curriculum', schedule, '--out', timed,
           '--updates', 112, '--seconds', .01, '--mode', 'interference', '--prompt', 'A')
    clocked = read(timed / 'result.json')
    assert 48 < clocked['end'] < 112 and clocked['live_seconds'] >= .01
    assert clocked['live_seconds'] < .01 + clocked['max_tick_ms'] / 1000 + .003
    continued = out / 'timed-resumed'
    native('run', '--checkpoint', base, '--resume', timed / 'latest.ckpt', '--curriculum', schedule,
           '--out', continued, '--updates', 112, '--mode', 'interference', '--prompt', 'A')
    assert sha(continued / 'latest.ckpt') == sha(out / 'fp32-interference/latest.ckpt')
    assert journal(timed / 'selection.jsonl') + journal(continued / 'selection.jsonl') == journal(out / 'fp32-interference/selection.jsonl')
    resume = out / 'fp32-interference-split/latest.ckpt'
    cases = [('changed-mode', ['--mode', 'uniform']), ('changed-seed', ['--seed', '7']),
             ('missing-sidecar', ['--resume', base]), ('earlier-end', ['--updates', '30'])]
    corrupt = out / 'corrupt.ckpt'
    shutil.copyfile(resume, corrupt)
    sidecar = Path(str(corrupt) + '.sgpriority')
    raw = bytearray(Path(str(resume) + '.sgpriority').read_bytes())
    raw[16] ^= 1
    sidecar.write_bytes(raw)
    cases.append(('corrupt-sidecar', ['--resume', corrupt]))
    for name, replacement in cases:
        options = dict(checkpoint=base, resume=resume, curriculum=schedule, out=out / name,
                       updates=112, mode='interference', prompt='A')
        options.update({k[2:]: v for k, v in zip(replacement[::2], replacement[1::2])})
        cmd = [str(exe), 'run']
        for k, v in options.items():
            cmd += ['--' + k, str(v)]
        result = subprocess.run(cmd, capture_output=True)
        (out / (name + '.log')).write_bytes(result.stdout + result.stderr)
        assert result.returncode and not (out / name).exists(), name
    write(out / 'result.json', dict(passed=True, executable_sha256=sha(exe), native_commands=len(native.commands),
                                  setup_commands=len(setup.commands), native_reference_sha256=sha(production),
                                  wall_budget_stops_after_complete_tick=True, exact_resume_after_time_stop=True,
                                  cases=results, rejected_policies=[n for n, _ in cases]))
    print('Replay selector: exact old control, complete restart, actual overrides and policy rejections passed.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exe', type=Path, default=Path('build/replay-selection-probe/synaptic-replay-selection-probe.exe'))
    parser.add_argument('--native', type=Path, default=Path('build/pre-replay-priority/synapticgenesis.exe'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    check(args.exe, args.native, args.out)
