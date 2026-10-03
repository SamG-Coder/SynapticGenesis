"""Bounded actual-update replay diagnostic on explicitly admitted ancestors."""
import argparse
from pathlib import Path
import subprocess

from corpus.selection import require_training_spec, selection
from experiment_checkpoint import state_record
from extend_curriculum import read_schedule
from native_experiment import NativeCommands, read, sha, write


SOURCE_FILES = ['src/live.cuh', 'experiments/replay_priority_probe.cu',
                'experiments/replay_priority_scoring.cuh', 'scripts/replay_priority_experiment.py',
                'tests/replay_priority_probe.py', 'tests/replay_priority_experiment.py',
                'scripts/summarize_replay_priority.py']


def admitted_inputs():
    policy = selection()
    for edition in policy['active_editions'].values():
        require_training_spec(edition['sources'])
    schedule = Path(policy['continuation_schedule']).resolve()
    assert sha(schedule) == policy['continuation_schedule_sha256']
    _, stages = read_schedule(schedule)
    history = read('reports/teacher-retention-language.json')
    assert len(stages) == 4
    identities = {str(schedule): sha(schedule), 'data/training-selection.json': sha('data/training-selection.json')}
    for stage, prior in zip(stages, history['protocol']['source_admission']['editions']):
        assert sha(stage['source']) == prior['sha256']
        identities[str(stage['source'])] = prior['sha256']
    bases = policy['diagnostic_bases']
    assert [r['seed'] for r in bases] == [1337, 2026, 31415]
    for base in bases:
        assert sha(base['checkpoint']) == base['sha256']
        prior = next(r for r in history['runs'] if r['seed'] == base['seed']
                     and r['arm'] == 'associative-control' and r['online_updates'] == base['online_updates'])
        assert prior['checkpoint_sha256'] == base['sha256'] and base['online_updates'] == 160000
        identities[base['checkpoint']] = base['sha256']
    return schedule, bases, identities


def run(args):
    schedule, bases, identities = admitted_inputs()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    executables = {k: Path(getattr(args, k)).resolve() for k in ('native', 'original', 'probe')}
    identities.update({str(p): sha(p) for p in executables.values()})
    protocol = dict(status='declared_before_execution', start=160000, end=160512,
        rounds=2, observe_every=32, candidates_per_group=8, candidate_seed=42,
        schedule=str(schedule), bases=bases, authenticated_inputs=identities,
        source_sha256={p: sha(p) for p in SOURCE_FILES},
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        hardware=subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version,memory.total',
                                         '--format=csv,noheader'], text=True).strip(),
        other_gpu_contexts_active=True, reserved_tests_scored=False, generated_text_targets=False,
        priority_replay_enabled=False, learning_math='Inherited TF32; existing source then uniform stage replay',
        scoring_math='Strict FP32, reset-state windows, before and after the actual source AdamW update',
        score_scope='Previously stored training windows only; score before scheduled replay and reservoir admission',
        timing_scope='Probe live loop includes native learning, replay, scheduled speech and scoring/journal work. '
                     'Excludes process startup, initial model/source load and final checkpoint save. '
                     'Each process lazily creates its views; no learned-run snapshots or state-hash audits in timed arms.',
        memory_scope='Owned array bytes exclude shared weights/moments and cuBLAS/driver allocations. '
                     'Free-memory deltas include library allocation and desktop variability; not absolute peak VRAM.',
        limitations='Two timed runs per seed on one desktop GPU. Immediate native loss changes do not prove durable '
                    'retention or independent numerical parity. This measures a signal, without changing replay selection.')
    write(out / 'protocol.json', protocol)
    commands = {}
    for kind, exe in executables.items():
        directory = out / (kind + '-commands')
        directory.mkdir()
        commands[kind] = NativeCommands(exe, directory)
    rows = []
    for index, base in enumerate(bases):
        seed, ancestor = base['seed'], Path(base['checkpoint']).resolve()
        expected = None
        for kind in ('original', 'native'):
            destination = out / f'{seed}-{kind}-control'
            commands[kind]('live', '--resume', ancestor, '--curriculum', schedule, '--out', destination,
                           '--updates', protocol['end'], '--prompt', 'The bird ',
                           '--log-every', 4096, '--save-every', 4096)
            actual = sha(destination / 'latest.ckpt')
            if expected is None:
                expected = actual
            assert actual == expected, 'Production executable changes the learned trajectory'
        for repeat in range(protocol['rounds']):
            order = ['disabled', 'measured'] if (index + repeat) % 2 == 0 else ['measured', 'disabled']
            for mode in order:
                destination = out / f'{seed}-{repeat}-{mode}'
                commands['probe']('run', '--checkpoint', ancestor, '--curriculum', schedule, '--out', destination,
                                  '--updates', protocol['end'], '--every', 32 if mode == 'measured' else 0,
                                  '--per-group', 8, '--seed', 42, '--prompt', 'The bird ')
                assert sha(destination / 'latest.ckpt') == expected, 'Measurement changes the learned trajectory'
                row = dict(seed=seed, repeat=repeat, mode=mode, directory=destination.name,
                           result=read(destination / 'result.json'), final=state_record(destination / 'latest.ckpt'),
                           scores_sha256=sha(destination / 'scores.jsonl'))
                rows.append(row)
                write(out / 'runs.json', rows)
                print(seed, repeat, mode, row['result']['live_seconds'], 'seconds', flush=True)
        # A separate first-update snapshot supplies independent CPU verification.
        destination = out / f'{seed}-oracle'
        commands['probe']('run', '--checkpoint', ancestor, '--curriculum', schedule, '--out', destination,
                          '--updates', 160001, '--every', 32, '--per-group', 8, '--seed', 42,
                          '--prompt', 'The bird ', '--snapshots', 1, '--audit-state', 1)
    assert all(sha(p) == value for p, value in identities.items())
    assert all(sha(p) == value for p, value in protocol['source_sha256'].items())
    result = dict(complete=True, protocol=protocol, runs=rows, inputs_unchanged=True,
                  commands={k: len(v.commands) for k, v in commands.items()},
                  exact_original_native_and_probe_checkpoints=True)
    write(out / 'comparison.json', result)
    print('Completed; execution audit and CPU scoring checks are separate.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native', type=Path, default=Path('build/synapticgenesis.exe'))
    parser.add_argument('--original', type=Path, default=Path('build/pre-replay-priority/synapticgenesis.exe'))
    parser.add_argument('--probe', type=Path, default=Path('build/replay-priority-probe/synaptic-replay-priority-probe.exe'))
    parser.add_argument('--out', type=Path, required=True)
    run(parser.parse_args())
