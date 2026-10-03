"""Replay-score isolation against the preserved native executable and CPU math."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from experiment_checkpoint import checkpoint, state_record
from native_experiment import read, sha, write
from probes_cli import Reference


def speech_bytes(path):
    # Generation sanitizes control bytes. Strip only the native log headers.
    raw = path.read_bytes()
    return re.sub(rb'\n\[(?:session starts at online update \d+|online update \d+; global update \d+)\]\n', b'', raw)


def journal(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def cpu_scores(before, after, docs, rows, answer_documents):
    """Check every candidate around one measured update; never alter training."""
    errors = []
    torch.set_num_threads(1)
    for phase, snapshot in [('before', before), ('after', after)]:
        reference = Reference(snapshot)
        for row in rows:
            doc = docs[row['document']]
            offset, length = row['offset'], row['length']
            sample = doc[offset:offset + length + 1]
            assert len(sample) == length + 1
            with torch.no_grad():
                logp = reference.logits(sample[:-1]).log_softmax(-1)
            losses = [-float(logp[i, byte]) for i, byte in enumerate(sample[1:])]
            answer_start = (doc.index(b'\nAnswer: ') + len(b'\nAnswer: ')
                            if row['document'] in answer_documents else len(doc) + 1)
            weights = [64 if offset + i + 1 >= answer_start else 1 for i in range(length)]
            answer = [loss for loss, weight in zip(losses, weights) if weight > 1]
            expected = dict(mean=sum(losses) / length,
                            weighted=sum(v * w for v, w in zip(losses, weights)) / sum(weights),
                            answer=sum(answer) / len(answer) if answer else None)
            assert row[phase]['answer_targets'] == len(answer)
            error = max(abs(expected[k] - row[phase][k]) for k in expected if expected[k] is not None)
            errors.append(dict(phase=phase, document=row['document'], offset=offset,
                               length=length, max_abs_error=error, passed=error < 3e-5))
    return dict(passed=all(r['passed'] for r in errors), tolerance=3e-5,
                max_abs_error=max(r['max_abs_error'] for r in errors), checks=errors)


def check(native, probe, out):
    native, probe, out = native.resolve(), probe.resolve(), out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    docs = [b'The bird sings.', b'A.', b'The key is in the box.\nAnswer: box.',
            b'The cup stands on the shelf.\nAnswer: shelf.', b'Rain fills a stream beside the garden.']
    for index, count in enumerate((2, 4, 5), 1):
        (out / f'{index}.dat').write_bytes(b'\x1e'.join(docs[:count]))
    schedule = out / 'curriculum.sg'
    schedule.write_text('SGCURRICULUM3\n12 "1.dat" 1 all 1\n32 "2.dat" .5 new 64\n80 "3.dat" .25 new 1\n')
    sources = {str(p): sha(p) for p in [schedule, *(out / f'{i}.dat' for i in (1, 2, 3))]}
    calls, rejected, cases = [], [], []

    def command(exe, arguments, name, failure=False):
        argv = [str(exe), *map(str, arguments)]
        calls.append(argv)
        write(out / 'commands.json', calls)
        result = subprocess.run(argv, capture_output=True)
        (out / (name + '.log')).write_bytes(result.stdout + result.stderr)
        assert (result.returncode != 0) == failure, (name, result.stderr.decode(errors='replace'))

    for fast in (False, True):
        label = 'tf32' if fast else 'fp32'
        base = out / (label + '-base')
        command(native, ['live', '--curriculum', schedule, '--out', base, '--channels', 16,
                        '--hidden', 24, '--layers', 2, '--cell', 'associative', '--chunk', 16,
                        '--updates', 24, '--replay', 'stage', '--replay-capacity', 64,
                        '--replay-every', 2, '--speak-every', 8, '--tokens', 13, '--graph',
                        '--prompt', 'A', '--lr', .0003, '--seed', 1337,
                        *(['--fast'] if fast else [])], label + '-base')
        ancestor = base / 'latest.ckpt'
        original = sha(ancestor)
        control = out / (label + '-control')
        command(native, ['live', '--curriculum', schedule, '--resume', ancestor, '--out', control,
                        '--updates', 56, '--prompt', 'A'], label + '-control')

        def run(name, source=ancestor, end=56, every=4, flags=()):
            destination = out / (label + '-' + name)
            command(probe, ['run', '--checkpoint', source, '--curriculum', schedule, '--out', destination,
                            '--updates', end, '--every', every, '--per-group', 64, '--seed', 42,
                            '--prompt', 'A', *flags], label + '-' + name)
            return destination

        disabled = run('disabled', every=0)
        measured = run('measured', flags=['--audit-state', '1', '--snapshots', '1'])
        reverse = run('reverse', flags=['--reverse-candidates', '1', '--audit-state', '1'])
        split = run('split', end=39)
        resumed = run('resumed', source=split / 'latest.ckpt')
        for directory in (disabled, measured, reverse, resumed):
            assert sha(directory / 'latest.ckpt') == sha(control / 'latest.ckpt'), directory
        expected_speech = speech_bytes(control / 'transcript.txt')
        assert expected_speech and (measured / 'speech.txt').read_bytes() == expected_speech
        assert (disabled / 'speech.txt').read_bytes() == expected_speech
        assert (reverse / 'speech.txt').read_bytes() == expected_speech
        assert (split / 'speech.txt').read_bytes() + (resumed / 'speech.txt').read_bytes() == expected_speech
        scores = journal(measured / 'scores.jsonl')
        assert scores == journal(split / 'scores.jsonl') + journal(resumed / 'scores.jsonl')
        reversed_scores = journal(reverse / 'scores.jsonl')
        for row in reversed_scores:
            row['candidates'].reverse()
        assert scores == reversed_scores
        assert [r['source_observation'] for r in scores] == list(range(25, 57, 4))
        assert len({r['length'] for event in scores for r in event['candidates']}) >= 3
        assert any(r['length'] == 1 for event in scores for r in event['candidates'])
        assert any(r['before']['answer_targets'] for event in scores for r in event['candidates'])
        assert len(state_record(measured / 'latest.ckpt')['replay_groups']) == 3
        meta_before = checkpoint(measured / 'score-before.ckpt')[0]
        meta_after = checkpoint(measured / 'score-after.ckpt')[0]
        assert meta_after[7] == meta_before[7] + 1 and meta_before[17] == meta_after[17] == 0
        oracle = cpu_scores(measured / 'score-before.ckpt', measured / 'score-after.ckpt',
                            docs, scores[0]['candidates'], {2, 3})
        write(measured / 'cpu-oracle.json', oracle)
        assert oracle['passed'], oracle
        assert sha(ancestor) == original
        cases.append(dict(learning_tf32=fast, exact_old_executable_checkpoint=True,
                          exact_speech=True, exact_restart=True, exact_reversed_scores=True,
                          oracle=oracle, result=read(measured / 'result.json')))

    for name, changes in [('old-end', ['--updates', '24']), ('excess-end', ['--updates', '81']),
                          ('bad-prompt', ['--prompt', 'B']), ('bad-cadence', ['--every', '-1']),
                          ('bad-pool', ['--per-group', '65']), ('bad-switch', ['--snapshots', '2']),
                          ('nonlive', ['--checkpoint', measured / 'score-before.ckpt'])]:
        options = dict(checkpoint=ancestor, curriculum=schedule, out=out / name, updates=56,
                       every=4, **{'per-group': 4}, prompt='A')
        for flag, value in zip(changes[::2], changes[1::2]):
            options[flag[2:]] = value
        args = ['run']
        for key, value in options.items():
            args += ['--' + key, value]
        command(probe, args, name, failure=True)
        assert not (out / name).exists()
        rejected.append(name)
    assert all(sha(p) == identity for p, identity in sources.items())
    result = dict(passed=True, native_sha256=sha(native), probe_sha256=sha(probe),
                  native_commands=len(calls), cases=cases, rejected_before_output=rejected,
                  inputs_unchanged=True, scope='Synthetic isolation and scoring tests, not language quality.')
    write(out / 'result.json', result)
    print(f'{len(calls)} native calls pass; exact checkpoint, speech, score-order and restart controls; CPU scores pass.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--native', type=Path, default=Path('build/pre-replay-priority/synapticgenesis.exe'))
    parser.add_argument('--probe', type=Path, default=Path('build/replay-priority-probe/synaptic-replay-priority-probe.exe'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    check(args.native, args.probe, args.out)
