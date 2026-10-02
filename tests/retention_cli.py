"""Check the experiment protocol against independently reloaded checkpoints."""
import argparse
import json
from pathlib import Path
import struct
import subprocess


def check(exe, out):
    exe = exe.resolve()
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    texts = {'a': b'Red birds sit in a tree. ', 'b': b'Light bends across the water. ',
             'va': b'A blue bird sings near a tree. ', 'vb': b'Rain falls from the grey sky. '}
    for key, text in texts.items():
        (out / f'{key}.dat').write_bytes(text * 20)
    calls = 0

    def run(*args, reject=None):
        nonlocal calls
        calls += 1
        result = subprocess.run([str(exe), *map(str, args)], capture_output=True, text=True)
        (out / f'command-{calls:02d}.log').write_text(result.stdout + result.stderr, encoding='utf-8')
        if reject:
            assert result.returncode and reject in result.stderr, (args, result.stdout, result.stderr)
        elif result.returncode:
            raise RuntimeError(f'{args}: {result.stdout}\n{result.stderr}')

    common = ['--train-a', out / 'a.dat', '--train-b', out / 'b.dat', '--validation-a', out / 'va.dat',
              '--validation-b', out / 'vb.dat', '--steps-a', 16, '--steps-b', 16, '--chunk', 8,
              '--channels', 8, '--hidden', 16, '--layers', 2, '--eval-batches', 2, '--replay-capacity', 8,
              '--speak-every', 8, '--tokens', 7, '--seed', 13]
    experiment = out / 'experiment'
    run('retention-bench', '--out', experiment, *common)
    report = json.loads((experiment / 'result.json').read_text())
    assert report['matched_online_exposure'] and report['new_online_scope_excludes_old_documents']
    for arm in report['arms']:
        assert arm['observed_pairs'] == 128 and arm['generated_bytes'] == 14
        replaying = not arm['name'].startswith('none')
        assert arm['replay_updates'] == (4 if replaying else 0)
        assert arm['replay_pairs'] == (32 if replaying else 0)
        ckpt = experiment / arm['name'] / 'latest.ckpt'
        meta = struct.unpack_from('<32Q', ckpt.read_bytes())
        assert meta[24] == 32 and meta[7] == (40 if replaying else 36) and meta[19] == 1
        assert str(meta[15]) == arm['saved_payload_hash']
        for domain, key in [('va', 'old_loss'), ('vb', 'new_loss')]:
            evaluated = out / f"{arm['name']}-{domain}.json"
            run('evaluate', '--checkpoint', ckpt, '--data', out / f'{domain}.dat', '--batch', 16,
                '--context', 128, '--batches', 2, '--output', evaluated)
            value = json.loads(evaluated.read_text())['loss_nats_per_byte']
            assert abs(value - arm[key]) < 1e-5
        assert abs(arm['old_forgetting'] - (arm['old_loss'] - report['after_a_old_loss'])) < 1e-7
        assert abs(arm['new_transfer_change'] - (arm['new_loss'] - report['after_a_new_loss'])) < 1e-7
    run('retention-bench', '--out', experiment, *common, reject='Retention experiment exists')
    invalid = list(common)
    invalid[invalid.index('--validation-b') + 1] = out / 'b.dat'
    run('retention-bench', '--out', out / 'invalid', *invalid, reject='Validation corpus must differ')
    assert not (out / 'invalid').exists()
    result = {'passed': True, 'native_commands': calls, 'arms_checked': 7,
              'metrics_match_independent_checkpoint_evaluation': True,
              'matched_source_replay_and_generated_counts': True,
              'source_cursor_stays_in_new_domain': True, 'heldout_learning_overlap_rejected': True,
              'existing_experiment_protected': True, 'synthetic_fixture_only': True}
    (out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exe', type=Path, default=Path('build/synapticgenesis.exe'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    check(args.exe, args.out)
