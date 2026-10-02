"""Exercise the real training CLI's prefix policy and checkpoint continuation.

Synthetic corpus and checkpoints are disposable; no book model is modified.
Uses the standard library only. No training or gradient computation in Python.
"""
import argparse
import json
from pathlib import Path
import struct
import subprocess


def read_checkpoint(path):
    data = path.read_bytes()
    meta = struct.unpack_from('<32Q', data)
    hp = struct.unpack_from('<8f', data, 256)
    params = struct.unpack_from(f'<{3*meta[14]}f', data, 288)
    return meta, hp, params


def check(exe, output):
    exe = exe.resolve()
    output.mkdir(parents=True, exist_ok=False)
    train = output / 'train.dat'
    val = output / 'validation.dat'
    train.write_bytes(b'The child counts seeds. The child closes the gate. ' * 32)
    val.write_bytes(b'A child speaks. A teacher brings a letter. ' * 24)
    common = ['--data', str(train), '--validation', str(val), '--eval-every', '4',
              '--eval-batches', '2', '--save-every', '4']

    def run(destination, flags):
        command = [str(exe), 'train', *common, '--out', str(destination), *flags]
        result = subprocess.run(command, capture_output=True, text=True)
        (output / (destination.name + '.log')).write_text(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError(f'{destination.name}: {result.stdout}\n{result.stderr}')
        return read_checkpoint(destination / 'latest.ckpt')

    results = []
    for cell, version in [('lif', 1), ('alif', 2)]:
        for policy in ('warm', 'reset'):
            name = f'{cell}-{policy}'
            first = output / name
            meta, _, _ = run(first, ['--steps', '4', '--warmup', '0', '--batch', '2',
                                    '--context', '16', '--channels', '32', '--hidden', '64',
                                    '--layers', '2', '--cell', cell, '--burn-in', '32',
                                    '--burn-policy', policy, '--seed', '17'])
            assert meta[1] == version and meta[19:21] == (32, int(policy == 'reset'))
            resume = ['--resume', str(first / 'latest.ckpt'), '--steps', '8']
            inherited = run(output / (name + '-inherited'), resume)
            explicit = run(output / (name + '-explicit'),
                           [*resume, '--burn-in', '32', '--burn-policy', policy])
            # Hash values may differ with atomic accumulation order; compare the
            # actual weights/moments and all other metadata with a tight bound.
            im, ih, ip = inherited
            em, eh, ep = explicit
            assert im[:15] + im[16:] == em[:15] + em[16:]
            error = max(abs(a-b) for a, b in zip(ip, ep))
            assert error < 3e-5 and max(abs(a-b) for a, b in zip(ih, eh)) < 3e-5
            switched_policy = 'reset' if policy == 'warm' else 'warm'
            sm, _, sp = run(output / (name + '-switched'),
                            [*resume, '--burn-policy', switched_policy])
            assert sm[19:21] == (32, int(switched_policy == 'reset'))
            effect = max(abs(a-b) for a, b in zip(ip[:im[14]], sp[:im[14]]))
            assert effect > 1e-6, 'Policy change did not affect learning'
            results.append({'cell': cell, 'policy': policy, 'resume_max_abs_error': error,
                            'switched_policy_weight_max_abs_difference': effect})
    report = {'passed': True, 'scope': 'CLI policy persistence and explicit overrides on tiny synthetic models',
              'pairs': results, 'continuous_schedule_equivalence_tested': False}
    (output / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exe', type=Path, default=Path('build/synapticgenesis.exe'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    check(args.exe, args.out)
