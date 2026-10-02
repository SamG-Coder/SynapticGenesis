"""Matched native delayed-cue experiments; no Python model computation.

All models are disposable synthetic controls, never language-model ancestors.
The fixed protocol compares three cells over three seeds and three delays.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess


def run(exe, out):
    exe = exe.resolve()
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    for seed in (1337, 2026, 31415):
        for delay in (64, 128, 256):
            # Rotate the first architecture to limit systematic ordering bias.
            cells = ['lif', 'alif', 'trace']
            shift = (1337, 2026, 31415).index(seed)
            cells = cells[shift:] + cells[:shift]
            for cell in cells:
                dest = out / f'{cell}-{seed}-{delay}'
                command = [str(exe), 'memory-bench', '--cell', cell, '--delay', str(delay),
                           '--steps', '2000', '--seed', str(seed), '--batch', '32',
                           '--channels', '64', '--hidden', '128', '--layers', '2',
                           '--lr', '.001', '--fast', '--out', str(dest)]
                p = subprocess.run(command, capture_output=True)
                (out / f'{dest.name}.log').write_bytes(p.stdout + p.stderr)
                if p.returncode:
                    raise RuntimeError(f'{dest.name}: {p.stderr.decode(errors="replace")}')
                row = json.loads((dest / 'result.json').read_text())
                assert row['seed'] == seed and row['delay'] == delay and row['steps'] == 2000
                assert row['all_state_erased_accuracy'] == .5 and row['synthetic_only']
                row['checkpoint_sha256'] = hashlib.sha256((dest / 'synthetic-only.ckpt').read_bytes()).hexdigest()
                row['cell_option'] = cell
                rows.append(row)
                print(f'{cell} seed={seed} delay={delay}: accuracy={row["accuracy"]:.4f}', flush=True)
    means = []
    for delay in (64, 128, 256):
        for cell in ('lif', 'alif', 'trace'):
            selected = [r for r in rows if r['delay'] == delay and r['cell_option'] == cell]
            means.append(dict(cell=cell, delay=delay,
                              mean_accuracy=statistics.mean(r['accuracy'] for r in selected),
                              min_accuracy=min(r['accuracy'] for r in selected),
                              max_accuracy=max(r['accuracy'] for r in selected),
                              mean_training_seconds=statistics.mean(r['training_seconds'] for r in selected)))
    report = dict(protocol='Matched native delayed A/B cue recall; fresh random initialization per run.',
                  seeds=[1337, 2026, 31415], delays=[64, 128, 256], steps=2000,
                  channels=64, hidden=128, layers=2, batch=32, learning_rate=.001,
                  fast_math=True, evaluation_sequences_per_run=2048,
                  exact_common_initial_weights_verified_by_native_cell_tests=True,
                  ordered_sequentially_with_rotating_cell_order=True,
                  same_examples_within_each_seed_and_delay=True,
                  all_state_erased_accuracy=.5, language_training_used=False,
                  executable_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),
                  means=means, runs=rows,
                  limitations='Synthetic one-bit recall, not language understanding. Three seeds. '
                              'Training time excludes model setup and evaluation; no energy claim.')
    (out / 'comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(means, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--exe', type=Path, default=Path('build/synapticgenesis.exe'))
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    run(a.exe, a.out)
