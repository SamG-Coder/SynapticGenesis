"""Matched native read-gate versus retention-gate language learning and exposure.

All weights start from the same random values. The existing selected binding-v2
material is unchanged; only its final exposure limit is declared longer at birth.
Python orchestrates the native learner and reads results, never trains a model.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import struct
import subprocess

from audit_binding import audit
from extend_curriculum import read_schedule
from prepare_lessons import quoted


def run(exe, out, seed=1337):
    exe = exe.resolve()
    out.mkdir(parents=True, exist_ok=False)
    selected = Path('data/prepared/binding-v2')
    manifest = json.loads((selected / 'manifest.json').read_text())
    for name, record in manifest['files'].items():
        assert hashlib.sha256((selected / name).read_bytes()).hexdigest() == record['sha256'], name
    _, stages = read_schedule(selected / 'curriculum.sg')
    assert [s['end_update'] for s in stages] == [6000, 10000, 34000]
    stages[-1]['end_update'] = 130000
    schedule = out / 'curriculum.sg'
    schedule.write_bytes(('SGCURRICULUM3\n' + '\n'.join(
        f'{s["end_update"]} {quoted(s["source"].as_posix())} {s["rate"]} {s["scope"]} {s["answer"]}'
        for s in stages) + '\n').encode())
    calls, rows = 0, []

    def native(*args):
        nonlocal calls
        calls += 1
        result = subprocess.run([str(exe), *map(str, args)], capture_output=True)
        (out / f'command-{calls:02d}.log').write_bytes(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError(f'{args}: {result.stdout}\n{result.stderr}')

    for index, end in enumerate((34000, 67000, 130000)):
        cells = ('gated', 'selective') if index % 2 == 0 else ('selective', 'gated')
        for cell in cells:
            dest = out / cell
            if index:
                initialization = ['--resume', dest / 'latest.ckpt']
            else:
                initialization = ['--cell', cell, '--seed', seed, '--lr', .0003, '--chunk', 128,
                                  '--replay', 'reservoir', '--replay-capacity', 1024, '--replay-every', 4,
                                  '--graph', '--speak-every', 500, '--tokens', 96, '--fast']
            native('live', '--curriculum', schedule, '--out', dest, '--updates', end, *initialization,
                   '--prompt', 'The bird ', '--validation', 'data/prepared/development-v2-final/13853.txt',
                   '--eval-batches', 32, '--log-every', 6000, '--save-every', 5000)
            snapshot = dest / f'checkpoint-{end}.ckpt'
            shutil.copyfile(dest / 'latest.ckpt', snapshot)
            raw = snapshot.read_bytes()
            meta = struct.unpack_from('<32Q', raw)
            row = dict(cell=cell, seed=seed, online_updates=end, parameters=meta[14],
                       checkpoint_sha256=hashlib.sha256(raw).hexdigest(),
                       session=json.loads((dest / 'session.json').read_text()))
            for split in ('train', 'development'):
                report = dest / f'{split}-{end}.json'
                native('language-probes', '--checkpoint', snapshot, '--probes', selected / f'{split}.sgprobe',
                       '--output', report)
                row[split] = {k: v for k, v in json.loads(report.read_text()).items() if k != 'results'}
                row[split + '_audit'] = audit(report, split)
            rows.append(row)
            print(f'{cell} online={end}: train joint={row["train"]["joint_accuracy"]:.4f}, '
                  f'dev joint={row["development"]["joint_accuracy"]:.4f}, '
                  f'reader={row["session"]["final_validation_loss"]:.4f}', flush=True)
            (out / 'partial.json').write_text(json.dumps(rows, indent=2) + '\n')
    initial = [(out / cell / 'initial.ckpt').read_bytes() for cell in ('gated', 'selective')]
    # Architecture IDs and checksums differ; all serialized arrays/policies and
    # numerical hyperparameters must be byte-identical before the first update.
    assert initial[0][256:] == initial[1][256:]
    common_initial_sha = hashlib.sha256(initial[0][256:]).hexdigest()
    for end in (34000, 67000, 130000):
        selected_rows = [r for r in rows if r['online_updates'] == end]
        assert len(selected_rows) == 2
        for key in ('online_updates', 'global_updates', 'observed_pairs', 'replay_updates', 'replay_pairs',
                    'generated_bytes', 'learning_rate', 'curriculum_hash'):
            assert selected_rows[0]['session'][key] == selected_rows[1]['session'][key], (end, key)
        assert selected_rows[0]['parameters'] == selected_rows[1]['parameters'] == 1716736
    decode = {}
    for cell in ('gated', 'selective'):
        dest = out / cell
        native('decode-bench', '--checkpoint', dest / 'latest.ckpt', '--tokens', 1024,
               '--rounds', 7, '--out', dest / 'decode')
        decode[cell] = json.loads((dest / 'decode/benchmark.json').read_text())
    result = dict(protocol='docs/selective-trace.md', seed=seed, native_commands=calls,
                  source_manifest_sha256=hashlib.sha256((selected / 'manifest.json').read_bytes()).hexdigest(),
                  source_spec_sha256=manifest['spec_sha256'], selected_material_unchanged=True,
                  final_binding_exposure_limit=120000, measurements_at_online_updates=[34000, 67000, 130000],
                  common_initial_arrays_and_policy_sha256=common_initial_sha,
                  initial_parameters_and_state_identical=True, matched_parameter_count=True,
                  same_source_and_replay_exposure=True, reserved_test_evaluated=False,
                  executable_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),
                  runs=rows, decode=decode,
                  limits='Paired one-seed architecture/exposure experiment. Independent probes use '
                         'training and development partitions; no general dialogue or biological mechanism claim.')
    (out / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--exe', type=Path, default=Path('build/synapticgenesis.exe'))
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--seed', type=int, default=1337)
    a = p.parse_args()
    run(a.exe, a.out, a.seed)
