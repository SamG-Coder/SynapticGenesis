"""Resume the same pre-binding live state at three plasticity rates.

The native runtime retains replay, optimizer history, speech and source cursor.
Python only orchestrates experiments and records results.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess


def run(exe, out):
    exe = exe.resolve()
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    for cell in ('lif', 'trace'):
        source = Path(f'runs/binding-{cell}-1337/stage-2.ckpt')
        raw = source.read_bytes()
        meta = struct.unpack_from('<32Q', raw)
        assert meta[24] == 10000 and meta[17] == 4
        source_hash = hashlib.sha256(raw).hexdigest()
        for base_rate in (.0003, .0012, .004):
            dest = out / f'{cell}-{base_rate:g}'
            command = [str(exe), 'live', '--resume', str(source),
                       '--curriculum', 'data/prepared/binding-v2/curriculum.sg',
                       '--out', str(dest), '--updates', '34000', '--lr', str(base_rate),
                       '--prompt', 'The bird ', '--validation', 'data/prepared/development-v2-final/13853.txt',
                       '--eval-batches', '32', '--log-every', '6000']
            p = subprocess.run(command, capture_output=True)
            (out/f'{dest.name}.log').write_bytes(p.stdout+p.stderr)
            if p.returncode:
                raise RuntimeError(f'{dest.name}: {p.stderr}')
            report = dict(cell=cell, base_rate=base_rate, actual_binding_rate=base_rate*.25,
                          source_checkpoint=source.as_posix(), source_sha256=source_hash,
                          checkpoint_sha256=hashlib.sha256((dest/'latest.ckpt').read_bytes()).hexdigest(),
                          session=json.loads((dest/'session.json').read_text()))
            for split in ('train', 'development'):
                p = subprocess.run([str(exe), 'language-probes', '--checkpoint', str(dest/'latest.ckpt'),
                                    '--probes', f'data/prepared/binding-v2/{split}.sgprobe',
                                    '--output', str(dest/f'{split}.json')], capture_output=True)
                if p.returncode:
                    raise RuntimeError(f'Probes {dest.name}: {p.stderr}')
                report[split] = {k:v for k,v in json.loads((dest/f'{split}.json').read_text()).items() if k!='results'}
            assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash
            rows.append(report)
            print(f'{cell} rate={base_rate*.25:g}: train={report["train"]["joint_accuracy"]:.4f} '
                  f'dev={report["development"]["joint_accuracy"]:.4f} reader={report["session"]["final_validation_loss"]:.4f}', flush=True)
            (out/'partial.json').write_text(json.dumps(rows, indent=2)+'\n')
    for name in ('observed_pairs', 'replay_pairs', 'replay_updates', 'generated_bytes', 'global_updates'):
        assert len({r['session'][name] for r in rows}) == 1, name
    result = dict(protocol='Fixed 24,000 binding observations and 6,000 replay updates; rate is the only changed policy.',
                  seed=1337, checkpoint_before_binding_used_for_all_rates=True,
                  original_curriculum_identity_preserved=True,
                  no_test_evaluation=True, same_source_and_replay_exposure=True,
                  limits='Exploratory one-seed rate comparison, not a generalization estimate or architecture ranking.',
                  runs=rows)
    (out/'comparison.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--exe', type=Path, default=Path('build/synapticgenesis.exe'))
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    run(a.exe, a.out)
