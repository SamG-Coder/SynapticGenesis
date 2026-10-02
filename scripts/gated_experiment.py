"""Train the input-gated trace at matched width and near-matched parameter count."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess

from audit_binding import audit


def run(exe, out):
    exe = exe.resolve()
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    for cell, hidden in [('gated',344), ('gated',512), ('trace',768)]:
        dest = out / f'{cell}-{hidden}'
        command = [str(exe), 'live', '--curriculum', 'data/prepared/binding-v2/curriculum.sg',
                   '--out', str(dest), '--cell', cell, '--hidden', str(hidden), '--seed', '1337',
                   '--lr', '.0003', '--chunk', '128', '--replay', 'reservoir',
                   '--replay-capacity', '1024', '--replay-every', '4', '--graph',
                   '--speak-every', '500', '--tokens', '96', '--prompt', 'The bird ', '--fast',
                   '--validation', 'data/prepared/development-v2-final/13853.txt',
                   '--eval-batches', '32', '--log-every', '6000']
        p = subprocess.run(command, capture_output=True)
        (out/f'{dest.name}.log').write_bytes(p.stdout+p.stderr)
        if p.returncode:
            raise RuntimeError(f'{dest.name}: {p.stderr}')
        raw = (dest/'latest.ckpt').read_bytes()
        meta = struct.unpack_from('<32Q',raw)
        row = dict(hidden=hidden, parameters=meta[14], cell=meta[1], cell_option=cell, seed=1337,
                   checkpoint_sha256=hashlib.sha256(raw).hexdigest(),
                   session=json.loads((dest/'session.json').read_text()))
        for split in ('train', 'development'):
            p = subprocess.run([str(exe), 'language-probes', '--checkpoint', str(dest/'latest.ckpt'),
                                '--probes', f'data/prepared/binding-v2/{split}.sgprobe',
                                '--output', str(dest/f'{split}.json')], capture_output=True)
            if p.returncode:
                raise RuntimeError(f'Probes {dest.name}: {p.stderr}')
            row[split] = {k:v for k,v in json.loads((dest/f'{split}.json').read_text()).items() if k!='results'}
            row[split+'_audit'] = audit(dest/f'{split}.json', split)
        p = subprocess.run([str(exe), 'decode-bench', '--checkpoint', str(dest/'latest.ckpt'),
                            '--tokens','1024','--rounds','7','--out',str(dest/'decode')], capture_output=True)
        if p.returncode:
            raise RuntimeError(f'Decode {dest.name}: {p.stderr}')
        row['decode'] = json.loads((dest/'decode/benchmark.json').read_text())
        rows.append(row)
        print(f'{cell} H={hidden}, P={meta[14]}: train={row["train"]["joint_accuracy"]:.4f}, '
              f'dev={row["development"]["joint_accuracy"]:.4f}, reader={row["session"]["final_validation_loss"]:.4f}', flush=True)
        (out/'partial.json').write_text(json.dumps(rows,indent=2)+'\n')
    for key in ('observed_pairs','replay_pairs','replay_updates','generated_bytes','global_updates'):
        assert len({r['session'][key] for r in rows}) == 1
    result = dict(seed=1337, protocol='Original binding-v2 live curriculum without changed data or rate.',
                  default_trace_parameters=1190400, matched_width=512, matched_parameter_hidden=344,
                  larger_trace_control_hidden=768,
                  same_source_and_replay_exposure=True, reserved_test_evaluated=False,
                  executable_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(), runs=rows,
                  limits='One seed. H=344 approximates the default trace parameter count; H=512 matches '
                         'default trace width and common initial parameters but adds gate capacity. '
                         'Trace H=768 approximates the larger gated parameter count. Different widths '
                         'change the initialization and number of recurrent states.')
    (out/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--exe',type=Path,default=Path('build/synapticgenesis.exe'))
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    run(a.exe,a.out)
