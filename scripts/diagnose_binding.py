"""Can the existing cells fit one complete selected binding group?

This is an optimization diagnostic on training examples, not a skill evaluation.
All neural computation stays in the native executable.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess

from prepare_binding import examples
from prepare_lessons import write_probes


def run(exe, out):
    exe = exe.resolve()
    out.mkdir(parents=True, exist_ok=False)
    spec_path = Path('data/lessons-binding-v2.json')
    _, _, partitions, _ = examples(json.loads(spec_path.read_text()))
    rows = partitions['train'][:4]
    assert len({r['pair'] for r in rows}) == 1
    docs = [r['context'] + r['query'] + r[f"choice{r['correct']}"] + '\n' for r in rows]
    random.Random(381970).shuffle(docs)
    (out / 'tiny.dat').write_bytes(b'\x1e'.join(s.encode('ascii') for s in docs))
    write_probes(out / 'tiny.sgprobe', rows, 'SGPROBE2')
    (out / 'curriculum.sg').write_bytes(b'SGCURRICULUM3\n6000 "tiny.dat" 1 all 64\n')
    results = []
    for cell in ('lif', 'trace'):
        for rate in (.000075, .0003, .001):
            dest = out / f'{cell}-{rate:g}'
            command = [str(exe), 'live', '--curriculum', str(out/'curriculum.sg'), '--out', str(dest),
                       '--cell', cell, '--seed', '1337', '--lr', str(rate), '--chunk', '128',
                       '--replay', 'none', '--speak-every', '500', '--tokens', '96', '--graph',
                       '--prompt', 'The bird ', '--fast', '--log-every', '2000']
            p = subprocess.run(command, capture_output=True)
            (out/f'{dest.name}.log').write_bytes(p.stdout+p.stderr)
            if p.returncode:
                raise RuntimeError(f'Training {dest.name}: {p.stderr}')
            p = subprocess.run([str(exe), 'language-probes', '--checkpoint', str(dest/'latest.ckpt'),
                                '--probes', str(out/'tiny.sgprobe'), '--output', str(dest/'fitted.json')], capture_output=True)
            if p.returncode:
                raise RuntimeError(f'Probes {dest.name}: {p.stderr}')
            report = json.loads((dest/'fitted.json').read_text())
            results.append(dict(cell=cell, rate=rate, probes=report,
                                session=json.loads((dest/'session.json').read_text())))
            print(f'{cell} lr={rate:g}: joint={report["joint_accuracy"]} greedy={report["greedy_exact_joint_accuracy"]}', flush=True)
    report = dict(source_spec_sha256=hashlib.sha256(spec_path.read_bytes()).hexdigest(),
                  group=rows[0]['pair'], steps=6000, answer_emphasis=64, from_scratch=True,
                  full_size=True, single_training_group=True, no_heldout_evaluation=True,
                  limits='Diagnostic fit of four training examples. No generalization or dialogue claim.',
                  runs=results)
    (out/'diagnostic.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--exe', type=Path, default=Path('build/synapticgenesis.exe'))
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    run(a.exe, a.out)
