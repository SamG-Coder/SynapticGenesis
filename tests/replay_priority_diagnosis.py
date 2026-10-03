"""Post-hoc trace diagnosis; retain the original strict CPU score failure."""
import argparse
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from extend_curriculum import read_schedule
from native_experiment import NativeCommands, read, sha, write
from probes_cli import Reference
from replay_priority_probe import journal
from trace_comparison import compare


def diagnose(root, execution, diagnostic, out):
    audit = read(execution)
    protocol = read(root / 'protocol.json')
    _, stages = read_schedule(Path(protocol['schedule']))
    docs = stages[-1]['content'].split(b'\x1e')
    out.mkdir(parents=True, exist_ok=False)
    commands = NativeCommands(diagnostic, out)
    results = []
    for row in audit['rows']:
        failures = [c for c in row['cpu_oracle']['checks'] if not c['passed']]
        for index, failure in enumerate(failures):
            seed, phase = row['seed'], failure['phase']
            snapshot = root / f'{seed}-oracle/score-{phase}.ckpt'
            identity = sha(snapshot)
            candidates = journal(root / f'{seed}-oracle/scores.jsonl')[0]['candidates']
            candidate = next(c for c in candidates if all(c[k] == failure[k] for k in ('document', 'offset', 'length')))
            source, trace = out / f'{seed}-{index}.txt', out / f'{seed}-{index}-trace'
            doc, offset, length = failure['document'], failure['offset'], failure['length']
            raw = docs[doc][offset:offset + length + 1]
            source.write_bytes(raw)
            commands(snapshot, source, trace)
            losses = np.fromfile(trace / 'losses.f32', dtype='<f4').astype('float64')
            answer = candidate[phase]['answer_targets']
            assert float(losses.mean()) == candidate[phase]['mean']
            if answer:
                assert float(losses[-answer:].mean()) == candidate[phase]['answer']
            with torch.inference_mode():
                result = compare(Reference(snapshot), raw, trace, answer_bytes=answer or length)
            results.append(dict(seed=seed, failure=failure, candidate=candidate,
                                checkpoint_sha256=identity, native_scores_reproduced_exactly=True,
                                answer_targets=answer, **result))
            assert sha(snapshot) == identity
    assert results and not audit['all_cpu_scores_passed']
    result = dict(selection='All failed first-update CPU candidate score checks; post-hoc trace diagnosis.',
                  execution_sha256=sha(execution), diagnostic_sha256=sha(diagnostic),
                  source_sha256={p: sha(p) for p in ['tests/replay_priority_diagnosis.py',
                      'tests/trace_comparison.py', 'tests/probes_cli.py', 'tests/learned_trace_dump.cu']},
                  records=results, original_strict_cpu_check_passed=False,
                  limits='Forced CPU spike is a diagnostic intervention, not an independent pass or a production fix.')
    write(out / 'result.json', result)
    print(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--execution', type=Path, required=True)
    parser.add_argument('--diagnostic', type=Path, default=Path('build/trace-diagnostic/synaptic-trace-diagnostic.exe'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    diagnose(args.root, args.execution, args.diagnostic, args.out)
