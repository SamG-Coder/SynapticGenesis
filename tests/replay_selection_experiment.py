"""Audit learned policy causality, compute controls and final numerical checks."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from native_experiment import read, sha
from binding_learned_oracle import development_rows
from probes_cli import Reference
from replay_priority_probe import journal
from replay_selection_probe import verify_choices


def write(path, value):
    Path(path).write_bytes((json.dumps(value, indent=2) + '\n').encode('utf-8'))


def audit(root, out):
    study = read(root / 'comparison.json')
    p = study['protocol']
    assert study['complete'] and study['inputs_unchanged']
    assert all(sha(path) == identity for path, identity in p['authenticated_inputs'].items())
    assert all(sha(path) == identity for path, identity in p['source_sha256'].items())
    fixture = development_rows(p)[:4]
    rows = []
    for row in study['runs']:
        directory, session = root / row['directory'], row['session']
        model = directory / 'latest.ckpt'
        assert sha(model) == row['state']['checkpoint_sha256']
        base = next(b for b in p['bases'] if b['seed'] == row['seed'])
        decisions = journal(directory / 'selection.jsonl')
        policy = verify_choices(Path(base['checkpoint']), Path(p['schedule']), decisions, model,
                                prioritize=row['mode'] == 'interference')
        if row['mode'] == 'none':
            assert not decisions and session['score_forward_calls'] == 0
        else:
            assert len(decisions) == session['comparisons'] == (session['end'] - 160000) // 4
            assert sum(d['chosen'] for d in decisions) == session['overrides']
            assert session['score_forward_calls'] == 2 * sum(len(d['candidates']) for d in decisions)
            assert session['scored_pairs'] == 2 * sum(c['length'] for d in decisions for c in d['candidates'])
        native = read(directory / f'development-{session["end"]}.json')['results'][:4]
        ref, error, greedy_equal = Reference(model), 0, True
        for test, actual in zip(fixture, native):
            assert test['id'] == actual['id']
            for context, field in [(test['context'], 'candidate_nll'), ('', 'context_erased_nll')]:
                scores = [ref.score(context + test['query'], test[f'choice{i}']) for i in range(2)]
                error = max(error, *(abs(a - b) for a, b in zip(scores, actual[field])))
            greedy_equal &= ref.greedy(test['context'] + test['query'], 4) == actual['greedy']
        rows.append(dict(seed=row['seed'], budget=row['budget'], mode=row['mode'], independent_policy=policy,
            cpu_score_error=error, tolerance=3e-5, cpu_score_passed=error < 3e-5,
            cpu_greedy_equal=greedy_equal, checkpoint_sha256=sha(model)))
        print(row['directory'], 'policy exact; CPU error', error, 'greedy', greedy_equal, flush=True)
    for base in p['bases']:
        seed = base['seed']
        assert sha(root / f'{seed}-updates-none/latest.ckpt') == sha(root / f'{seed}-updates-uniform/latest.ckpt')
        # Every priority comparison used the same candidate descriptors and
        # score workload as the cost-matched uniform arm at equal exposure.
        a = journal(root / f'{seed}-updates-uniform/selection.jsonl')
        b = journal(root / f'{seed}-updates-interference/selection.jsonl')
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert x['observation'] == y['observation'] and x['group'] == y['group']
            assert [{k: c[k] for k in ('document', 'offset', 'length')} for c in x['candidates']] == [
                {k: c[k] for k in ('document', 'offset', 'length')} for c in y['candidates']]
    write(out, dict(execution_passed=True, comparison_sha256=sha(root / 'comparison.json'),
                    equal_exposure_uniform_control_exact=True, candidate_workloads_matched=True,
                    all_cpu_scores_passed=all(r['cpu_score_passed'] for r in rows),
                    all_cpu_greedy_equal=all(r['cpu_greedy_equal'] for r in rows), rows=rows,
                    scope='Independent host policy for every update; CPU math covers one fixed development group per final model.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    audit(args.root, args.out)
