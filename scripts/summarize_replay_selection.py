"""Apply the declared retention gate without selecting a best seed or endpoint."""
import argparse
import json
from pathlib import Path
import statistics

from native_experiment import read, sha


def summarize(root, out):
    data = read(root / 'comparison.json')
    assert data['complete']
    rows, gates, aggregates = data['runs'], [], []
    for base in data['protocol']['bases']:
        seed = base['seed']
        ancestor = next(r for r in data['ancestors'] if r['seed'] == seed)
        for budget in data['protocol']['budgets']:
            arms = {r['mode']: r for r in rows if r['seed'] == seed and r['budget'] == budget}
            priority = arms['interference']
            failures = []
            binding = priority['development']['joint_accuracy']
            differences = {}
            for control in ('none', 'uniform'):
                other = arms[control]
                differences[control] = binding - other['development']['joint_accuracy']
                if differences[control] <= 0:
                    failures.append('binding_not_improved_vs_' + control)
                for name in priority['books']:
                    delta = priority['books'][name]['loss_nats_per_byte'] - other['books'][name]['loss_nats_per_byte']
                    if delta > .02:
                        failures.append(name + '_loss_regressed_vs_' + control)
            if binding < ancestor['development']['joint_accuracy']:
                failures.append('binding_below_ancestor')
            gates.append(dict(seed=seed, budget=budget, passed=not failures, failures=failures,
                              binding_delta=differences, ancestor_binding=ancestor['development']['joint_accuracy'],
                              priority_binding=binding))
    for budget in data['protocol']['budgets']:
        for mode in data['protocol']['modes']:
            selected = [r for r in rows if r['budget'] == budget and r['mode'] == mode]
            aggregates.append(dict(budget=budget, mode=mode,
                mean_binding_accuracy=statistics.mean(r['development']['joint_accuracy'] for r in selected),
                mean_live_seconds=statistics.mean(r['session']['live_seconds'] for r in selected),
                mean_source_observations=statistics.mean(r['session']['end'] - r['session']['start'] for r in selected),
                mean_new_byte_targets_per_second=statistics.mean(r['session']['source_pairs'] / r['session']['live_seconds'] for r in selected),
                mean_books={name: statistics.mean(r['books'][name]['loss_nats_per_byte'] for r in selected)
                            for name in selected[0]['books']},
                overrides=sum(r['session']['overrides'] for r in selected),
                comparisons=sum(r['session']['comparisons'] for r in selected)))
    result = dict(comparison_sha256=sha(root / 'comparison.json'), gates=gates, aggregates=aggregates,
                  candidate_passed=all(g['passed'] for g in gates),
                  mean_ancestor_binding=statistics.mean(r['development']['joint_accuracy'] for r in data['ancestors']),
                  scope='Declared small pilot on repeated development material; no general-language or lifespan claim.')
    Path(out).write_bytes((json.dumps(result, indent=2) + '\n').encode())
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    summarize(args.root, args.out)
