"""Describe response shortcuts using the selected, independently checked labels.

Sensitivity measures describe changed predictions, not necessarily correct ones.
"""
import argparse
import json
from pathlib import Path
import re

from prepare_binding import examples


def audit(path, split='development'):
    spec = json.loads(Path('data/lessons-binding-v2.json').read_text())
    _, _, partitions, _ = examples(spec)
    expected = {r['id']:r for r in partitions[split]}
    results = json.loads(path.read_text())['results']
    assert len(results) == len(expected) and {r['id'] for r in results} == set(expected)
    groups = {}
    first = last = 0
    for r in results:
        row = expected[r['id']]
        facts = re.findall(r'The (\w+) is in the (\w+)\.', row['context'])
        if not facts:
            facts = [(obj, loc) for loc,obj in re.findall(r'In the (\w+) is the (\w+)\.', row['context'])]
        answer = row[f"choice{r['prediction']}"] if r['prediction'] in (0,1) else None
        first += answer == facts[0][1]+'.'
        last += answer == facts[-1][1]+'.'
        groups.setdefault(row['pair'], []).append((row,r['prediction']))
    query_flips = fact_flips = comparisons = 0
    for rows in groups.values():
        by_context, by_query = {}, {}
        for row, prediction in rows:
            by_context.setdefault(row['context'],[]).append(prediction)
            by_query.setdefault(row['query'],[]).append(prediction)
        assert len(by_context) == len(by_query) == 2
        def changed(pair):
            assert len(pair) == 2
            return all(p in (0,1) for p in pair) and pair[0] != pair[1]
        query_flips += sum(changed(p) for p in by_context.values())
        fact_flips += sum(changed(p) for p in by_query.values())
        comparisons += 2
    return dict(split=split, items=len(results), groups=len(groups),
                predicted_first_location_fraction=first/len(results),
                predicted_last_location_fraction=last/len(results),
                changed_prediction_when_query_changes=query_flips/comparisons,
                changed_prediction_when_facts_swap=fact_flips/comparisons,
                sensitivity_is_not_accuracy=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('report', type=Path)
    p.add_argument('--split', choices=['train','development'], default='development')
    a = p.parse_args()
    print(json.dumps(audit(a.report,a.split), indent=2))
