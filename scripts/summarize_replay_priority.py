"""Describe measured interference and its cost, without claiming replay benefit."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics

import numpy as np

from native_experiment import read, sha, write


def ranks(values):
    result = np.empty(len(values), dtype=float)
    for value in set(values):
        indexes = [i for i, v in enumerate(values) if v == value]
        rank = sum(v < value for v in values) + (len(indexes) - 1) / 2
        result[indexes] = rank
    return result


def agreement(a, b):
    ra, rb = ranks(a), ranks(b)
    return float(np.corrcoef(ra, rb)[0, 1]) if ra.std() and rb.std() else None


def summarize(root, execution, out):
    comparison, audit = read(root / 'comparison.json'), read(execution)
    assert audit['execution_passed'] and audit['comparison_sha256'] == sha(root / 'comparison.json')
    results = []
    for base in comparison['protocol']['bases']:
        seed = base['seed']
        runs = [r for r in comparison['runs'] if r['seed'] == seed]
        before = [r['result'] for r in runs if r['mode'] == 'disabled']
        after = [r['result'] for r in runs if r['mode'] == 'measured']
        time_before = statistics.median(r['live_seconds'] for r in before)
        time_after = statistics.median(r['live_seconds'] for r in after)
        events = [json.loads(line) for line in (root / f'{seed}-0-measured/scores.jsonl').read_text().splitlines()]
        criteria = {}
        for criterion in ('mean', 'weighted', 'answer'):
            all_deltas, correlations, top_equal, top4_overlap, groups = [], [], [], [], defaultdict(list)
            for event in events:
                candidates = [c for c in event['candidates'] if c['before'][criterion] is not None]
                loss = [c['before'][criterion] for c in candidates]
                delta = [c['after'][criterion] - c['before'][criterion] for c in candidates]
                all_deltas.extend(delta)
                correlation = agreement(loss, delta)
                if correlation is not None:
                    correlations.append(correlation)
                by_loss = sorted(range(len(candidates)), key=lambda i: (-loss[i], i))
                by_delta = sorted(range(len(candidates)), key=lambda i: (-delta[i], i))
                top_equal.append(by_loss[0] == by_delta[0])
                k = min(4, len(candidates))
                top4_overlap.append(len(set(by_loss[:k]) & set(by_delta[:k])) / k)
                for c, d in zip(candidates, delta):
                    groups[c['group']].append(d)
            criteria[criterion] = dict(candidates=len(all_deltas), positive_fraction=sum(d > 0 for d in all_deltas) / len(all_deltas),
                positive_above_0_00003_fraction=sum(d > 3e-5 for d in all_deltas) / len(all_deltas),
                mean_delta=statistics.mean(all_deltas), min_delta=min(all_deltas), max_delta=max(all_deltas),
                median_rank_correlation=statistics.median(correlations), top1_agreement=statistics.mean(top_equal),
                mean_top4_overlap=statistics.mean(top4_overlap),
                groups={g: dict(count=len(ds), mean_delta=statistics.mean(ds),
                                positive_fraction=sum(d > 0 for d in ds) / len(ds)) for g, ds in groups.items()})
        results.append(dict(seed=seed, disabled_seconds=[r['live_seconds'] for r in before],
            measured_seconds=[r['live_seconds'] for r in after], median_disabled_seconds=time_before,
            median_measured_seconds=time_after, extra_live_time_percent=100 * (time_after / time_before - 1),
            source_byte_targets_per_second_disabled=before[0]['source_pairs'] / time_before,
            source_byte_targets_per_second_measured=after[0]['source_pairs'] / time_after,
            scoring_seconds=[r['scoring_seconds'] for r in after],
            view_setup_seconds=[r['scorer_setup_seconds'] for r in after],
            scorer_owned_array_bytes=after[0]['scorer_owned_array_bytes'], scorer_views=after[0]['scorer_views'],
            observed_free_memory_drop_bytes=[r['observed_free_memory_drop_bytes'] for r in after],
            score_forward_calls=after[0]['score_forward_calls'], scored_pairs=after[0]['scored_pairs'],
            scoring_tick_p95_ms=[r['scoring_tick_p95_ms'] for r in after],
            max_tick_ms=[r['max_tick_ms'] for r in after], criteria=criteria))
    result = dict(comparison_sha256=sha(root / 'comparison.json'), execution_sha256=sha(execution),
        execution_passed=True, all_cpu_scores_passed=audit['all_cpu_scores_passed'], rows=results,
        interpretations=['The loss-change signal measures immediate native interference only.',
            'Before-loss ranking is a best-case fresh difficulty proxy, not a tested stale-cache policy.',
            'The 3e-5 delta bin is descriptive; the single-score oracle tolerance is not a confidence bound on deltas.',
            'No priority selector was enabled; all learning replay remained the original uniform stage policy.',
            'A later selector study must compare retention and learning within matched time budgets.'])
    write(out, result)
    for row in results:
        print(row['seed'], 'time +', round(row['extra_live_time_percent'], 1), '%; weighted rank correlation',
              round(row['criteria']['weighted']['median_rank_correlation'], 3), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--execution', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    summarize(args.root, args.execution, args.out)
