"""Audit every logged source update without claiming replay-gradient coverage."""
import json
import math
from pathlib import Path

from membrane_study_state import require
from prose_founder import file_hash


def percentile(values, fraction):
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lo, hi = int(position), min(int(position) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def audit_metrics(path, expected, clip=1., membrane_cost=0.):
    path = Path(path)
    endpoints = sorted(map(int, expected))
    require(len(endpoints) == 2 and 0 < endpoints[0] < endpoints[1] and
            [expected[str(end)]['curriculum_stage'] for end in endpoints] == [1, 2],
            'Expected two consecutive learning stages')
    groups = {end: [] for end in endpoints}
    previous_pairs, update, transitions, starts = 0, 0, [], 0
    for line in path.read_text(encoding='utf-8').splitlines():
        row = json.loads(line)
        if 'event' in row:
            if row['event'] == 'session_start':
                require(starts == update == 0 and row['online_update'] == 0 and
                        row['curriculum_stage'] == 1 and row['resume'] is False and row['si_strength'] == 0 and
                        row['replay_every'] == 4 and math.isclose(row['learning_rate'], .0003, rel_tol=1e-6),
                        'Unexpected initial learning session')
                starts += 1
                continue
            require(row['event'] == 'curriculum_transition' and
                    row['online_update'] == update == endpoints[0] and row['stage'] == 2,
                    'Unknown or misplaced metrics event')
            transitions.append((row['online_update'], row['stage']))
            continue
        require(starts == 1, 'Source metrics precede the initial session')
        update += 1
        require(row['online_update'] == update and update <= endpoints[-1], 'Missing or duplicated source metric')
        end = next(end for end in endpoints if update <= end)
        stage = expected[str(end)]['curriculum_stage']
        pairs = row['observed_pairs'] - previous_pairs
        require(1 <= pairs <= 128 and row['curriculum_stage'] == stage and
                row['global_update'] == update + update // 4 and row['replay_updates'] == update // 4,
                'Metric exposure counters differ')
        require(all(math.isfinite(row[k]) and row[k] >= 0 for k in ('loss', 'spike_rate', 'gradient_norm')) and
                row['spike_rate'] <= 1 and row['consolidation_events'] == 0, 'Invalid source metric')
        require(math.isclose(row.get('membrane_cost', 0), membrane_cost, rel_tol=2e-6, abs_tol=1e-12) and
                row.get('membrane_band', 1.5) == 1.5, 'Metric membrane policy differs')
        previous_pairs = row['observed_pairs']
        groups[end].append((row, pairs))
        if update == end:
            require(previous_pairs == expected[str(end)]['observed_pairs'], 'Stage byte exposure differs')
    require(starts == 1 and update == endpoints[-1] and transitions == [(endpoints[0], 2)],
            'Incomplete metric stream or transitions')
    rows = []
    for end, observed in groups.items():
        weights = sum(pairs for _, pairs in observed)
        norms = [row['gradient_norm'] for row, _ in observed]
        rows.append(dict(stage=expected[str(end)]['curriculum_stage'], end_update=end,
            source_updates=len(observed), source_target_bytes=weights,
            mean_language_loss=sum(row['loss'] * pairs for row, pairs in observed) / weights,
            mean_spike_rate=sum(row['spike_rate'] * pairs for row, pairs in observed) / weights,
            gradient_norm_p50=percentile(norms, .5), gradient_norm_p95=percentile(norms, .95),
            gradient_norm_max=max(norms), reported_above_clip=sum(x > clip for x in norms),
            reported_equal_to_clip=sum(x == clip for x in norms), clip=clip))
    return dict(passed=True, metrics_sha256=file_hash(path), source_updates=update, stages=rows,
        every_source_update_present=True, replay_gradient_norms_measured=False,
        limit='Norms are the combined source objective before clipping, printed to ten significant digits. '
              'Rows equal to the threshold can be ambiguous after rounding. Replay-update norms are not logged.')
