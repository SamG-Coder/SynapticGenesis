"""Report every breadth/replay condition and its predeclared retention gate."""
import argparse
from pathlib import Path
import statistics

from native_experiment import read, sha, write
from reading_breadth_sources import grouped_reading


def summarize(root):
    data, audit = read(root / 'comparison.json'), read(root / 'execution-check.json')
    assert audit['passed'] and audit['comparison_sha256'] == sha(root / 'comparison.json')
    p = data['protocol']
    baselines = {b['seed']: b for b in data['baselines']}
    rows = []
    for r in data['runs']:
        b = baselines[r['seed']]
        values, original = grouped_reading(r['reading'], p['assessment']), grouped_reading(b['reading'], p['assessment'])
        row = dict(seed=r['seed'], arm=r['arm'], additional_observations=r['additional_observations'], **values,
                   **{key+'_change': value-original[key] for key, value in values.items()},
                   binding_joint=r['development']['joint_accuracy'], binding_change_pp=100*(r['development']['joint_accuracy']-b['development']['joint_accuracy']),
                   cumulative_live_seconds=r['cumulative_live_seconds'], source_pairs=r['observed_pairs']-b['observed_pairs'],
                   replay_pairs=r['replay_pairs']-b['replay_pairs'], optimizer_updates=r['global_updates']-b['global_updates'],
                   generated_bytes=r['generated_bytes']-b['generated_bytes'])
        for book, value in r['books'].items():
            row[book+'_loss'] = value['loss_nats_per_byte']
            row[book+'_change'] = value['loss_nats_per_byte']-b['books'][book]['loss_nats_per_byte']
        if 'decode' in r:
            row['graph_bytes_per_second'] = 1e6/r['decode']['graph_us_per_byte']
        rows.append(row)
    means = []
    for arm in p['arms']:
        for end in p['additional_endpoints']:
            group = [r for r in rows if r['arm'] == arm and r['additional_observations'] == end]
            assert sorted(r['seed'] for r in group) == p['seeds']
            means.append(dict(arm=arm, additional_observations=end, **{
                key:statistics.mean(r[key] for r in group) for key in group[0] if key not in ('seed', 'arm', 'additional_observations')}))
    pairs = []
    for seed in p['seeds']:
        for cadence in (4, 1):
            for end in p['additional_endpoints']:
                a, b = [next(r for r in rows if (r['seed'], r['arm'], r['additional_observations']) == (seed, f'{source}-{cadence}', end))
                        for source in ('starter', 'broader')]
                pair = dict(seed=seed, replay_every=cadence, additional_observations=end,
                            binding_improvement_pp=100*(b['binding_joint']-a['binding_joint']),
                            development_differences={name:b[name]-a[name] for name in ('starter_validation', 'expansion_validation')},
                            earlier_book_differences={name:b[name+'_loss']-a[name+'_loss'] for name in p['book_validation']},
                            live_cost_ratio=b['cumulative_live_seconds']/a['cumulative_live_seconds'])
                assert a['optimizer_updates'] == b['optimizer_updates'] and a['generated_bytes'] == b['generated_bytes']
                if end == p['primary_endpoint']:
                    gate = p['gate']
                    pair['gate'] = dict(
                        both_development_sets_improve_from_ancestor=all(b[name+'_change'] < 0 for name in pair['development_differences']),
                        both_development_sets_no_worse_than_starter=all(d <= 0 for d in pair['development_differences'].values()),
                        binding_ancestor_margin=b['binding_change_pp'] >= -gate['binding_ancestor_drop_limit_pp'],
                        binding_starter_margin=pair['binding_improvement_pp'] >= -gate['binding_starter_drop_limit_pp'],
                        earlier_book_ancestor_margin=all(b[name+'_change'] <= gate['earlier_book_ancestor_loss_limit'] for name in p['book_validation']),
                        earlier_book_starter_margin=all(d <= gate['earlier_book_starter_loss_limit'] for d in pair['earlier_book_differences'].values()))
                    pair['passes_gate'] = all(pair['gate'].values())
                pairs.append(pair)
    gate = dict(evaluated=p['status'] != 'smoke_only', automatic_promotion=False,
                by_replay_cadence={str(c): None if p['status'] == 'smoke_only' else
                    all(r['passes_gate'] for r in pairs if r['replay_every'] == c and 'passes_gate' in r) for c in (4, 1)})
    result = dict(comparison_sha256=sha(root / 'comparison.json'), execution_check_sha256=sha(root / 'execution-check.json'),
                  protocol=p, rows=rows, means=means, paired_comparisons=pairs, declared_gate=gate,
                  baselines=[dict(seed=b['seed'], reading=grouped_reading(b['reading'], p['assessment']),
                                  binding_joint=b['development']['joint_accuracy'], books=b['books']) for b in data['baselines']],
                  samples=[dict(seed=r['seed'], arm=r['arm'], samples=r['samples']) for r in data['runs'] if 'samples' in r],
                  limits=p['limits'])
    write(root / 'summary.json', result)
    print('Gate:', gate, flush=True)
    for r in means:
        if r['additional_observations'] in (p['primary_endpoint'], p['additional_endpoints'][-1]):
            print(r, flush=True)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    summarize(p.parse_args().root)
