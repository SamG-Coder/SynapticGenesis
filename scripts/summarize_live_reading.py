"""Report all live replay conditions, including cost, forgetting and failed gates."""
import argparse
from pathlib import Path
import statistics

from native_experiment import read, sha, write


def summarize(root):
    data, audit = read(root / 'comparison.json'), read(root / 'execution-check.json')
    p = data['protocol']
    assert audit['passed'] and audit['comparison_sha256'] == sha(root / 'comparison.json')
    baselines = {r['seed']: r for r in data['baselines']}
    rows = []
    for r in data['runs']:
        b = baselines[r['seed']]
        row = dict(seed=r['seed'], arm=r['arm'], additional_observations=r['additional_observations'],
            new_train_loss=r['reading']['train']['loss'], new_development_loss=r['reading']['validation']['loss'],
            new_development_change=r['reading']['validation']['loss'] - b['reading']['validation']['loss'],
            binding_joint=r['development']['joint_accuracy'],
            binding_change_pp=100 * (r['development']['joint_accuracy'] - b['development']['joint_accuracy']),
            cumulative_live_seconds=r['cumulative_live_seconds'],
            source_pairs=r['observed_pairs'] - b['observed_pairs'],
            replay_pairs=r['replay_pairs'] - b['replay_pairs'],
            optimizer_updates=r['global_updates'] - b['global_updates'],
            generated_bytes=r['generated_bytes'] - b['generated_bytes'])
        for book, value in r['books'].items():
            row[book + '_loss'] = value['loss_nats_per_byte']
            row[book + '_change'] = value['loss_nats_per_byte'] - b['books'][book]['loss_nats_per_byte']
        if 'decode' in r:
            row['graph_bytes_per_second'] = 1e6 / r['decode']['graph_us_per_byte']
        rows.append(row)
    means = []
    for arm in p['arms']:
        for end in p['additional_endpoints'][arm]:
            group = [r for r in rows if r['arm'] == arm and r['additional_observations'] == end]
            assert sorted(r['seed'] for r in group) == p['seeds']
            means.append(dict(arm=arm, additional_observations=end,
                **{k: statistics.mean(r[k] for r in group) for k in group[0] if k not in ('seed', 'arm', 'additional_observations')}))
    pairs = []
    final = p['common_exposure_endpoints'][-1]
    comparisons = [('equal_exposure', end, end) for end in p['common_exposure_endpoints']]
    comparisons.append(('equal_optimizer_updates', final, p['matched_optimizer_count_pair']['every-1']))
    for seed in p['seeds']:
        for kind, a_end, b_end in comparisons:
            a = next(r for r in rows if (r['seed'], r['arm'], r['additional_observations']) == (seed, 'every-4', a_end))
            b = next(r for r in rows if (r['seed'], r['arm'], r['additional_observations']) == (seed, 'every-1', b_end))
            pair = dict(seed=seed, comparison=kind, every_4_observations=a_end, every_1_observations=b_end,
                binding_improvement_pp=100 * (b['binding_joint'] - a['binding_joint']),
                new_development_loss_difference=b['new_development_loss'] - a['new_development_loss'],
                earlier_book_loss_differences={book: b[book + '_loss'] - a[book + '_loss'] for book in p['book_validation']},
                observed_live_cost_ratio=b['cumulative_live_seconds'] / a['cumulative_live_seconds'])
            if kind == 'equal_optimizer_updates':
                assert a['optimizer_updates'] == b['optimizer_updates'] == p['matched_additional_optimizer_updates']
            if kind == 'equal_exposure' and a_end == final:
                pair['gate'] = dict(binding_improves_at_least_five_pp=pair['binding_improvement_pp'] >= 5,
                    new_development_within_margin=pair['new_development_loss_difference'] <= .03,
                    old_books_within_margin=all(v <= .02 for v in pair['earlier_book_loss_differences'].values()),
                    new_development_improves_from_ancestor=b['new_development_change'] < 0)
                pair['passes_gate'] = all(pair['gate'].values())
            pairs.append(pair)
    finals = [r for r in pairs if 'gate' in r]
    gate = dict(evaluated=p['status'] != 'smoke_only', passed=None if p['status'] == 'smoke_only'
                else all(r['passes_gate'] for r in finals), seed_conditions=finals,
                automatic_promotion=False)
    result = dict(comparison_sha256=sha(root / 'comparison.json'), execution_check_sha256=sha(root / 'execution-check.json'),
        protocol=p, rows=rows, means=means, comparisons=pairs, declared_gate=gate,
        baselines=[dict(seed=b['seed'], reading=b['reading'], development=b['development'], books=b['books']) for b in data['baselines']],
        samples=[dict(seed=r['seed'], arm=r['arm'], samples=r['samples']) for r in data['runs'] if 'samples' in r],
        limits='One small edition and three related ancestors; development probes are reused. Equal optimizer count is not equal compute, target-byte exposure or speech. '
               'Timing is shared-desktop wall time. No proof of general conversation, lifelong retention or biological age.')
    write(root / 'summary.json', result)
    print('Gate:', gate['evaluated'], gate['passed'], flush=True)
    for r in means:
        if r['additional_observations'] == final:
            print(r, flush=True)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    summarize(p.parse_args().root)
