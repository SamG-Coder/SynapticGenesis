"""Publish all continuation models, language changes, retained skills and costs."""
import argparse
from pathlib import Path
import statistics

from native_experiment import read, sha, write


def summarize(root):
    data = read(root / 'comparison.json')
    p = data['protocol']
    assert p['status'] == 'declared_before_training'
    assert p['seeds'] == [1337, 2026, 31415] and p['online_endpoints'] == [160000, 190000]
    execution, oracle = read(root / 'execution-check.json'), read(root / 'learned-oracle.json')
    assert execution['passed'] and len(oracle['records']) == 9
    assert data['native_commands'] == 180 and len(data['runs']) == 27
    assert sha(Path(p['parent_directory']) / 'comparison.json') == p['parent_comparison_sha256']
    assert not p['reserved_test_evaluated']
    rows, samples = [], []
    for record in data['runs']:
        seed, arm, end = record['seed'], record['arm'], record['online_updates']
        initial = next(r for r in data['runs'] if r['seed'] == seed and r['arm'] == arm
                       and r['online_updates'] == p['baseline_online_updates'])
        directory = root / f'{seed}-{arm}'
        assert sha(directory / f'checkpoint-{end}.ckpt') == record['checkpoint_sha256']
        row = dict(seed=seed, arm=arm, online_updates=end, parameters=record['parameters'],
                   checkpoint_sha256=record['checkpoint_sha256'],
                   development_joint=record['development']['joint_accuracy'],
                   development_greedy_joint=record['development']['greedy_exact_joint_accuracy'],
                   development_item_accuracy=record['development']['accuracy'],
                   context_erased_accuracy=record['development']['context_erased_accuracy'],
                   binding_group_change=record['development']['joint_accuracy'] - initial['development']['joint_accuracy'],
                   cumulative_narrative_seconds=record['cumulative_narrative_seconds'],
                   observed_pairs_added=record['observed_pairs'] - initial['observed_pairs'],
                   replay_pairs_added=record['replay_pairs'] - initial['replay_pairs'],
                   replay_updates_added=record['replay_updates'] - initial['replay_updates'],
                   generated_bytes_added=record['generated_bytes'] - initial['generated_bytes'])
        for book in p['book_validation']:
            row[book + '_loss'] = record['books'][book]['loss_nats_per_byte']
            row[book + '_change'] = row[book + '_loss'] - initial['books'][book]['loss_nats_per_byte']
        if 'session' in record:
            row['live_segment_seconds'] = record['session']['elapsed_seconds']
            row['normal_tick_p95_ms'] = record['session']['update_tick_p95_ms']
            row['speech_tick_p95_ms'] = record['session']['update_and_speech_tick_p95_ms']
        if 'decode' in record:
            row['graph_us_per_byte'] = record['decode']['graph_us_per_byte']
            row['regular_us_per_byte'] = record['decode']['regular_us_per_byte']
        rows.append(row)
        for index, sample in enumerate(record['samples']):
            assert sha(directory / f'sample-{end}-{index}.txt') == sample['file_sha256']
        samples.append(dict(seed=seed, arm=arm, online_updates=end, samples=record['samples']))
    endpoints = [p['baseline_online_updates']] + p['online_endpoints']
    fields = ['development_joint', 'development_greedy_joint', 'development_item_accuracy',
              'context_erased_accuracy', 'binding_group_change', 'cumulative_narrative_seconds']
    fields += [book + suffix for book in p['book_validation'] for suffix in ('_loss', '_change')]
    means = []
    for endpoint in endpoints:
        for arm in p['arms']:
            selected = [r for r in rows if r['arm'] == arm and r['online_updates'] == endpoint]
            measures = fields + (['graph_us_per_byte', 'regular_us_per_byte'] if endpoint == endpoints[-1] else [])
            means.append(dict(arm=arm, online_updates=endpoint,
                              **{key: statistics.mean(r[key] for r in selected) for key in measures}))
    paired = []
    for seed in p['seeds']:
        for endpoint in endpoints:
            memory = next(r for r in rows if r['seed'] == seed and r['arm'] == 'associative'
                          and r['online_updates'] == endpoint)
            for arm in ('selective', 'wide-selective'):
                baseline = next(r for r in rows if r['seed'] == seed and r['arm'] == arm
                                and r['online_updates'] == endpoint)
                paired.append(dict(seed=seed, control=arm, online_updates=endpoint,
                                   associative_minus_control={key: memory[key] - baseline[key] for key in fields}))
    result = dict(protocol=p, rows=rows, means=means, paired=paired, samples=samples,
                  execution_check=execution, cpu_oracle=oracle, full_development_cpu_audit_completed=False,
                  interpretation='Every parent, endpoint, architecture and seed is reported. Loss deltas compare '
                    'each model with its own authenticated parent. Architectures start with different learned '
                    'capabilities. Book loss uses sampled reset-state byte windows; samples do not establish '
                    'conversation or long-story comprehension. Strict numerical failures remain visible.')
    write('reports/narrative-language.json', data)
    write('reports/narrative-summary.json', result)
    write('reports/narrative-execution.json', execution)
    write('reports/narrative-learned-oracle.json', oracle)
    for row in means:
        print(row['online_updates'], row['arm'],
              {key: round(row[key], 6) for key in ('narrative_change', 'reader_change', 'geography_change',
                                                 'development_joint', 'cumulative_narrative_seconds')})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('runs/narrative-panel'))
    summarize(parser.parse_args().root)
