"""Report every frozen-self/control pair, including tradeoffs and failed checks."""
import argparse
from pathlib import Path
import statistics

from native_experiment import read, sha, write


def summarize(root, output, allow_smoke=False):
    data = read(root/'comparison.json')
    p = data['protocol']
    assert p['status'] == ('smoke_only' if allow_smoke else 'declared_before_training')
    execution = read(root/'execution-check.json')
    assert execution['passed'] and len(data['runs']) == len(p['seeds'])*18
    oracle = read(root/'learned-oracle.json') if (root/'learned-oracle.json').exists() else None
    assert allow_smoke or (oracle and len(oracle['records']) == 18)
    base, final = p['baseline_online_updates'], p['online_endpoints'][-1]
    rows, all_samples = [], []
    for record in data['runs']:
        seed, arm, end = record['seed'], record['arm'], record['online_updates']
        initial = next(r for r in data['runs'] if r['seed'] == seed and r['arm'] == arm and r['online_updates'] == base)
        directory = root/f'{seed}-{arm}'
        assert sha(directory/f'checkpoint-{end}.ckpt') == record['checkpoint_sha256']
        if end == final and oracle:
            checked = next(r for r in oracle['records'] if r['seed'] == seed and r['arm'] == arm)
            assert checked['checkpoint_sha256'] == record['checkpoint_sha256']
        row = {k: record[k] for k in ('seed', 'arm', 'architecture', 'variant', 'online_updates', 'parameters', 'checkpoint_sha256')}
        row.update(development_joint=record['development']['joint_accuracy'],
                   development_greedy_joint=record['development']['greedy_exact_joint_accuracy'],
                   development_item_accuracy=record['development']['accuracy'],
                   context_erased_accuracy=record['development']['context_erased_accuracy'],
                   binding_change=record['development']['joint_accuracy']-initial['development']['joint_accuracy'],
                   cumulative_narrative_seconds=record['cumulative_narrative_seconds'],
                   observed_pairs_added=record['observed_pairs']-initial['observed_pairs'],
                   replay_pairs_added=record['replay_pairs']-initial['replay_pairs'],
                   replay_updates_added=record['replay_updates']-initial['replay_updates'],
                   generated_bytes_added=record['generated_bytes']-initial['generated_bytes'])
        for book in p['book_validation']:
            row[book+'_loss'] = record['books'][book]['loss_nats_per_byte']
            row[book+'_change'] = row[book+'_loss']-initial['books'][book]['loss_nats_per_byte']
        if end != base:
            session = record['session']
            row.update(live_segment_seconds=session['elapsed_seconds'],
                       normal_tick_p95_ms=session['update_tick_p95_ms'],
                       speech_tick_p95_ms=session['update_and_speech_tick_p95_ms'],
                       teacher_extra_gpu_bytes=session.get('teacher_extra_gpu_bytes', 0),
                       teacher_updates=session.get('teacher_updates', 0), teacher_pairs=session.get('teacher_pairs', 0))
        if 'decode' in record:
            row.update({key: record['decode'][key] for key in ('graph_us_per_byte', 'regular_us_per_byte')})
        rows.append(row)
        for i, sample in enumerate(record['samples']):
            assert sha(directory/f'sample-{end}-{i}.txt') == sample['file_sha256']
        all_samples.append(dict(seed=seed, arm=arm, online_updates=end, samples=record['samples']))
    endpoints = [base]+p['online_endpoints']
    fields = ['development_joint', 'development_greedy_joint', 'development_item_accuracy', 'context_erased_accuracy',
              'binding_change', 'cumulative_narrative_seconds']
    fields += [book+suffix for book in p['book_validation'] for suffix in ('_loss', '_change')]
    means = []
    for end in endpoints:
        for arm in p['arms']:
            group = [r for r in rows if r['arm'] == arm and r['online_updates'] == end]
            measurements = fields + (['graph_us_per_byte', 'regular_us_per_byte', 'teacher_extra_gpu_bytes'] if end == final else [])
            means.append(dict(arm=arm, architecture=group[0]['architecture'], variant=group[0]['variant'], online_updates=end,
                              **{k: statistics.mean(r[k] for r in group) for k in measurements}))
    pairs = []
    for seed in p['seeds']:
        for architecture in p['architectures']:
            for end in endpoints:
                a, b = [next(r for r in rows if r['seed'] == seed and r['architecture'] == architecture and
                              r['variant'] == v and r['online_updates'] == end) for v in p['variants']]
                pair = dict(seed=seed, architecture=architecture, online_updates=end,
                            teacher_minus_control={k: b[k]-a[k] for k in fields})
                if end != base:
                    pair['observed_live_cost_ratio'] = b['cumulative_narrative_seconds']/a['cumulative_narrative_seconds']
                    pair['teacher_extra_gpu_bytes'] = b['teacher_extra_gpu_bytes']
                if end == final:
                    pair['gate'] = dict(binding_improves=b['development_joint'] > a['development_joint'],
                                        books_within_margin=all(b[book+'_loss'] <= a[book+'_loss']+.02 for book in p['book_validation']),
                                        narrative_improves_from_parent=b['narrative_change'] < 0)
                    pair['passes_gate'] = all(pair['gate'].values())
                pairs.append(pair)
    final_pairs = [r for r in pairs if r['online_updates'] == final]
    gate = dict(evaluated=not allow_smoke, passed=all(r['passes_gate'] for r in final_pairs) if not allow_smoke else None,
                pairs=len(final_pairs), improving_binding_pairs=sum(r['gate']['binding_improves'] for r in final_pairs),
                pairs_within_book_margin=sum(r['gate']['books_within_margin'] for r in final_pairs),
                threshold_nats_per_byte=.02)
    environment = read(root/'environment-note.json') if (root/'environment-note.json').exists() else None
    result = dict(protocol=p, rows=rows, means=means, paired=pairs, declared_gate=gate,
                  samples=all_samples, execution_check=execution, cpu_oracle=oracle,
                  timing_environment=environment, whole_device_timing_isolation_verified=False,
                  full_development_cpu_audit_completed=False,
                  retained_objective_failure=read('reports/live-teachers-validation.json')['retained_strict_failures'],
                  interpretation='Report all architectures, seeds, endpoints and both variants. Repeated development probes '
                                 'and sampled book loss are narrow measures. Runtime and source exposure are paired; teacher '
                                 'compute is extra and reported. Whole-device isolation is not verified: observed timing '
                                 'ratios do not establish intrinsic overhead or a speedup. No claim of useful conversation, '
                                 'lifelong retention or biological age.')
    write(output, result)
    print(gate)
    for architecture in p['architectures']:
        group = [r for r in final_pairs if r['architecture'] == architecture]
        print(architecture, {key: statistics.mean(r['teacher_minus_control'][key] for r in group)
                             for key in ('development_joint', 'narrative_loss', 'reader_loss', 'geography_loss')},
              'mean_observed_live_cost_ratio', statistics.mean(r['observed_live_cost_ratio'] for r in group))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--allow-smoke', action='store_true')
    a = p.parse_args()
    summarize(a.root, a.output, a.allow_smoke)
