"""Collect preparation, state-continuity and all declared ordering comparisons."""
import argparse
from collections import Counter
from pathlib import Path
import statistics

from native_experiment import read, sha, write


def validation():
    data = read('runs/curriculum-order-data-check/result.json')
    refactor = read('runs/curriculum-order-refactor-smoke/refactor-check.json')
    smoke = read('runs/curriculum-order-smoke/comparison.json')
    oracle = read('runs/curriculum-order-smoke/learned-oracle.json')
    prefix = read('runs/curriculum-order-prefix-check/result.json')
    assert all(record['passed'] for record in (data,refactor,oracle,prefix))
    assert smoke['native_commands'] == 34 and len(smoke['runs']) == 8
    assert smoke['protocol']['status'] == 'smoke_only'
    executable = sha('build/synapticgenesis.exe')
    assert smoke['protocol']['executable_sha256'] == prefix['executable_sha256'] == executable
    assert read('runs/binding-diversity-panel/learned-oracle.json') == read('reports/lesson-diversity-learned-oracle.json')
    for name,record in data['prepared_manifest']['files'].items():
        assert sha(Path('data/prepared/curriculum-order-v1')/name) == record['sha256']
    result = dict(passed=True,executable_sha256=executable,data=data,shared_driver_refactor=refactor,
                  shared_cpu_oracle_matches_previous_six_model_report=True,learned_prefix=prefix,
                  native_five_stage_smoke=dict(passed=True,native_commands=34,measured_checkpoints=8,
                    complete_shared_ancestor=True,source_byte_counts_verified_at_every_endpoint=True,
                    replay_quotas_checked_at_each_boundary=True,final_observation_counts_match=True,
                    cpu_oracle=oracle,comparison_sha256=sha('runs/curriculum-order-smoke/comparison.json'),
                    learning_quality_claim=False))
    write('reports/curriculum-order-validation.json',result)
    return result


def main(root,validation_only):
    checked = validation()
    if validation_only:
        print('Collected selected-order validation and exact learned-prefix/state checks.')
        return
    data = read(root/'comparison.json')
    p = data['protocol']
    assert p['status'] == 'declared_before_training' and p['seeds'] == [1337,2026,31415]
    assert p['online_endpoints'] == [34000,66640,78160,130000]
    assert len(data['runs']) == 24 and data['native_commands'] == 99
    assert p['executable_sha256'] == checked['executable_sha256'] and not p['reserved_test_evaluated']
    commands = read(root/'commands.json')
    assert len(commands) == 99
    probes = Counter(Path(c[c.index('--probes')+1]).name for c in commands if c[1] == 'language-probes')
    assert probes == {'train.sgprobe':24,'development.sgprobe':24,'expanded-train-monitor.sgprobe':24}
    assert all('15659' not in ' '.join(c) for c in commands)
    oracle = read(root/'learned-oracle.json')
    assert oracle['passed'] and len(oracle['records']) == 6
    ancestors = {row['seed']:row for row in data['shared_ancestors']}
    rows = []
    for seed in p['seeds']:
        ancestor = ancestors[seed]
        assert ancestor['previous_prerequisite_arrays_identical']
        assert sha(root/f'{seed}-common/latest.ckpt') == ancestor['checkpoint_sha256']
        for arm in p['arms']:
            points = sorted((r for r in data['runs'] if r['seed']==seed and r['arm']==arm),key=lambda r:r['online_updates'])
            assert [r['online_updates'] for r in points] == p['online_endpoints']
            final = points[-1]
            s = final['session']
            for row in points:
                assert row['ancestor_checkpoint_sha256'] == ancestor['checkpoint_sha256']
                assert sha(root/f'{seed}-{arm}'/f'checkpoint-{row["online_updates"]}.ckpt') == row['checkpoint_sha256']
                for split in ('train','development','expanded-train-monitor'):
                    assert row[split]['parameters_and_optimizer_unchanged'] and row[split]['context_erased_joint_accuracy'] == 0
            numeric = next(r for r in oracle['records'] if r['seed']==seed and r['arm']==arm)
            assert numeric['checkpoint_sha256'] == final['checkpoint_sha256'] and numeric['oracle_max_score_error'] < 3e-5
            rows.append(dict(seed=seed,arm=arm,parameters=final['parameters'],
                reading_ancestor_loss=ancestor['reading_session']['final_validation_loss'],
                common_prefix_loss=ancestor['session']['final_validation_loss'],final_reader_loss=s['final_validation_loss'],
                reader_change_from_reading=s['final_validation_loss']-ancestor['reading_session']['final_validation_loss'],
                reader_change_from_common=s['final_validation_loss']-ancestor['session']['final_validation_loss'],
                train_joint=final['train']['joint_accuracy'],development_joint=final['development']['joint_accuracy'],
                development_greedy_joint=final['development']['greedy_exact_joint_accuracy'],
                expanded_monitor_joint=final['expanded-train-monitor']['joint_accuracy'],
                expanded_monitor_greedy_joint=final['expanded-train-monitor']['greedy_exact_joint_accuracy'],
                live_segment_seconds=sum(r['session']['elapsed_seconds'] for r in points),
                observed_pairs=s['observed_pairs'],replay_pairs=s['replay_pairs'],global_updates=s['global_updates'],
                replay_updates=s['replay_updates'],generated_bytes=s['generated_bytes'],replay_state_bytes=s['replay_state_bytes'],
                replay_groups=s['replay_groups'],final_tick_p95_ms=s['update_tick_p95_ms'],
                final_speech_tick_p95_ms=s['update_and_speech_tick_p95_ms'],checkpoint_sha256=final['checkpoint_sha256']))
    keys = ('final_reader_loss','reader_change_from_reading','reader_change_from_common','train_joint',
            'development_joint','development_greedy_joint','expanded_monitor_joint','expanded_monitor_greedy_joint',
            'live_segment_seconds','observed_pairs','replay_pairs')
    means = [dict(arm=arm,**{key:statistics.mean(r[key] for r in rows if r['arm']==arm) for key in keys}) for arm in p['arms']]
    paired = []
    for seed in p['seeds']:
        ramped = next(r for r in rows if r['seed']==seed and r['arm']=='ramped')
        shuffled = next(r for r in rows if r['seed']==seed and r['arm']=='shuffled')
        for key in ('parameters','observed_pairs','global_updates','replay_updates','generated_bytes','replay_state_bytes'):
            assert ramped[key] == shuffled[key],(seed,key)
        assert ramped['observed_pairs'] == ancestors[seed]['session']['observed_pairs']+8879952
        paired.append(dict(seed=seed,**{key:ramped[key]-shuffled[key] for key in keys}))
    summary = dict(protocol=p,final_rows=rows,means=means,paired_difference_ramped_minus_shuffled=paired,
                   exact_final_document_multiset_matches=True,executed_probe_commands=dict(probes),
                   cost_boundary=p['timing'],cpu_oracle_max_error=max(r['oracle_max_score_error'] for r in oracle['records']),
                   prior_three_stage_experiment_is_not_a_matched_control=True,
                   no_general_language_or_biological_equivalence_claim=True)
    write('reports/curriculum-order-language.json',data)
    write('reports/curriculum-order-summary.json',summary)
    write('reports/curriculum-order-learned-oracle.json',oracle)
    print(means)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path('runs/curriculum-order-panel'))
    p.add_argument('--validation-only',action='store_true')
    a = p.parse_args()
    main(a.root,a.validation_only)
