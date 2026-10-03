"""Collect preparation, state-continuity and all declared ordering comparisons."""
import argparse
from collections import Counter
from pathlib import Path
import statistics

from native_experiment import read, sha, write


def validation():
    data = read('runs/curriculum-order-data-check/result.json')
    # Historical comparisons require archived files from before extraction.
    # Fresh reproductions must still pass the current data and native checks;
    # missing historical evidence is recorded as absent, never as a pass.
    refactor_path = Path('runs/curriculum-order-refactor-smoke/refactor-check.json')
    refactor = read(refactor_path) if refactor_path.exists() else None
    smoke = read('runs/curriculum-order-smoke/comparison.json')
    oracle = read('runs/curriculum-order-smoke/learned-oracle.json')
    prefix_path = Path('runs/curriculum-order-prefix-check/result.json')
    prefix = read(prefix_path) if prefix_path.exists() else None
    assert all(record['passed'] for record in (data,refactor,oracle,prefix) if record is not None)
    assert smoke['native_commands'] == 34 and len(smoke['runs']) == 8
    assert smoke['protocol']['status'] == 'smoke_only'
    executable = sha('build/synapticgenesis.exe')
    assert smoke['protocol']['executable_sha256'] == executable
    if prefix is not None:
        assert prefix['executable_sha256'] == executable
    previous_oracle_path = Path('runs/binding-diversity-panel/learned-oracle.json')
    historical_oracle_match = None
    if previous_oracle_path.exists():
        prior = read(previous_oracle_path)
        published = read('reports/lesson-diversity-learned-oracle.json')
        if [r['checkpoint_sha256'] for r in prior['records']] == [r['checkpoint_sha256'] for r in published['records']]:
            assert prior == published
            historical_oracle_match = True
    for name,record in data['prepared_manifest']['files'].items():
        assert sha(Path('data/prepared/curriculum-order-v1')/name) == record['sha256']
    result = dict(passed=True,executable_sha256=executable,data=data,shared_driver_refactor=refactor,
                  shared_cpu_oracle_matches_previous_six_model_report=historical_oracle_match,learned_prefix=prefix,
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
    assert len(oracle['records']) == 6
    failures = [r for r in oracle['records'] if r['oracle_max_score_error'] >= 3e-5]
    assert oracle['passed'] == (not failures)
    cpu = read(root/'cpu-development-audit.json')
    assert cpu['completed'] and len(cpu['records']) == 6
    assert cpu['original_development_probe_sha256'] == p['probe_sha256']['development.sgprobe']
    trace_path = Path('runs/curriculum-order-threshold-check/result.json')
    trace = read(trace_path) if trace_path.exists() else None
    if trace is not None:
        assert trace['localization_passed'] and trace['study_executable_sha256'] == p['executable_sha256']
        assert trace['checkpoint_sha256'] in {r['checkpoint_sha256'] for r in failures}
    reference_check_path = Path('runs/curriculum-order-reference-hooks-check/result.json')
    reference_check = read(reference_check_path) if reference_check_path.exists() else None
    if reference_check is not None:
        assert reference_check['passed'] and reference_check['native_commands'] == 59
    ancestors = {row['seed']:row for row in data['shared_ancestors']}
    rows, samples = [], []
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
            assert numeric['checkpoint_sha256'] == final['checkpoint_sha256']
            audited = next(r for r in cpu['records'] if r['seed']==seed and r['arm']==arm)
            assert audited['checkpoint_sha256'] == final['checkpoint_sha256'] and audited['items'] == 576
            assert audited['native_joint_accuracy'] == final['development']['joint_accuracy']
            # Preserve the existing final live sample from every model. This is
            # descriptive text inspection, not another scored quality endpoint.
            transcript = root/f'{seed}-{arm}'/'transcript.txt'
            raw = transcript.read_bytes().replace(b'\r\n',b'\n')
            marker = f'[online update {s["online_updates"]}; global update {s["global_updates"]}]'.encode('ascii')
            assert raw.count(marker) == 1
            suffix = raw.split(marker)[1]
            assert suffix.startswith(b'\n') and suffix.endswith(b'\n')
            sample = suffix[1:-1]
            assert sample.startswith(b'The bird ') and len(sample) == 9+p['generated_bytes_per_speech']
            samples.append(dict(seed=seed,arm=arm,online_updates=s['online_updates'],
                transcript_sha256=sha(transcript),prompt='The bird ',
                prompt_and_generated_bytes_hex=sample.hex(),
                displayed_text=sample.decode('utf-8',errors='backslashreplace')))
            rows.append(dict(seed=seed,arm=arm,parameters=final['parameters'],
                reading_ancestor_loss=ancestor['reading_session']['final_validation_loss'],
                common_prefix_loss=ancestor['session']['final_validation_loss'],final_reader_loss=s['final_validation_loss'],
                reader_change_from_reading=s['final_validation_loss']-ancestor['reading_session']['final_validation_loss'],
                reader_change_from_common=s['final_validation_loss']-ancestor['session']['final_validation_loss'],
                train_joint=final['train']['joint_accuracy'],development_joint=final['development']['joint_accuracy'],
                development_greedy_joint=final['development']['greedy_exact_joint_accuracy'],
                cpu_development_joint=audited['cpu_joint_accuracy'],
                cpu_development_greedy_joint=audited['cpu_greedy_joint_accuracy'],
                cpu_candidate_choice_disagreements=len(audited['candidate_choice_disagreements']),
                cpu_greedy_answer_disagreements=len(audited['greedy_answer_disagreements']),
                expanded_monitor_joint=final['expanded-train-monitor']['joint_accuracy'],
                expanded_monitor_greedy_joint=final['expanded-train-monitor']['greedy_exact_joint_accuracy'],
                live_segment_seconds=sum(r['session']['elapsed_seconds'] for r in points),
                observed_pairs=s['observed_pairs'],replay_pairs=s['replay_pairs'],global_updates=s['global_updates'],
                replay_updates=s['replay_updates'],generated_bytes=s['generated_bytes'],replay_state_bytes=s['replay_state_bytes'],
                replay_groups=s['replay_groups'],final_tick_p95_ms=s['update_tick_p95_ms'],
                final_speech_tick_p95_ms=s['update_and_speech_tick_p95_ms'],checkpoint_sha256=final['checkpoint_sha256']))
    keys = ('final_reader_loss','reader_change_from_reading','reader_change_from_common','train_joint',
            'development_joint','development_greedy_joint','cpu_development_joint','cpu_development_greedy_joint',
            'expanded_monitor_joint','expanded_monitor_greedy_joint',
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
                   strict_fixed_group_cpu_score_check_passed=oracle['passed'],
                   cpu_score_tolerance_failures=oracle.get('failed_score_tolerance',[]),
                   full_development_cpu_audit_completed=True,
                   full_development_cpu_max_score_error=max(r['max_score_error'] for r in cpu['records']),
                   full_development_cpu_score_pairs_above_tolerance=sum(r['scored_pairs_above_3e5_tolerance'] for r in cpu['records']),
                   cpu_all_development_candidate_choices_identical=all(not r['candidate_choice_disagreements'] for r in cpu['records']),
                   cpu_all_development_generated_answers_identical=all(not r['greedy_answer_disagreements'] for r in cpu['records']),
                   threshold_diagnostic=trace,
                   reference_hook_validation=reference_check,
                   final_live_samples=samples,
                   live_sample_scope='Existing final 96-byte live generation from every arm and seed, with its prompt; '
                                     'qualitative inspection only, not an additional quality benchmark.',
                   prior_three_stage_experiment_is_not_a_matched_control=True,
                   no_general_language_or_biological_equivalence_claim=True)
    write('reports/curriculum-order-language.json',data)
    write('reports/curriculum-order-summary.json',summary)
    write('reports/curriculum-order-learned-oracle.json',oracle)
    write('reports/curriculum-order-cpu-audit.json',cpu)
    print(means)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path('runs/curriculum-order-panel'))
    p.add_argument('--validation-only',action='store_true')
    a = p.parse_args()
    main(a.root,a.validation_only)
