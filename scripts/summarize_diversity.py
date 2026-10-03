"""Collect selected-lesson validation and all declared paired learning results."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(name, value):
    Path('reports', name).write_text(json.dumps(value, indent=2)+'\n')


def validation():
    data = read('runs/binding-diversity-validation/result.json')
    original = read('runs/binding-refactor-validation/result.json')
    # This historical extraction check uses archived earlier-run checkpoints.
    # A fresh reproduction can collect its new data/smoke checks without them;
    # absence is recorded, never replaced with a fabricated passing result.
    helper_path = Path('runs/binding-diversity-helper-check.json')
    helper = read(helper_path) if helper_path.exists() else None
    smoke = read('runs/binding-diversity-smoke/comparison.json')
    assert data['passed'] and original['passed'] and (helper is None or helper['passed'])
    assert smoke['protocol']['status'] == 'smoke_only' and len(smoke['runs']) == 6
    assert smoke['native_commands'] == 25
    assert data['executable_sha256'] == smoke['protocol']['executable_sha256'] == sha('build/synapticgenesis.exe')
    assert not smoke['protocol']['reserved_test_evaluated']
    for row in smoke['runs']:
        assert row['ancestor_checkpoint_sha256'] == smoke['shared_ancestors'][0]['checkpoint_sha256']
        assert row['session']['gradient_reductions'] == 'ordered_v1'
        for split in ('train', 'development', 'expanded-train-monitor'):
            assert row[split]['parameters_and_optimizer_unchanged'] and row[split]['context_erased_joint_accuracy'] == 0
    for name, record in data['prepared_manifest']['files'].items():
        assert sha(Path('data/prepared/binding-diversity-v1')/name) == record['sha256']
    prior_native = read('reports/ordered-reductions-validation.json')
    same_native = prior_native['executable_sha256'] == data['executable_sha256']
    report = dict(data=data, original_binding=original, extracted_checkpoint_view=helper,
                  native_driver_smoke=dict(passed=True, native_commands=25, endpoints=[64, 96, 128],
                    arms=2, shared_ancestor_checkpoint_sha256=smoke['shared_ancestors'][0]['checkpoint_sha256'],
                    extension_and_resume=True, paired_update_speech_memory_counters_match=True,
                    checkpoint_unchanged_by_scoring=True, context_erased_joint_accuracy=0,
                    full_comparison_sha256=sha('runs/binding-diversity-smoke/comparison.json'),
                    learning_quality_claim=False),
                  executable_sha256=data['executable_sha256'],
                  native_implementation_unchanged_from='19c55fc' if same_native else None,
                  native_14_suite_and_cpu_gradient_evidence='ordered-reductions-validation.json' if same_native else None)
    write('lesson-diversity-validation.json', report)
    return report


def main(root, validation_only):
    checked = validation()
    if validation_only:
        print('Collected data, original-renderer and 25-command native smoke validation; historical checkpoint evidence is recorded when available.')
        return
    data = read(root/'comparison.json')
    p = data['protocol']
    assert p['status'] == 'declared_before_training' and p['seeds'] == [1337, 2026, 31415]
    assert p['online_endpoints'] == [34000, 67000, 130000] and len(data['runs']) == 18
    assert p['executable_sha256'] == checked['executable_sha256']
    assert not p['reserved_test_evaluated'] and data['native_commands'] == 75
    commands = read(root/'commands.json')
    assert len(commands) == 75
    probe_commands = [command for command in commands if command[1] == 'language-probes']
    scored = Counter(Path(command[command.index('--probes')+1]).name for command in probe_commands)
    assert scored == {'train.sgprobe': 18, 'development.sgprobe': 18, 'expanded-train-monitor.sgprobe': 18}
    assert all('15659' not in ' '.join(command) for command in commands)
    for records in p['source_records'].values():
        assert all(sha(record['source']) == record['sha256'] for record in records)
    oracle = read(root/'learned-oracle.json')
    assert oracle['passed'] and len(oracle['records']) == 6
    ancestors = {row['seed']: row for row in data['shared_ancestors']}
    rows = []
    for seed in p['seeds']:
        for arm in p['arms']:
            points = sorted((row for row in data['runs'] if row['seed'] == seed and row['arm'] == arm),
                            key=lambda row: row['online_updates'])
            assert [row['online_updates'] for row in points] == p['online_endpoints']
            final = points[-1]
            session = final['session']
            numeric = next(row for row in oracle['records'] if row['seed'] == seed and row['arm'] == arm)
            assert numeric['checkpoint_sha256'] == final['checkpoint_sha256'] and numeric['oracle_max_score_error'] < 3e-5
            assert all(row['ancestor_checkpoint_sha256'] == ancestors[seed]['checkpoint_sha256'] for row in points)
            for row in points:
                assert sha(root/f'{seed}-{arm}'/f'checkpoint-{row["online_updates"]}.ckpt') == row['checkpoint_sha256']
                for split in ('train', 'development', 'expanded-train-monitor'):
                    assert row[split]['parameters_and_optimizer_unchanged'] and row[split]['context_erased_joint_accuracy'] == 0
            rows.append(dict(seed=seed, arm=arm, parameters=final['parameters'],
                ancestor_reader_loss=ancestors[seed]['session']['final_validation_loss'],
                final_reader_loss=session['final_validation_loss'],
                reader_loss_change=session['final_validation_loss']-ancestors[seed]['session']['final_validation_loss'],
                train_joint=final['train']['joint_accuracy'],
                development_joint=final['development']['joint_accuracy'],
                development_greedy_joint=final['development']['greedy_exact_joint_accuracy'],
                expanded_monitor_joint=final['expanded-train-monitor']['joint_accuracy'],
                expanded_monitor_greedy_joint=final['expanded-train-monitor']['greedy_exact_joint_accuracy'],
                live_segment_seconds=sum(row['session']['elapsed_seconds'] for row in points),
                observed_pairs=session['observed_pairs'], replay_pairs=session['replay_pairs'],
                replay_updates=session['replay_updates'], global_updates=session['global_updates'],
                generated_bytes=session['generated_bytes'], replay_state_bytes=session['replay_state_bytes'],
                replay_groups=session['replay_groups'],
                final_tick_p95_ms=session['update_tick_p95_ms'],
                final_speech_tick_p95_ms=session['update_and_speech_tick_p95_ms'],
                checkpoint_sha256=final['checkpoint_sha256']))
    keys = ('final_reader_loss', 'reader_loss_change', 'train_joint', 'development_joint',
            'development_greedy_joint', 'expanded_monitor_joint', 'expanded_monitor_greedy_joint',
            'live_segment_seconds', 'observed_pairs', 'replay_pairs')
    means = [dict(arm=arm, **{key: statistics.mean(row[key] for row in rows if row['arm'] == arm) for key in keys})
             for arm in p['arms']]
    paired = []
    for seed in p['seeds']:
        control = next(row for row in rows if row['seed'] == seed and row['arm'] == 'control')
        diversity = next(row for row in rows if row['seed'] == seed and row['arm'] == 'diversity')
        for field in ('parameters', 'replay_updates', 'global_updates', 'generated_bytes', 'replay_state_bytes'):
            assert control[field] == diversity[field], (seed, field)
        paired.append(dict(seed=seed, **{key: diversity[key]-control[key] for key in keys}))
    exposure = []
    for arm in p['arms']:
        selected = Path('data/prepared/binding-v2' if arm == 'control' else 'data/prepared/binding-diversity-v1')
        stages = []
        for name, observations in [('prerequisites', 4000), ('binding', 120000)]:
            raw = (selected/f'{name}.dat').read_bytes()
            expected_hash = checked['data']['prepared_manifest']['files'][f'{name}.dat']['sha256'] if arm == 'diversity' else (
                read(selected/'manifest.json')['files'][f'{name}.dat']['sha256'])
            assert hashlib.sha256(raw).hexdigest() == expected_hash
            lengths = [len(doc)-1 for doc in raw.split(b'\x1e')]
            assert all(0 < length <= p['chunk'] for length in lengths)
            cycles, remainder = divmod(observations, len(lengths))
            stages.append(dict(stage=name, documents=len(lengths), online_observations=observations,
                               complete_online_passes=cycles, next_pass_documents=remainder,
                               expected_observed_pairs=cycles*sum(lengths)+sum(lengths[:remainder])))
        for row in rows:
            if row['arm'] == arm:
                assert row['observed_pairs']-ancestors[row['seed']]['session']['observed_pairs'] == sum(
                    stage['expected_observed_pairs'] for stage in stages)
        exposure.append(dict(arm=arm, stages=stages, replay_excluded_from_pass_counts=True))
    summary = dict(protocol=p, final_rows=rows, means=means, paired_difference_diversity_minus_control=paired,
                   online_lesson_exposure=exposure,
                   executed_probe_commands=dict(scored),
                   learned_model_cpu_oracle_max_error=max(row['oracle_max_score_error'] for row in oracle['records']),
                   cost_boundary=p['timing'],
                   monitor_is_sample_not_full_training_accuracy=True,
                   no_general_language_or_biological_equivalence_claim=True)
    write('lesson-diversity-language.json', data)
    write('lesson-diversity-learned-oracle.json', oracle)
    write('lesson-diversity-summary.json', summary)
    print(json.dumps(dict(final_rows=rows, means=means, paired_differences=paired), indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path('runs/binding-diversity-panel'))
    p.add_argument('--validation-only', action='store_true')
    a = p.parse_args()
    main(a.root, a.validation_only)
