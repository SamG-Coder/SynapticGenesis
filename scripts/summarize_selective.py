"""Collect selective-trace language, memory, numerical and lifecycle evidence."""
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def regression_example():
    """First development group correct at 67k and incomplete at 130k; illustrative only."""
    from prepare_binding import examples
    rows = examples(read('data/lessons-binding-v2.json'))[2]['development']
    earlier = {r['id']: r for r in read('runs/selective-binding-panel/selective/development-67000.json')['results']}
    later = {r['id']: r for r in read('runs/selective-binding-panel/selective/development-130000.json')['results']}
    groups = {}
    for row in rows:
        groups.setdefault(row['pair'], []).append(row)
    for group, members in groups.items():
        def correct(predictions):
            return all(predictions[r['id']]['greedy'] == r[f'choice{r["correct"]}'] for r in members)
        if correct(earlier) and not correct(later):
            return dict(selection='First complete development group in source order that regressed from '
                                  'all-correct greedy answers at 67000 to incomplete at 130000. '
                                  'Selected after evaluation for illustration, not an independent metric.',
                        group=group, examples=[dict(id=r['id'], prompt=r['context'] + r['query'],
                            expected=r[f'choice{r["correct"]}'], greedy_67000=earlier[r['id']]['greedy'],
                            greedy_130000=later[r['id']]['greedy']) for r in members])
    return None


def main():
    language = read('runs/selective-binding-panel/comparison.json')
    cues = read('runs/selective-cue-panel/comparison.json')
    binary_sha = hashlib.sha256(Path('build/synapticgenesis.exe').read_bytes()).hexdigest()
    assert language['executable_sha256'] == cues['executable_sha256'] == binary_sha
    assert language['initial_parameters_and_state_identical'] and language['same_source_and_replay_exposure']
    assert language['matched_parameter_count'] and not language['reserved_test_evaluated']
    assert hashlib.sha256(Path('data/lessons-binding-v2.json').read_bytes()).hexdigest() == language['source_spec_sha256']
    assert '100% tests passed out of 12' in Path('build-selective-final.log').read_text(encoding='utf-8-sig')
    sources = dict(native_cell='build/selective-test-results/native.json',
                   native_evolution='build/evolution-test-results/native.json',
                   native_curriculum='build/curriculum-test-results/native.json',
                   cpu_gradients='build/selective-test-results/oracle.json',
                   cpu_weighted_gradients='build/feedback-test-results/selective/oracle.json',
                   prior_trace_cpu='build/trace-test-results/oracle.json',
                   prior_read_gate_cpu='build/gated-test-results/oracle.json',
                   curriculum_cli='runs/selective-curriculum-cli-v2/result.json',
                   extension_cli='runs/selective-extension-cli/result.json',
                   scoring_cli='runs/selective-probes-cli/result.json',
                   binding_cli='runs/selective-binding-cli/result.json',
                   population_live='runs/selective-population-cli/result.json',
                   population_scarcity='runs/selective-scarcity-cli-v2/result.json',
                   burn_policy='runs/selective-burn-cli/result.json')
    validation = dict(native_suites_passed=12, executable_sha256=binary_sha)
    for name, path in sources.items():
        validation[name] = read(path)
        assert validation[name]['passed'], name
    assert validation['curriculum_cli']['feedback_resume_cases'] == 10
    assert validation['extension_cli']['extension_cases'] == 40
    # Weights and passive base decay are observations, not measured memory spans.
    from inspect_dynamics import inspect
    dynamics = {cell: inspect(Path('runs/selective-binding-panel') / cell / 'latest.ckpt')
                for cell in ('gated', 'selective')}
    Path('reports/selective-binding.json').write_text(json.dumps(language, indent=2) + '\n')
    Path('reports/selective-cue.json').write_text(json.dumps(cues, indent=2) + '\n')
    Path('reports/selective-validation.json').write_text(json.dumps(validation, indent=2) + '\n')
    Path('reports/selective-dynamics.json').write_text(json.dumps(dynamics, indent=2) + '\n')
    Path('reports/selective-regression-example.json').write_text(json.dumps(regression_example(), indent=2) + '\n')
    table = []
    for row in language['runs']:
        table.append(dict(cell=row['cell'], online_updates=row['online_updates'],
                          train_joint=row['train']['joint_accuracy'],
                          development_joint=row['development']['joint_accuracy'],
                          development_greedy_joint=row['development']['greedy_exact_joint_accuracy'],
                          development_accuracy=row['development']['accuracy'],
                          reader_loss=row['session']['final_validation_loss'],
                          live_segment_seconds=row['session']['elapsed_seconds']))
    print(json.dumps(table, indent=2))


if __name__ == '__main__':
    main()
