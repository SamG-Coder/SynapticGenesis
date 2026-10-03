"""Report every learned-memory intervention and its numerical limitations."""
import argparse
from pathlib import Path
import statistics

from native_experiment import read, sha, write


def summarize(root, learning):
    data = read(root / 'comparison.json')
    source = read(learning / 'comparison.json')
    protocol = data['protocol']
    assert data['completed'] and data['controls']['passed']
    assert protocol['learning_comparison_sha256'] == sha(learning / 'comparison.json')
    assert protocol['seeds'] == [1337, 2026, 31415]
    assert protocol['modes'] == ['normal', 'discard_history', 'zero_read']
    assert protocol['questions'] == 576 and protocol['groups'] == 144
    assert not protocol['reserved_test_evaluated'] and not protocol['model_parameters_changed']
    assert len(data['rows']) == 9 and len(data['numerical']) == 3
    assert len({(r['seed'], r['mode']) for r in data['rows']}) == 9
    rows, paired = [], []
    for seed in protocol['seeds']:
        native = next(r for r in source['runs'] if r['seed'] == seed and r['arm'] == 'associative'
                      and r['online_updates'] == protocol['online_endpoint'])
        checkpoint = learning / f'{seed}-associative/checkpoint-130000.ckpt'
        assert sha(checkpoint) == native['checkpoint_sha256']
        normal = next(r for r in data['rows'] if r['seed'] == seed and r['mode'] == 'normal')
        for mode in protocol['modes']:
            row = next(r for r in data['rows'] if r['seed'] == seed and r['mode'] == mode)
            assert row['checkpoint_sha256'] == native['checkpoint_sha256']
            raw = read(root / f'{seed}-{mode}.json')
            assert raw['items'] == 576 and raw['groups'] == 144 and len(raw['results']) == 576
            assert all(raw[key] == row[key] for key in raw if key != 'results')
            rows.append(dict(**row,
                             native_joint_accuracy=native['development']['joint_accuracy'],
                             native_greedy_joint_accuracy=native['development']['greedy_exact_joint_accuracy']))
            if mode != 'normal':
                paired.append(dict(seed=seed, mode=mode,
                    joint_drop=normal['joint_accuracy'] - row['joint_accuracy'],
                    greedy_joint_drop=normal['greedy_joint_accuracy'] - row['greedy_joint_accuracy']))
    keys = ['joint_accuracy', 'greedy_joint_accuracy', 'accuracy', 'context_erased_accuracy']
    means = [dict(mode=mode, **{key: statistics.mean(r[key] for r in rows if r['mode'] == mode)
                               for key in keys}) for mode in protocol['modes']]
    numerical = data['numerical']
    result = dict(protocol=protocol, controls=data['controls'], rows=rows, means=means, paired=paired,
        numerical=numerical,
        all_normal_scores_within_tolerance=data['all_normal_scores_within_tolerance'],
        normal_maximum_score_error=max(r['maximum_score_error'] for r in numerical),
        normal_questions=sum(r['items'] for r in numerical),
        normal_score_failure_questions=sum(f['max_score_error'] >= protocol['numerical_tolerance']
                                          for r in numerical for f in r['failures']),
        normal_candidate_disagreements=sum(r['candidate_choice_disagreements'] for r in numerical),
        normal_greedy_disagreements=sum(r['greedy_disagreements'] for r in numerical),
        intervention_results_sha256={f'{r["seed"]}-{r["mode"]}.json': sha(root / f'{r["seed"]}-{r["mode"]}.json')
                                     for r in rows},
        interpretation='Read-only intervention in all three final associative models. Removing history or reads '
            'changes activation distributions, so this measures reliance in trained models, not the performance '
            'of a retrained architecture. Native/CPU score or answer disagreements are retained without '
            'relaxing the strict tolerance. No general-language or biological-memory claim.')
    write('reports/associative-history.json', result)
    for row in means:
        print(row['mode'], {key: round(row[key], 6) for key in keys})
    print('Normal CPU/native score tolerance passed:', result['all_normal_scores_within_tolerance'])
    print('Normal CPU/native disagreements:', result['normal_candidate_disagreements'],
          'candidate choices,', result['normal_greedy_disagreements'], 'greedy answers')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('runs/associative-history-panel'))
    parser.add_argument('--learning', type=Path, default=Path('runs/associative-long-panel'))
    args = parser.parse_args()
    summarize(args.root, args.learning)
