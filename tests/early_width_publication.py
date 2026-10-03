"""Check published width aggregation and reject broken case/activity boundaries."""
import argparse
import copy
from fractions import Fraction
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from native_experiment import read
from prose_founder import file_hash, write
from publish_early_width import summaries


def check(report):
    source = Path('runs/early-width-publication-v1.json')
    publication = read(source)
    rows, spec = publication['rows'], publication['protocol']['specification']
    actual = summaries(rows, spec)
    assert actual == publication['summary']
    # Independently transcribed six-book assessment means, in seed order.
    # Each value is the mean of the four held-out validation books only.
    expected = {
        'c512-control': ['2.1810795', '2.1860955', '2.1806865'],
        'c1024-control': ['2.1929085', '2.22856225', '2.2268665'],
        'c256-control': ['2.1779215', '2.2011105', '2.197869'],
        'c512-core-half': ['2.190656', '2.18973575', '2.1742345'],
        'c1024-core-quarter': ['2.20364675', '2.20942425', '2.20288925'],
    }
    for group in actual['cases']:
        reference = [Fraction(v) for v in expected[group['case']['name']]]
        assert abs(group['validation_mean_across_seeds'] - float(sum(reference) / 3)) < 1e-14
        assert all(abs(a - float(b)) < 1e-14 for a, b in zip(group['validation_means'], reference))
        assert group['source_clipping_fractions'] == [1., 1., 1.]
    assert [(p['improved_seeds'], p['worsened_seeds']) for p in actual['rate_pairs']] == [(1, 2), (2, 1)]
    # The final activity observation is a ten-byte tail, not a 128-byte window.
    assert all(r['observations'][-1]['target_bytes'] == 10 for r in rows)
    assert all(r['observations'][4]['target_bytes'] == 128 for r in rows)
    rejected = []
    for label in ('missing_case', 'duplicate_case', 'reordered_seeds', 'altered_shape',
                  'missing_final_update', 'missing_final_layer'):
        changed = copy.deepcopy(rows)
        if label == 'missing_case': changed.pop()
        elif label == 'duplicate_case': changed[-1] = copy.deepcopy(changed[-2])
        elif label == 'reordered_seeds': changed[:5], changed[5:10] = changed[5:10], changed[:5]
        elif label == 'altered_shape': changed[-1]['case']['layers'] += 1
        elif label == 'missing_final_update': changed[-1]['observations'].pop()
        elif label == 'missing_final_layer': changed[-1]['observations'][-1]['pre_update_forward_layers'].pop()
        try:
            summaries(changed, spec)
        except (ValueError, AssertionError):
            rejected.append(label)
        else:
            raise AssertionError('Broken aggregate accepted: ' + label)
    assert read(source) == publication
    write(report, dict(passed=True, actual_groups_checked=5, paired_directions_checked=2,
        final_activity_target_bytes=10, earlier_activity_target_bytes=128,
        rejected_cases=rejected, input_sha256=file_hash(source), raw_outputs_unchanged=True,
        implementation_sha256={str(p): file_hash(p) for p in (Path(__file__), Path('scripts/publish_early_width.py'))},
        new_native_commands=0, independent_neural_forward_executed=False))
    print('Passed five actual groups and two paired directions; rejected six broken aggregates.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    assert not args.report.exists(), 'Use a fresh report'
    check(args.report)
