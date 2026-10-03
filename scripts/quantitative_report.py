"""Audit native quantitative scores and preserve the byte-level generation contract."""
from collections import defaultdict
import math

from membrane_study_state import require


def suite_hash(raw):
    value = 14695981039346656037
    for byte in raw:
        value = ((value ^ byte) * 1099511628211) & ((1 << 64) - 1)
    return str(value)


def rounding_radius(value):
    """Conservative error from native setprecision(12), plus float parsing."""
    require(type(value) in (float, int) and math.isfinite(value), 'Nonfinite or nonnumeric native value')
    if value == 0:
        return 0.
    return .5 * 10. ** (math.floor(math.log10(abs(value))) - 11) + 2 * math.ulp(float(value))


def rounded_equal(actual, expected, additional_error=0.):
    require(abs(actual - expected) <= rounding_radius(actual) + additional_error + 4 * math.ulp(float(expected)),
            'Native aggregate differs from its item records')


def prediction(scores, reported):
    require(isinstance(scores, list) and len(scores) == 2 and
            all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in scores),
            'Invalid native candidate likelihoods')
    require(type(reported) is int and reported in (-1, 0, 1), 'Invalid native prediction')
    delta = scores[0] - scores[1]
    radius = sum(rounding_radius(v) for v in scores)
    low, high, threshold = delta - radius, delta + radius, 1e-8
    possible = set()
    if low < -threshold:
        possible.add(0)
    if high > threshold:
        possible.add(1)
    if low <= threshold and high >= -threshold:
        possible.add(-1)
    require(reported in possible, 'Native prediction contradicts its likelihoods')
    return len(possible) > 1


def audit(report, rows, raw_suite, payload_hash):
    require(report['format'] == 'SGPROBE2' and report['suite_hash'] == suite_hash(raw_suite) and
            report['checkpoint_payload_hash'] == str(payload_hash), 'Native assessment identity differs')
    require(report['strict_fp32'] is True and report['parameters_and_optimizer_unchanged'] is True,
            'Native read-only or arithmetic contract differs')
    require((report['items'], report['groups'], report['group_size']) == (128, 32, 4) and len(rows) == 128,
            'Quantitative assessment coverage differs')
    require([r['id'] for r in report['results']] == [r['id'] for r in rows], 'Native item order differs')
    require(type(report['elapsed_seconds']) in (int, float) and math.isfinite(report['elapsed_seconds']) and
            report['elapsed_seconds'] > 0, 'Invalid native assessment timing')
    groups, skills, erased_contexts = defaultdict(list), defaultdict(list), {}
    nll, error, targets, ambiguous = 0., 0., 0, 0
    outputs = []
    for actual, expected in zip(report['results'], rows):
        require(all(actual[k] == expected[k] for k in ('id', 'pair', 'skill')) and
                type(actual['gold']) is int and actual['gold'] == expected['correct'],
                'Native item label or identity differs')
        ambiguous += prediction(actual['candidate_nll'], actual['prediction'])
        ambiguous += prediction(actual['context_erased_nll'], actual['context_erased_prediction'])
        choices = [expected['choice0'].encode('ascii'), expected['choice1'].encode('ascii')]
        require(len(choices[0]) == len(choices[1]), 'Unequal quantitative answer lengths')
        # Native JSON serializes each unsigned output byte as a U+00xx code point.
        value = actual['greedy']
        require(isinstance(value, str) and all(ord(c) <= 255 for c in value), 'Native generation is not a byte string')
        raw = value.encode('latin-1')
        require(len(raw) == len(choices[0]), 'Greedy byte budget differs')
        correct = actual['prediction'] == actual['gold']
        erased = actual['context_erased_prediction'] == actual['gold']
        exact = raw == choices[actual['gold']]
        observed = (correct, erased, exact)
        groups[actual['pair']].append(observed)
        skills[actual['skill']].append((actual['pair'], observed))
        key = (expected['query'], expected['choice0'], expected['choice1'])
        erased_value = (actual['context_erased_nll'], actual['context_erased_prediction'])
        require(erased_contexts.setdefault(key, erased_value) == erased_value,
                'Identical context-erased inputs produced different results')
        loss = actual['candidate_nll'][actual['gold']]
        nll += loss
        error += rounding_radius(loss)
        targets += len(choices[actual['gold']])
        outputs.append(dict(id=actual['id'], prompt=expected['context'] + expected['query'],
            expected_answer=choices[actual['gold']].decode('ascii'),
            prediction=actual['prediction'], greedy_hex=raw.hex(),
            greedy_utf8=raw.decode('utf-8', errors='replace')))
    require(len(groups) == 32 and all(len(v) == 4 for v in groups.values()), 'Native group coverage differs')
    all_items = [v for group in groups.values() for v in group]
    counts = [sum(v[i] for v in all_items) for i in range(3)]
    joint = [sum(all(v[i] for v in group) for group in groups.values()) for i in range(3)]
    require(joint[1] == 0, 'Context-erased group correctness is inconsistent')
    for i, key in enumerate(('accuracy', 'context_erased_accuracy', 'greedy_exact_accuracy')):
        rounded_equal(report[key], counts[i] / 128)
    for i, key in enumerate(('joint_accuracy', 'context_erased_joint_accuracy', 'greedy_exact_joint_accuracy')):
        rounded_equal(report[key], joint[i] / 32)
    require(report['answer_bytes'] == targets, 'Native answer-byte denominator differs')
    rounded_equal(report['answer_loss_nats_per_byte'], nll / targets, error / targets)
    require(set(report['skills']) == set(skills), 'Native skill inventory differs')
    per_skill = {}
    for skill, values in skills.items():
        native = report['skills'][skill]
        require(native['items'] == len(values) == 16, 'Native skill coverage differs')
        by_group = defaultdict(list)
        for group, value in values:
            by_group[group].append(value)
        result = dict(items=len(values), groups=len(by_group))
        for i, key in enumerate(('accuracy', 'context_erased_accuracy', 'greedy_exact_accuracy')):
            result[key] = sum(v[i] for _, v in values) / len(values)
            rounded_equal(native[key], result[key])
        result['joint_accuracy'] = sum(all(v[0] for v in g) for g in by_group.values()) / len(by_group)
        result['greedy_exact_joint_accuracy'] = sum(all(v[2] for v in g) for g in by_group.values()) / len(by_group)
        per_skill[skill] = result
    return dict(passed=True, items=128, groups=32, correct_items=counts[0], correct_groups=joint[0],
        greedy_exact_items=counts[2], greedy_exact_groups=joint[2],
        context_erased_correct_items=counts[1], context_erased_correct_groups=joint[1],
        decisions_ambiguous_at_printed_precision=ambiguous, answer_bytes=targets,
        elapsed_seconds=report['elapsed_seconds'], per_skill=per_skill, outputs=outputs,
        boundary='Arithmetic and integrity of the saved native report; not an independent neural forward oracle.')
