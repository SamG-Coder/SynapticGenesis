"""Solve the displayed development questions independently and check shortcut controls."""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
import re
import shlex
import sys
import tempfile

import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from native_experiment import read, sha
from prepare_quantitative_probes import prepare
from prose_founder import write
from quantitative_probes import build, protect


PATTERNS = {
    'addition': r'A collection starts with (\d+) items and receives (\d+) more items\.',
    'subtraction': r'A collection starts with (\d+) items and then (\d+) items are removed\.',
    'multiplication': r'There are (\d+) rows with (\d+) items in every row\.',
    'division': r'There are (\d+) items shared equally among (\d+) boxes\.',
    'speed': r'The total distance travelled is (\d+) m and the elapsed time is (\d+) s\.',
    'net_force': r'An object has mass (\d+) kg and acceleration magnitude (\d+) m/s\^2\.',
    'work': r'The force magnitude is (\d+) N and the displacement magnitude is (\d+) m\.',
    'heat': r'A sample has mass (\d+) kg, specific heat (\d+) J/\(kg K\), and temperature rise (\d+) K\.',
}
QUESTIONS = {
    'addition': 'how many items are in the collection now?',
    'subtraction': 'how many items remain in the collection?',
    'multiplication': 'how many items are there in total?',
    'division': 'how many items are in each box?',
    'speed': 'what is the average speed in m/s?',
    'net_force': 'what is the magnitude of the net external force in N?',
    'work': 'what is the work done by the stated force in J?',
    'heat': 'how much heat is supplied in J?',
}


def decode(raw):
    fields = shlex.split(raw.decode('ascii'), comments=False, posix=True)
    assert fields[0] == 'SGPROBE2' and int(fields[1]) == 128 and len(fields) == 2 + 8 * 128
    rows = []
    for at in range(2, len(fields), 8):
        row = dict(zip(('id', 'pair', 'skill', 'correct', 'context', 'query', 'choice0', 'choice1'),
                       fields[at:at + 8]))
        row['correct'] = int(row['correct'])
        rows.append(row)
    return rows


def check_questions(rows):
    assert len(rows) == len({r['id'] for r in rows}) == 128
    assert Counter(r['skill'] for r in rows) == {key: 16 for key in PATTERNS}
    z = sp.Symbol('answer')
    groups = defaultdict(list)
    solutions, displayed = [], {}
    for row in rows:
        skill = row['skill']
        lines = row['context'].splitlines()
        assert len(lines) == 4 and lines[-1] == ''
        facts = {}
        for name, text in zip(('A', 'B'), lines[1:3]):
            assert text.startswith('Trial ' + name + ': ')
            match = re.fullmatch(PATTERNS[skill], text[len('Trial A: '):])
            assert match, 'Unexpected displayed quantitative facts'
            facts[name] = tuple(map(sp.Integer, match.groups()))
        query = re.fullmatch(r'Question: For trial ([AB]), (.+)\nAnswer:', row['query'])
        assert query and query[2] == QUESTIONS[skill]
        displayed[row['id']] = (facts[query[1]], facts['B' if query[1] == 'A' else 'A'])
        x, y, *rest = facts[query[1]]
        if skill == 'addition':
            equation, unit = sp.Eq(z - x, y), 'items'
        elif skill == 'subtraction':
            equation, unit = sp.Eq(z + y, x), 'items'
        elif skill == 'multiplication':
            equation, unit = sp.Eq(z / y, x), 'items'
        elif skill == 'division':
            equation, unit = sp.Eq(z * y, x), 'items'
        elif skill == 'speed':
            equation, unit = sp.Eq(z * y, x), 'm/s'
        elif skill == 'net_force':
            assert 'constant mass in an inertial frame' in lines[0]
            equation, unit = sp.Eq(z / x, y), 'N'
        elif skill == 'work':
            assert 'constant and parallel to the displacement in the same direction' in lines[0]
            equation, unit = sp.Eq(z / y, x), 'J'
        else:
            assert 'constant specific heat' in lines[0] and 'No phase change' in lines[0]
            assert 'work and heat loss are negligible' in lines[0]
            equation, unit = sp.Eq(z / (x * y), rest[0]), 'J'
        solution = sp.solve(equation, z)
        assert len(solution) == 1 and solution[0].is_Integer and solution[0] > 0
        expected = f' {solution[0]} {unit}'
        choices = [row['choice0'], row['choice1']]
        assert row['correct'] in (0, 1) and choices[row['correct']] == expected
        assert len(choices[0].encode()) == len(choices[1].encode()) and choices[0] != choices[1]
        assert all(re.fullmatch(r' [1-9]\d* ' + re.escape(unit), choice) for choice in choices)
        # Copying any visible integer cannot supply either answer, including unit exponents.
        assert all(int(choice.split()[0]) not in set(map(int, re.findall(r'\d+', row['context'])))
                   for choice in choices)
        assert len(row['context']) <= 2048 and len(row['query']) <= 512 and max(map(len, choices)) <= 128
        solutions.append(dict(id=row['id'], equation=str(equation), answer=expected))
        groups[row['pair']].append(row)
    assert len(groups) == 32
    for group in groups.values():
        assert len(group) == 4 and len({r['skill'] for r in group}) == 1
        assert len({(r['choice0'], r['choice1']) for r in group}) == 1
        assert len({(r['context'], r['query']) for r in group}) == 4
        for field in ('context', 'query'):
            values = {r[field] for r in group}
            assert len(values) == 2
            assert all({r['correct'] for r in group if r[field] == value} == {0, 1} for value in values)
    # Give context-only and query-only baselines the best possible label per key.
    # Opposite labels under the omitted input still make every full group fail.
    shortcuts = {}
    for label, field in (('context_only', 'context'), ('query_only', 'query'), ('constant', None)):
        correct = joint = 0
        for group in groups.values():
            lookup = {}
            for row in group:
                lookup.setdefault(row[field] if field else '', row['correct'])
            successes = [lookup[row[field] if field else ''] == row['correct'] for row in group]
            correct += sum(successes)
            joint += all(successes)
        assert correct == 64 and joint == 0
        shortcuts[label] = dict(item_accuracy=correct / 128, joint_accuracy=joint / 32)
    # These baselines bind the correct trial but rank answers using only a simple
    # operand statistic. They need no implementation of the requested operation.
    for name, statistic in (('rank_first_operand', lambda v: v[0]),
                            ('rank_last_operand', lambda v: v[-1]),
                            ('rank_largest_operand', max), ('rank_smallest_operand', min)):
        correct, joint, skills = 0, 0, defaultdict(lambda: [0, 0])
        for group in groups.values():
            successes = []
            for row in group:
                chosen, other = displayed[row['id']]
                first, second = statistic(chosen), statistic(other)
                answers = [int(row[f'choice{i}'].split()[0]) for i in (0, 1)]
                predicted = (-1 if first == second else answers.index(
                    min(answers) if first < second else max(answers)))
                successes.append(predicted == row['correct'])
            correct += sum(successes)
            joint += all(successes)
            skills[group[0]['skill']][0] += all(successes)
            skills[group[0]['skill']][1] += 1
        shortcuts[name] = dict(item_accuracy=correct / 128, joint_accuracy=joint / 32,
            per_skill={k: dict(correct_groups=v[0], groups=v[1], joint_accuracy=v[0]/v[1])
                       for k, v in skills.items()})
    return solutions, shortcuts


def audit(spec_path, prepared, report):
    assert not report.exists(), 'Preserve earlier reports'
    spec, manifest = read(spec_path), read(prepared / 'manifest.json')
    assert manifest['source_spec_sha256'] == sha(spec_path)
    assert all(sha(path) == value for path, value in manifest['authenticated_inputs'].items())
    assert all(sha(prepared / name) == value for name, value in manifest['prepared_files'].items())
    assert {p.name for p in prepared.iterdir()} == {'manifest.json', *manifest['prepared_files']}
    rows = decode((prepared / 'development.sgprobe').read_bytes())
    assert rows == read(prepared / 'questions.json')
    solutions, shortcuts = check_questions(rows)
    if spec['version'] == 'quantitative-development-v2':
        assert all(row['correct_groups'] <= 2 for key, baseline in shortcuts.items()
                   if key.startswith('rank_') for row in baseline['per_skill'].values())
    rejected = []
    for name, edit in (
        ('incorrect-gold', lambda r: r[0].update(correct=1-r[0]['correct'])),
        ('wrong-unit', lambda r: r[0].update(choice0=r[0]['choice0'].replace('items', 'kg'))),
        ('changed-operand', lambda r: r[0].update(context=r[0]['context'].replace('26 items', '27 items'))),
        ('wrong-question', lambda r: r[0].update(query=r[0]['query'].replace('now?', 'removed?'))),
        ('missing-inertial-frame', lambda r: next(x for x in r if x['skill']=='net_force').update(
            context=next(x for x in r if x['skill']=='net_force')['context'].replace('in an inertial frame', ''))),
        ('work-angle', lambda r: next(x for x in r if x['skill']=='work').update(
            context=next(x for x in r if x['skill']=='work')['context'].replace('same direction', 'opposite direction'))),
        ('phase-change', lambda r: next(x for x in r if x['skill']=='heat').update(
            context=next(x for x in r if x['skill']=='heat')['context'].replace('No phase change', 'A phase change'))),
        ('duplicate-item', lambda r: r.__setitem__(1, deepcopy(r[0]))),
    ):
        altered = deepcopy(rows)
        edit(altered)
        assert altered != rows
        try:
            check_questions(altered)
        except AssertionError:
            rejected.append(name)
        else:
            raise AssertionError('Invalid displayed question accepted: ' + name)
    _, statements, _ = build(spec)
    assert statements == read(prepared / 'protected-statements.json')
    with tempfile.TemporaryDirectory(prefix='quantitative-probes-') as folder:
        temporary = Path(folder)
        for name, body in (('whole-context', rows[0]['context']),
                           ('trial-statement', statements[0]),
                           ('whitespace-changed-trial', statements[0].replace(' ', '\n  '))):
            path = temporary / 'contaminated.dat'
            path.write_text('New lesson\n\n' + body + '\n\nMore unrelated text.', encoding='utf-8')
            try:
                protect(rows, statements, [dict(path=path, sha256=sha(path))])
            except ValueError as error:
                assert 'proposed training stream' in str(error)
                rejected.append(name)
            else:
                raise AssertionError('Evaluation text admitted for learning')
        altered = deepcopy(spec)
        altered['skills'][0]['cases'][0]['trials'][0][0] = 0
        path = temporary / 'bad-spec.json'
        write(path, altered)
        destination = temporary / 'rejected'
        try:
            prepare(path, destination)
        except ValueError as error:
            assert 'Trial operands' in str(error) and not destination.exists()
            rejected.append('invalid-trial-before-output')
        else:
            raise AssertionError('Invalid operands produced output')
    assert all(sha(path) == value for path, value in manifest['authenticated_inputs'].items())
    result = dict(passed=True, version=spec['version'], source_spec_sha256=sha(spec_path),
        manifest_sha256=sha(prepared / 'manifest.json'), prepared_files=manifest['prepared_files'],
        items=128, groups=32, skills=8, independent_solver='SymPy equations from parsed displayed facts',
        solver_version=sp.__version__, all_displayed_answers_verified=True,
        assumptions_and_answer_units_verified=True, all_choice_byte_lengths_equal=True,
        crossed_context_query_groups_verified=True, visible_integer_copy_baseline_accuracy=0,
        shortcut_controls=shortcuts, rejected_inputs=rejected, solutions=solutions,
        input_hashes_unchanged=True, audit_script_sha256=sha(Path(__file__)),
        native_commands=0, model_quality_verified=False, training_admitted=False,
        reserved_tests_scored=False, reproduction_admitted=False)
    report.parent.mkdir(parents=True, exist_ok=True)
    write(report, result)
    print('Verified 128 displayed answers, 32 crossed groups, seven shortcut controls and',
          len(rejected), 'rejections; no model execution.')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', type=Path, default=ROOT / 'data/quantitative-probes-v1.json')
    parser.add_argument('--prepared', type=Path, default=Path('data/prepared/quantitative-development-v1'))
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    audit(args.spec, args.prepared, args.report)
