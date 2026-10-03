"""Original quantitative development probes; no model or training implementation."""
from fractions import Fraction
from pathlib import Path

from corpus.paired_selection import normalized
from corpus.reviewed_edition import require
from native_experiment import sha


# The physical assumptions are part of the visible question, not hidden metadata.
TASKS = {
    'addition': ('items', None, 'Each trial describes an independent collection of items.',
        'A collection starts with {0} items and receives {1} more items.',
        'how many items are in the collection now?'),
    'subtraction': ('items', None, 'Each trial describes an independent collection of items.',
        'A collection starts with {0} items and then {1} items are removed.',
        'how many items remain in the collection?'),
    'multiplication': ('items', None, 'Each trial describes an independent rectangular arrangement.',
        'There are {0} rows with {1} items in every row.',
        'how many items are there in total?'),
    'division': ('items', None, 'Each trial describes an independent equal distribution.',
        'There are {0} items shared equally among {1} boxes.',
        'how many items are in each box?'),
    'speed': ('m/s', 'm54104', 'Each trial describes an independent journey.',
        'The total distance travelled is {0} m and the elapsed time is {1} s.',
        'what is the average speed in m/s?'),
    'net_force': ('N', 'm54142', 'Each trial uses constant mass in an inertial frame.',
        'An object has mass {0} kg and acceleration magnitude {1} m/s^2.',
        'what is the magnitude of the net external force in N?'),
    'work': ('J', 'm54271', 'In each trial, the force is constant and parallel to the displacement in the same direction.',
        'The force magnitude is {0} N and the displacement magnitude is {1} m.',
        'what is the work done by the stated force in J?'),
    'heat': ('J', 'm54290', 'Each hypothetical sample has constant specific heat. No phase change occurs; work and heat loss are negligible.',
        'A sample has mass {0} kg, specific heat {1} J/(kg K), and temperature rise {2} K.',
        'how much heat is supplied in J?'),
}


def answer(skill, values):
    x, y = map(Fraction, values[:2])
    if skill == 'addition':
        result = x + y
    elif skill == 'subtraction':
        result = x - y
    elif skill in ('division', 'speed'):
        result = x / y
    else:
        result = x * y
        if skill == 'heat':
            result *= Fraction(values[2])
    require(result > 0 and result.denominator == 1, 'Answer must be a positive exact integer')
    return str(result.numerator)


def build(spec):
    require(spec['version'] == 'quantitative-development-v1' and spec['native_format'] == 'SGPROBE2'
            and spec['group_size'] == spec['groups_per_skill'] == 4, 'Unexpected quantitative suite')
    require(spec['policy']['development_only'] and not spec['policy']['training_admitted'] and
            not spec['policy']['test_set_created'], 'This suite is only for development scoring')
    require([s['id'] for s in spec['skills']] == list(TASKS), 'Quantitative skill inventory differs')
    rows, statements, groups, identities = [], [], [], set()
    for skill in spec['skills']:
        ident = skill['id']
        unit, module, intro, statement, question = TASKS[ident]
        require(skill['answer_unit'] == unit and skill['reviewed_training_module'] == module and
                len(skill['cases']) == 4, 'Concept, unit or numerical group count differs')
        for number, case in enumerate(skill['cases']):
            group = case['id']
            require(group == f'{ident}-{number + 1}' and group not in identities,
                    'Invalid or duplicate quantitative group')
            identities.add(group)
            values = case['trials']
            require(len(values) == 2 and all(len(v) == (3 if ident == 'heat' else 2) for v in values)
                    and all(type(x) is int and 0 < x < 10000 for v in values for x in v),
                    'Trial operands must be positive integers with the required arity')
            gold = [answer(ident, v) for v in values]
            require(gold[0] != gold[1] and all(int(g) not in {x for v in values for x in v} for g in gold),
                    'Answers must differ and must not copy a supplied operand')
            labels = list(reversed(gold)) if number % 2 else gold
            choices = [f' {g} {unit}' for g in labels]
            require(len(choices[0].encode('ascii')) == len(choices[1].encode('ascii')),
                    'Candidate answer byte lengths must match')
            facts = [statement.format(*v) for v in values]
            statements.extend(facts)
            for reverse in (0, 1):
                ordered = facts[::-1] if reverse else facts
                context = intro + '\nTrial A: ' + ordered[0] + '\nTrial B: ' + ordered[1] + '\n\n'
                for trial, name in enumerate(('A', 'B')):
                    correct = labels.index(gold[trial ^ reverse])
                    rows.append(dict(id=f'{group}-{reverse}-{name}', pair=group, skill=ident,
                        correct=correct, context=context,
                        query=f'Question: For trial {name}, {question}\nAnswer:',
                        choice0=choices[0], choice1=choices[1]))
            groups.append(dict(id=group, skill=ident, trial_values=values, exact_answers=gold,
                               answer_unit=unit))
    return rows, list(dict.fromkeys(statements)), groups


def protect(rows, statements, sources):
    """Reject exact authored contexts or trial statements in a proposed learning stream."""
    contexts = {normalized(r['context']) for r in rows}
    facts = {normalized(s) for s in statements}
    require(len(sources) > 0, 'No training streams supplied for overlap protection')
    verified = {}
    for row in sources:
        path = row['path']
        require(sha(path) == row['sha256'], 'Proposed training stream changed: ' + str(path))
        text = normalized(Path(path).read_text(encoding='utf-8'))
        require(not any(context in text for context in contexts), 'Evaluation context in proposed training stream')
        require(not any(fact in text for fact in facts), 'Evaluation trial statement in proposed training stream')
        verified[str(path)] = row['sha256']
    return verified
