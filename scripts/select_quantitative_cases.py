"""Select fixed numeric contrasts against declared heuristics before model evaluation."""
import argparse
from collections import defaultdict
from copy import deepcopy
import itertools
from pathlib import Path
import random

from native_experiment import read, sha
from prose_founder import write
from quantitative_probes import TASKS, answer, build


ROOT = Path(__file__).resolve().parents[1]
SEED = 20261004


def trial(skill, rng):
    if skill == 'addition':
        return [rng.randint(12, 69), rng.randint(12, 69)]
    if skill == 'subtraction':
        return [rng.randint(40, 99), rng.randint(11, 39)]
    if skill in ('division', 'speed'):
        numerator, denominator = rng.randint(12, 39), rng.randint(2, 7)
        return [numerator * denominator, denominator]
    return [rng.randint(2, 7) for _ in range(3)] if skill == 'heat' else [rng.randint(3, 15) for _ in range(2)]


def select(base_path, out, report):
    if out.exists() or report.exists():
        raise ValueError('Use fresh selection and report paths')
    base = read(base_path)
    build(base)
    result = deepcopy(base)
    result['version'] = 'quantitative-development-v2'
    rng, records = random.Random(SEED), []
    for skill in result['skills']:
        options = defaultdict(list)
        for _ in range(20000):
            values = [trial(skill['id'], rng), trial(skill['id'], rng)]
            try:
                gold = [int(answer(skill['id'], v)) for v in values]
            except ValueError:
                continue
            if any(g < 10 or g > 99 or g in [x for v in values for x in v] for g in gold) or gold[0] == gold[1]:
                continue
            ranked = [[f(v) for v in values] for f in (lambda v: v[0], lambda v: v[-1], max, min)]
            if any(a == b for a, b in ranked):
                continue
            mask = tuple(int((a < b) == (gold[0] < gold[1])) for a, b in ranked) + (int(gold[0] < gold[1]),)
            if len(options[mask]) < 8 and values not in options[mask]:
                options[mask].append(values)
        keys = sorted(options)
        combinations = itertools.chain(itertools.combinations(keys, 4),
                                       itertools.combinations_with_replacement(keys, 4))
        masks = next((combo for combo in combinations
                      if all(sum(m[i] for m in combo) == 2 for i in range(5))), None)
        if masks is None:
            raise ValueError('Could not balance declared numerical controls: ' + skill['id'])
        selected = [options[mask].pop(0) for mask in masks]
        assert len({tuple(map(tuple, v)) for v in selected}) == 4
        skill['cases'] = [dict(id=f"{skill['id']}-{i+1}", trials=v) for i, v in enumerate(selected)]
        records.append(dict(skill=skill['id'], candidate_draws=20000, available_masks=len(keys),
            chosen_masks=masks, ranking_heuristic_correct_groups=[sum(m[i] for m in masks) for i in range(4)],
            first_trial_answer_smaller_groups=sum(m[4] for m in masks)))
    result['case_selection'] = dict(seed=SEED, base_spec_sha256=sha(base_path), model_evaluations=0,
        controls=['rank_first_operand', 'rank_last_operand', 'rank_largest_operand', 'rank_smallest_operand'],
        policy='Choose four groups per skill with exactly two successful groups for each declared operand-ranking heuristic, no operand-statistic ties, and two cases where the first trial answer is smaller. This balances these controls only; it does not rule out other shortcuts.')
    result['policy']['limits'].append('Numbers were selected against four declared operand-ranking heuristics before any model evaluation. Each heuristic passes exactly half the groups in each skill; stronger strategies remain possible.')
    build(result)
    write(out, result)
    evidence = dict(passed=True, seed=SEED, prototype_spec_sha256=sha(base_path), selected_spec_sha256=sha(out),
        selection_script_sha256=sha(Path(__file__)), arithmetic_helper_sha256=sha(ROOT/'scripts/quantitative_probes.py'),
        cases=records, labels_derived_from_arithmetic_not_model_scores=True, model_evaluations=0,
        native_commands=0, independent_displayed_question_audit_required=True)
    write(report, evidence)
    print('Selected 32 numerical groups; each declared ranking heuristic passes 2/4 per skill.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, default=ROOT/'data/quantitative-probes-v1.json')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    select(args.base, args.out, args.report)
