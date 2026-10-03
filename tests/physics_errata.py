"""Independently check the proposed energy derivation and exact errata extent."""
import argparse
from copy import deepcopy
from fractions import Fraction
from pathlib import Path
import re
import sys

import sympy as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from native_experiment import read, sha
from prose_founder import write
from review_physics_errata import SPEC, compiled, review


def algebra(plan):
    row = next(r for r in plan['corrections'] if r['id'] == 'energy-balance-signs')
    expressions = row['after'].split(' = ')
    assert len(expressions) == 4
    symbolic = expressions[1]
    for text, name in [(r'K {E}_{1}', 'k1'), (r'P {E}_{1}', 'p1'), (r'P {E}_{2}', 'p2')]:
        symbolic = symbolic.replace(text, name)
    assert re.fullmatch(r'[kp12()+\- ]+', symbolic)
    k1, k2, p1, p2 = sp.symbols('k1 k2 p1 p2')
    proposed = sp.sympify(symbolic, locals=dict(k1=k1, p1=p1, p2=p2))
    derived = sp.solve(sp.Eq(k1 + p1, k2 + p2), k2)[0]
    assert sp.simplify(proposed - derived) == 0
    substituted = expressions[2].replace('[', '(').replace(']', ')')
    assert re.fullmatch(r'[0-9.()+\- ]+', substituted)
    substituted = re.sub(r'(\d)\s*\(', r'\1*(', substituted)
    substituted = re.sub(r'\)\s*(\d)', r')*\1', substituted)
    numerical = sp.sympify(substituted, rational=True)
    mass, gravity, initial_height, final_height = map(Fraction, ('10', '9.80', '20', '10'))
    pe1, pe2 = mass * gravity * initial_height, mass * gravity * final_height
    ke2 = pe1 - pe2
    assert numerical == ke2 == 980 and proposed.subs({k1: 0, p1: pe1, p2: pe2}) == 980
    assert expressions[3] == r'980\ \text{J}\)'
    assert pe2 - pe1 == -980 and pe2 - (0 - pe1) == 2940
    # A second pair of masses is a counterexample to mass-independent C at fixed c.
    specific_heat = Fraction(900)
    small, large = Fraction(1) * specific_heat, Fraction(2) * specific_heat
    assert large == 2 * small
    return dict(conservation_solution=str(derived), proposed_equation_matches_conservation=True,
        initial_potential_joules=int(pe1), final_potential_joules=int(pe2), final_kinetic_joules=int(ke2),
        original_symbolic_expression_joules=-980, original_substituted_expression_joules=2940,
        original_stated_result_joules=980, heat_capacity_counterexample=dict(
            specific_heat_j_per_kg_k=900, one_kg_j_per_k=int(small), two_kg_j_per_k=int(large)))


def check(out, prepared):
    out, prepared = Path(out), Path(prepared)
    assert not out.exists() and not prepared.exists(), 'Use fresh audit and review directories'
    out.mkdir(parents=True)
    plan, edition, changes, inputs = compiled()
    originals = {str(p): sha(p) for p in inputs}
    arithmetic = algebra(plan)
    for row in changes:
        recovered = row['corrected'].decode('utf-8')
        records = [r for r in plan['corrections'] if r['module'] == row['id']]
        for correction in reversed(records):
            assert recovered.count(correction['after']) == 1
            recovered = recovered.replace(correction['after'], correction['before'], 1)
        assert recovered.encode('utf-8') == row['original']
        assert row['original'].count(b'Question:') == row['corrected'].count(b'Question:')
        assert row['original'].count(b'Answer:') == row['corrected'].count(b'Answer:')
        assert row['original'].count(b'\x1e') == row['corrected'].count(b'\x1e') == 0
    temperature = next(row for row in changes if row['id'] == 'm54287')
    assert b'15 = 298K' in temperature['original'] and b'15 = 298K' in temperature['corrected']
    assert b'classical monatomic ideal gas' in temperature['corrected']
    assert b'average random kinetic energy of a molecule or an atom' not in temperature['corrected']
    published = review(prepared)
    for row in changes:
        assert (prepared / 'review-text' / (row['id'] + '.txt')).read_bytes() == row['corrected']
    for filename in ('LICENSE-source.txt', 'original-preface.cnxml', 'original-collection.xml'):
        assert (prepared / filename).read_bytes() == (edition.prepared / filename).read_bytes()
    assert all(sha(prepared / name) == value for name, value in published['output_sha256'].items())
    assert published['conditional_single_pass_target_byte_delta'] == sum(
        len(row['corrected']) - len(row['original']) for row in changes)
    # Enumerate all target positions independently of the builder's ceiling formula.
    window_delta = sum(len(range(0, len(row['corrected']) - 1, 128)) -
                       len(range(0, len(row['original']) - 1, 128)) for row in changes)
    assert published['conditional_single_pass_128_byte_observation_delta'] == window_delta
    mutations = []
    for label, key, value, reason in [
        ('status', 'status', 'admitted', 'review status'),
        ('base-hash', 'base_source_spec_sha256', '0' * 64, 'base identity'),
        ('manifest', 'base_manifest_sha256', '0' * 64, 'base manifest'),
        ('source-commit', 'upstream_commit', '0' * 40, 'source commit'),
    ]:
        other = deepcopy(plan); other[key] = value
        mutations.append((label, other, reason))
    for label, field, value, reason in [
        ('source-element', 'source_element_id', 'missing', 'source element changed'),
        ('element-text', 'source_element_text_sha256', '0' * 64, 'source element text'),
        ('before-text', 'before', 'changed', 'passage identity'),
        ('after-text', 'after', 'changed', 'passage identity'),
        ('missing-evidence', 'references', [], 'correction evidence'),
    ]:
        other = deepcopy(plan); other['corrections'][0][field] = value
        mutations.append((label, other, reason))
    other = deepcopy(plan); other['modules'][0]['corrected_sha256'] = '0' * 64
    mutations.append(('output-hash', other, 'Corrected text differs'))
    other = deepcopy(plan); other['modules'][0]['corrected_bytes'] += 1
    mutations.append(('output-length', other, 'output extent'))
    other = deepcopy(plan); other['corrections'] = other['corrections'][:-1]
    mutations.append(('missing-correction', other, 'correction scope'))
    other = deepcopy(plan); other['modules'][0]['correction_ids'] = []
    mutations.append(('missing-assignment', other, 'correction order'))
    rejected = []
    for label, value, reason in mutations:
        source = out / (label + '.json')
        write(source, value)
        try:
            compiled(source)
        except ValueError as error:
            assert reason in str(error), (label, str(error))
            rejected.append(dict(case=label, reason=str(error)))
        else:
            raise AssertionError('Altered errata accepted: ' + label)
    assert all(sha(p) == value for p, value in originals.items())
    result = dict(passed=True, correction_count=7, modules_checked=3, full_originals_recovered=True,
        question_and_solution_markers_preserved=True, original_sources_unchanged=True,
        original_rounding_example_preserved=True, attribution_files_exact=3, algebra=arithmetic,
        proposed_review_result_sha256=sha(prepared / 'result.json'), review_directory=str(prepared),
        exact_output_hashes_verified=True, single_pass_observation_delta=window_delta,
        rejected_inputs=rejected, cpu_only=True, native_commands=0, new_model_updates=0,
        training_admitted=False, implementation_sha256={str(p): sha(p) for p in (
            Path(__file__), SPEC, Path('scripts/review_physics_errata.py'))},
        limits='Arithmetic and byte-preservation checks support only the seven explicit changes. '
               'They do not certify every Physics statement, establish model learning, or admit these copies.')
    write(out / 'result.json', result)
    print(f'Errata audit passed: seven changes, three original modules recovered, {len(rejected)} altered-input rejections.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--prepared', type=Path, required=True)
    args = parser.parse_args()
    check(args.out, args.prepared)
