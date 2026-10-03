"""Verify real source corrections, empty-wrapper boundaries and source identity."""
import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from corpus.openstax import CN, MATH
from corpus.physics_edits import digest, edit_document, omit_empty_exercises
from native_experiment import read
from prose_founder import write
from review_physics_edition import compile_edition


def check(directory, report):
    spec_path, edits_path = Path('data/physics-source-review-v1.json'), Path('data/physics-edits-v1.json')
    cache = Path('runs/science-source-review')
    spec, edits, _, bodies, records = compile_edition(spec_path, edits_path, cache)
    result = read(directory / 'result.json')
    assert result['records'] == records and result['modules_extracted'] == 99
    assert result['edits_sha256'] == digest(edits_path.read_bytes())
    for name, value in result['implementation_sha256'].items():
        assert digest(Path(name).read_bytes()) == value
    for ident, body in bodies.items():
        assert body == (directory / 'review-text' / f'{ident}.txt').read_bytes()
        assert not any(p == b'Question:' for p in body.split(b'\n\n'))
    prior = read('reports/physics-source-review.json')
    changed_modules = {r['module'] for r in edits['records']}
    original_dir = Path('runs/physics-source-audit-v1/reproducible-v3/review-text')
    preserved = []
    for row in prior['records']:
        if row['module'] in changed_modules or not row['candidate_extracted']:
            continue
        original = (original_dir / f"{row['module']}.txt").read_bytes()
        assert digest(original) == row['text_sha256']
        # A separately expressed output check: the only change to these 93
        # modules is removal of empty Question blocks. All other bytes remain.
        cleaned = b'\n\n'.join(p for p in original.rstrip(b'\n').split(b'\n\n') if p != b'Question:') + b'\n'
        assert bodies[row['module']] == cleaned, row['module']
        preserved.append(row['module'])
    assert len(preserved) == 93
    kepler = bodies['m54192'].decode()
    for row in [r'Kepler’s third law | \(\frac{{T}_{1}^{2}}{{T}_{2}^{2}} = \frac{{r}_{1}^{3}}{{r}_{2}^{3}}\)',
                r'eccentricity | \(e = \frac{f}{a}\)', r'area of an ellipse | \(A = π a b\)',
                r'semi-major axis of an ellipse | \(a = ( {r}_{\text{a}} + {r}_{\text{p}} ) / 2\)',
                r'semi-minor axis of an ellipse | \(b = \sqrt{{r}_{\text{a}} {r}_{\text{p}}}\)']:
        assert row in kepler
    assert b'longitudinal wave | mechanical wave | medium | wave\nperiodic wave | pulse wave | transverse wave |' in bodies['m54314']
    energy = bodies['m54273'].decode()
    positions = [energy.index(label) for label in ['(I) An object', '(II) An object', '(III) An object', '(IV) An object',
                                                  'd. III and IV only', 'Answer: The correct answer is (d)']]
    assert positions == sorted(positions)
    nucleus = bodies['m54580'].decode()
    # Two repaired occurrences plus the already correct later decay question.
    assert nucleus.count(r'\({}_{Z}^{A} X {}_{N}\)') == 3 and r'\({}_{A}^{Z} X {}_{N}\)' not in nucleus
    assert r'\({}_{1}^{2} \text{H} {}_{1}\)' in nucleus and r'\({}_{1}^{2} \text{H} {}_{2}\)' not in nucleus
    assert r'\({}_{1}^{3} \text{H} {}_{2}\)' in nucleus
    for module in ['m54357', 'm54365']:
        rule = next(r for r in edits['records'] if r['module'] == module)
        assert rule['after'] in bodies[module].decode() and 'Image description: ...' not in bodies[module].decode()
    refraction_angle = math.degrees(math.asin(1.33 * math.sin(math.radians(25))))
    assert round(refraction_angle) == 34 and 'd. 34°' in bodies['m54365'].decode()
    assert 'is 25°' in bodies['m54365'].decode()
    fixture = f'''<document xmlns="{CN[1:-1]}" xmlns:m="{MATH[1:-1]}"><title>Fixture</title><content>
      <para>Before.</para><exercise id="empty"><problem id="empty-problem"><para> </para></problem></exercise>After.
      <exercise id="text"><problem>Keep this question.</problem><solution>Keep its solution.</solution></exercise>
      <exercise id="formula"><problem><m:math><m:mn>2</m:mn></m:math></problem></exercise>
      <exercise id="diagram"><problem><media alt="A diagram"><image src="x.png"/></media></problem></exercise>
      <exercise id="reference"><problem><link target-id="text"/></problem></exercise>
      </content></document>'''.encode()
    cleaned, omitted = omit_empty_exercises(fixture)
    parsed = ET.fromstring(cleaned)
    assert omitted == {'problem':['empty-problem'], 'exercise':['empty']}
    assert {e.get('id') for e in parsed.iter(CN + 'exercise')} == {'text','formula','diagram','reference'}
    assert 'Before.After.' in ''.join(parsed.itertext()).replace('\n', '').replace(' ', '')
    assert 'Keep its solution.' in ''.join(parsed.itertext())
    rejected = []
    def reject(label, action):
        try: action()
        except (ValueError, AssertionError): rejected.append(label)
        else: raise AssertionError('Altered source edit accepted: ' + label)
    first = edits['records'][0]
    raw = (cache / 'raw/physics' / f"{first['module']}.cnxml").read_bytes()
    reject('changed_source', lambda: edit_document(raw + b' ', first['module'], edits['records']))
    for label in ['element_hash','unknown_operation','missing_target','duplicate_target','old_columns','new_columns',
                  'roman_labels','formula_hash','duplicate_formula','wrong_formula_namespace','media_before']:
        changed = deepcopy(edits['records'])
        if label == 'element_hash': changed[0]['element_sha256'] = '0'*64
        elif label == 'unknown_operation': changed[0]['operation'] = 'guess'
        elif label == 'missing_target': changed[0]['element_id'] = 'missing'
        elif label == 'duplicate_target': changed.append(deepcopy(changed[0]))
        elif label == 'old_columns': changed[0]['old_columns'] = 2
        elif label == 'new_columns': changed[0]['new_columns'] = 3
        elif label == 'roman_labels': changed[2]['labels'][1] = '(VII)'
        elif label == 'formula_hash': changed[3]['formulas'][0]['before_sha256'] = '0'*64
        elif label == 'duplicate_formula': changed[3]['formulas'].append(deepcopy(changed[3]['formulas'][0]))
        elif label == 'wrong_formula_namespace': changed[3]['formulas'][0]['replacement_xml'] = '<math><mi>Z</mi></math>'
        elif label == 'media_before': changed[5]['before'] = 'Different source'
        index = 2 if label == 'roman_labels' else 3 if 'formula' in label else 5 if label == 'media_before' else 0
        module = changed[index]['module']
        raw = (cache / 'raw/physics' / f'{module}.cnxml').read_bytes()
        reject(label, lambda: edit_document(raw, module, changed))
    with TemporaryDirectory() as temporary:
        path = Path(temporary) / 'edits.json'
        for label in ['source_spec_hash','unknown_empty_policy','wrong_image_binding','image_hash','image_url',
                      'image_path_escape','image_git_identity','unapplied_preface_edit']:
            altered = deepcopy(edits)
            if label == 'source_spec_hash': altered['source_review_spec_sha256'] = '0'*64
            elif label == 'unknown_empty_policy': altered['empty_exercise_policy'] = 'drop_questions'
            elif label == 'wrong_image_binding': altered['records'][5]['media_sha256'] = altered['inspected_media'][1]['sha256']
            elif label == 'image_hash': altered['inspected_media'][0]['sha256'] = '0'*64
            elif label == 'image_url': altered['inspected_media'][0]['url'] = 'https://example.invalid/different.jpg'
            elif label == 'image_path_escape': altered['inspected_media'][0]['cache_file'] = '../outside.jpg'
            elif label == 'image_git_identity': altered['inspected_media'][0]['git_blob_sha1'] = '0'*40
            elif label == 'unapplied_preface_edit':
                rule = deepcopy(altered['records'][0]); rule['module'] = 'm54081'; altered['records'].append(rule)
            path.write_bytes((json.dumps(altered)+'\n').encode())
            reject(label, lambda: compile_edition(spec_path,path,cache))
    for row in spec['modules']:
        assert digest((cache / 'raw/physics' / f"{row['module']}.cnxml").read_bytes()) == row['raw_sha256']
    write(report,dict(passed=True,review_result_sha256=digest((directory/'result.json').read_bytes()),
        modules_recompiled=99,unaltered_modules_exact_except_empty_question_blocks=preserved,
        explicit_table_rows_verified=7,roman_labels_and_answer_preserved=True,
        corrected_formulas_checked=3,inspected_descriptions_preserved=2,
        refraction_question_independent_angle_degrees=refraction_angle,empty_wrapper_boundaries_checked=True,
        rejected_cases=rejected,original_source_modules_unchanged=100,new_native_commands=0,
        training_admitted=False,implementation_sha256={s:digest(Path(s).read_bytes()) for s in
            ['tests/physics_edits.py','scripts/review_physics_edition.py','scripts/corpus/physics_edits.py']},
        limits='Checks the explicit edits, identities and empty-wrapper behavior. This is not full-book factual certification or independent validation of all 565 diagrams.'))
    print('Verified 99 modules, preserved 93 outside explicit edits, and rejected',len(rejected),'altered cases.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--review', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    assert not args.report.exists(), 'Use a fresh report'
    check(args.review,args.report)
