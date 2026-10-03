"""Check the actual Physics review bundle, source rejection and review evidence."""
import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from corpus.openstax import CN
from corpus.physics import extract_physics
from review_physics import authenticate, digest, review, source_bundle, write
from physics_extraction import check


def audit(spec_path, cache, prepared, cases_path):
    spec_raw = spec_path.read_bytes()
    spec = json.loads(spec_raw)
    report_path = prepared / 'result.json'
    report = json.loads(report_path.read_text(encoding='utf-8'))
    assert report['complete'] and not report['training_admitted'] and report['native_calls'] == 0
    assert report['review_spec_sha256'] == digest(spec_raw)
    assert (prepared / 'review-spec.json').read_bytes() == spec_raw
    auxiliary, sources = source_bundle(spec, cache)
    rows = report['records']
    assert [r['module'] for r in rows] == [r['module'] for r in spec['modules']]
    assert report['authenticated_modules'] == len(sources) == 100
    titles = {r['module']: r['title'] for r in spec['modules']}
    totals, policies, extraction, punctuation, failures = Counter(), Counter(), Counter(), Counter(), Counter()
    for row, source in zip(rows, spec['modules']):
        ident = row['module']
        assert all(row[key] == source[key] for key in
                   ('raw_sha256', 'git_blob_sha1', 'chapter', 'title', 'path', 'review_role'))
        if row['review_role'] == 'attribution_only':
            assert not row['candidate_extracted'] and not (prepared / 'review-text' / f'{ident}.txt').exists()
            continue
        totals['content_modules'] += 1
        try:
            body, stats = extract_physics(sources[ident], titles)
        except ValueError as error:
            assert not row['candidate_extracted'] and row['rejection'] == str(error)
            assert not (prepared / 'review-text' / f'{ident}.txt').exists()
            failures[str(error)] += 1
        else:
            assert row['candidate_extracted'] and digest(body) == row['text_sha256']
            assert row['text_file'] == f'review-text/{ident}.txt'
            assert body == (prepared / row['text_file']).read_bytes()
            assert len(body) == row['text_bytes']
            words = len(re.findall(r"[A-Za-z]+(?:['-][A-Za-z]+)*", body.decode('utf-8')))
            assert words == row['word_like_units']
            assert all(row[key] == value for key, value in stats.items())
            totals['candidate_modules'] += 1
            totals['candidate_bytes'] += len(body)
            totals['candidate_word_like_units'] += words
            policies.update(stats['policy'])
            extraction.update(stats['extraction'])
            punctuation.update(stats['punctuation_repairs'])
        assert digest(sources[ident]) == digest((cache / 'raw/physics' / f'{ident}.cnxml').read_bytes())
    assert all(report['totals'][key] == value for key, value in totals.items())
    assert report['policy_totals'] == policies and report['extraction_totals'] == extraction
    assert report['punctuation_repairs'] == punctuation and report['candidate_rejections'] == failures
    assert len(list((prepared / 'review-text').glob('*.txt'))) == totals['candidate_modules'] == 96
    assert {r['module'] for r in rows if 'rejection' in r} == {'m54192', 'm54314', 'm54273'}
    for name, target in [('license', 'LICENSE-source.txt'), ('collection', 'original-collection.xml'),
                         ('preface', 'original-preface.cnxml')]:
        assert (prepared / target).read_bytes() == auxiliary[name]
    assert (prepared / 'ATTRIBUTION.txt').read_bytes() == (spec['attribution'] + '\n').encode('utf-8')
    assert not (prepared / 'train.dat').exists() and not (prepared / 'curriculum.sg').exists()
    assert all(digest(Path(path).read_bytes()) == expected
               for path, expected in report['implementation_sha256'].items())

    cases_raw = cases_path.read_bytes()
    cases = json.loads(cases_raw)
    assert cases['status'] == 'sampled_source_review_not_admission'
    assert cases['upstream_commit'] == spec['upstream_commit']
    for case in cases['records']:
        raw = sources[case['module']]
        assert digest(raw) == case['source_sha256']
        nodes = [n for n in ET.fromstring(raw).iter() if n.get('id') == case['element_id']]
        assert len(nodes) == 1 and nodes[0].tag == CN + case['element_type']
        assert digest(ET.tostring(nodes[0], encoding='utf-8')) == case['element_serialization_sha256']
        if 'candidate_fragment' in case:
            text = (prepared / 'review-text' / f"{case['module']}.txt").read_text(encoding='utf-8')
            assert case['candidate_fragment'] in text

    negative = []

    def rejected(label, callback):
        try:
            callback()
        except ValueError:
            negative.append(label)
        else:
            raise AssertionError('Corrupted source/review accepted: ' + label)

    first = spec['modules'][0]
    rejected('changed_source_bytes', lambda: authenticate(sources[first['module']] + b'\n', first))
    rejected('changed_git_blob_pin', lambda: authenticate(sources[first['module']], dict(first, git_blob_sha1='0' * 40)))
    rejected('changed_source_length', lambda: authenticate(sources[first['module']], dict(first, bytes=0)))
    for label in ('module_order', 'module_title', 'module_source_pin', 'duplicate_module', 'unknown_role',
                  'preface_used_as_content', 'unreviewed_status', 'wrong_license', 'missing_auxiliary'):
        changed = copy.deepcopy(spec)
        if label == 'module_order': changed['modules'][0:2] = reversed(changed['modules'][0:2])
        elif label == 'module_title': changed['modules'][1]['title'] = 'Altered source title'
        elif label == 'module_source_pin': changed['modules'][1]['raw_sha256'] = '0' * 64
        elif label == 'duplicate_module': changed['modules'].append(changed['modules'][0])
        elif label == 'unknown_role': changed['modules'][1]['review_role'] = 'admitted'
        elif label == 'preface_used_as_content': changed['modules'][0]['review_role'] = 'content_candidate'
        elif label == 'unreviewed_status': changed['status'] = 'training_admitted'
        elif label == 'wrong_license': changed['license_url'] = 'https://example.com/license'
        elif label == 'missing_auxiliary': changed['auxiliary_files'].pop()
        rejected(label, lambda: source_bundle(changed, cache))
    rejected('existing_output_directory', lambda: review(spec_path, cache, prepared))
    return dict(passed=True, scope='Source/extraction verification, not a factual or training-quality pass',
        review_spec_sha256=digest(spec_raw), source_report_sha256=digest(report_path.read_bytes()),
        case_file_sha256=digest(cases_raw), sampled_observations_authenticated=len(cases['records']),
        source_modules_authenticated=100, content_modules_checked=totals['content_modules'],
        candidate_modules_reproduced=totals['candidate_modules'], candidate_bytes=totals['candidate_bytes'],
        candidate_word_like_units=totals['candidate_word_like_units'],
        remaining_source_rejections=dict(failures), rejected_corruption_cases=negative,
        fixtures=check(), source_bytes_unchanged=True, attribution_and_original_notices_preserved=True,
        preface_excluded_from_candidates=True, training_admitted=False, native_calls=0, models_trained=False,
        implementation_sha256={p.as_posix(): digest(p.read_bytes()) for p in
            (Path(__file__).relative_to(Path.cwd()), Path('tests/physics_extraction.py'))})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', type=Path, default=Path('data/physics-source-review-v1.json'))
    parser.add_argument('--cases', type=Path, default=Path('data/physics-review-cases-v1.json'))
    parser.add_argument('--cache', type=Path, default=Path('runs/science-source-review'))
    parser.add_argument('--review', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists(), 'Use a fresh verification report'
    result = audit(args.spec, args.cache, args.review, args.cases)
    write(args.out, result)
    print(json.dumps({k: result[k] for k in ('passed', 'content_modules_checked', 'candidate_modules_reproduced',
                'sampled_observations_authenticated', 'rejected_corruption_cases', 'fixtures')}, indent=2))
