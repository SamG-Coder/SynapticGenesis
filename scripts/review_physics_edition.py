"""Compile a corrected Physics review edition while retaining original sources."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import posixpath
import re
import xml.etree.ElementTree as ET

from corpus.openstax import CN
from corpus.physics import extract_physics, physics_document
from corpus.physics_edits import digest, edit_document, omit_empty_exercises, require
from native_experiment import read
from prepare_corpus import fetch
from prose_founder import write
from review_physics import media_inventory, source_bundle


def compile_edition(spec_path, edits_path, cache, download=False):
    spec, edits = read(spec_path), read(edits_path)
    require(edits['version'] == 'physics-edits-v1' and
            edits['source_review_spec_sha256'] == digest(spec_path.read_bytes()), 'Edit/source edition mismatch')
    require(edits['upstream_commit'] == spec['upstream_commit'], 'Edit/source commit mismatch')
    require(edits['empty_exercise_policy'] == 'omit_whitespace_only_problem_and_exercise_wrappers',
            'Unreviewed empty-exercise policy')
    auxiliary, sources = source_bundle(spec, cache, download)
    titles = {r['module']: r['title'] for r in spec['modules']}
    require(all(r['module'] in sources for r in edits['records']), 'Edit refers to an unknown module')
    for image in edits['inspected_media']:
        path = (cache / image['cache_file']).resolve()
        require(path.is_relative_to(cache.resolve()) and
                re.fullmatch(r'media/[A-Za-z0-9_.-]+\.(jpg|png)', image['upstream_path']), 'Invalid diagram path')
        url = f"https://raw.githubusercontent.com/openstax/osbooks-physics/{spec['upstream_commit']}/{image['upstream_path']}"
        require(image['url'] == url, 'Diagram URL differs from the pinned source')
        if not path.exists() and download:
            path.parent.mkdir(parents=True, exist_ok=True)
            fetch(url, path)
        require(digest(path.read_bytes()) == image['sha256'],
                'Inspected diagram identity changed')
        raw = path.read_bytes()
        require(len(raw) == image['bytes'] and
                hashlib.sha1(f'blob {len(raw)}\0'.encode() + raw).hexdigest() == image['git_blob_sha1'],
                'Inspected diagram length or Git identity changed')
    image_hashes = {r['sha256']:r for r in edits['inspected_media']}
    require(len(image_hashes) == len(edits['inspected_media']), 'Duplicate inspected diagram')
    for record in edits['records']:
        if record['operation'] != 'media_description':
            continue
        require(record.get('media_sha256') in image_hashes, 'Missing inspected diagram')
        original = ET.fromstring(sources[record['module']])
        matches = [e for e in original.iter(CN + 'media') if e.get('id') == record['element_id']]
        require(len(matches) == 1, 'Missing or duplicate described media')
        images = matches[0].findall(CN + 'image')
        require(len(images) == 1 and
                posixpath.normpath(posixpath.join('modules', record['module'], images[0].get('src', ''))) ==
                image_hashes[record['media_sha256']]['upstream_path'], 'Description cites a different diagram')
    bodies, records = {}, []
    for source in spec['modules']:
        if source['review_role'] != 'content_candidate':
            continue
        ident = source['module']
        amended, changes = edit_document(sources[ident], ident, edits['records'])
        transformed, policy = physics_document(amended)
        cleaned, empty = omit_empty_exercises(transformed)
        body, stats = extract_physics(cleaned, titles)
        # The shared source-class transformation is idempotent. Preserve the
        # first pass's counts rather than losing them in the final extraction.
        require(not stats['policy'], 'Source-class transformations were not idempotent')
        stats['policy'] = policy
        media, missing = media_inventory(ET.fromstring(cleaned))
        require(not missing, 'Unresolved placeholder diagram: ' + ident)
        require(not stats['extraction'].get('generic_cross_references'), 'Unresolved reference: ' + ident)
        require(not re.search(rb'(^|\n)Question:\s*(\n\n|$)', body), 'Empty extracted question: ' + ident)
        record = dict(source, edited_elements=changes, amended_source_sha256=digest(amended),
            omitted_empty_wrappers=empty,
            text_sha256=digest(body), text_bytes=len(body),
            word_like_units=len(re.findall(r"[A-Za-z]+(?:['-][A-Za-z]+)*", body.decode('utf-8'))),
            media=media, **stats)
        records.append(record)
        bodies[ident] = body
    require(len(records) == 99, 'The reviewed collection should have 99 content modules')
    require(sum(len(r['edited_elements']) for r in records) == len(edits['records']), 'An edit was not applied')
    return spec, edits, auxiliary, bodies, records


def review(spec_path, edits_path, cache, out, download=False):
    require(not out.exists(), 'Use a fresh corrected-review directory')
    spec, edits, auxiliary, bodies, records = compile_edition(spec_path, edits_path, cache, download)
    (out / 'review-text').mkdir(parents=True)
    for ident, body in bodies.items():
        (out / 'review-text' / f'{ident}.txt').write_bytes(body)
    for name, target in [('license', 'LICENSE-source.txt'), ('collection', 'original-collection.xml'),
                         ('preface', 'original-preface.cnxml')]:
        (out / target).write_bytes(auxiliary[name])
    (out / 'review-spec.json').write_bytes(spec_path.read_bytes())
    (out / 'edits.json').write_bytes(edits_path.read_bytes())
    (out / 'ATTRIBUTION.txt').write_bytes((edits['attribution'] + '\n').encode('utf-8'))
    totals = Counter()
    for r in records:
        for name in ['text_bytes', 'word_like_units']:
            totals[name] += r[name]
        totals['media_descriptions'] += r['media'].get('media_elements', 0)
        totals['retained_formulas'] += r['extraction'].get('math_expressions', 0)
        for kind, elements in r['omitted_empty_wrappers'].items():
            totals['omitted_empty_' + kind] += len(elements)
    result = dict(complete=True, source_review_spec_sha256=digest(spec_path.read_bytes()),
        edits_sha256=digest(edits_path.read_bytes()), upstream_commit=spec['upstream_commit'],
        totals=dict(totals), records=records, inspected_media=edits['inspected_media'],
        modules_extracted=len(records), edited_elements=len(edits['records']),
        training_admitted=False, native_commands=0, source_files_modified=False,
        implementation_sha256={s:digest(Path(s).read_bytes()) for s in
            ['scripts/review_physics_edition.py','scripts/corpus/physics_edits.py','scripts/corpus/physics.py',
             'scripts/corpus/openstax.py','scripts/review_physics.py']},
        limits='This resolves the enumerated source/extraction issues only. Remaining passages and diagram '
               'dependencies still need selection review and protected evaluation splits before admission.')
    write(out / 'result.json', result)
    print(json.dumps({k:result[k] for k in ['modules_extracted','edited_elements','totals','training_admitted']},indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', type=Path, default=Path('data/physics-source-review-v1.json'))
    parser.add_argument('--edits', type=Path, default=Path('data/physics-edits-v1.json'))
    parser.add_argument('--cache', type=Path, default=Path('runs/science-source-review'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--download', action='store_true', help='Fetch only missing pinned source files and diagrams')
    args = parser.parse_args()
    review(args.spec, args.edits, args.cache, args.out, args.download)
