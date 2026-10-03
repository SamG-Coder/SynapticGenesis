"""Authenticate and extract a pinned Physics book for review, never admission.

No curriculum, training split or native model is created. Missing source files
can be fetched from the recorded commit with --download; existing cache files
must match their SHA-256 and Git blob identities before extraction begins.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

from corpus.openstax import CN, MATH, OMITTED_NOTES, compact, extract, math_text
from corpus.physics import extract_physics, physics_document, physics_extension
from corpus.prealgebra_math import prealgebra_extension
from prepare_corpus import fetch


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def write(path, value):
    path.write_bytes((json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))


def authenticate(raw, record):
    require(digest(raw) == record['raw_sha256'], 'Source SHA-256 mismatch: ' + record['path'])
    blob = hashlib.sha1(f'blob {len(raw)}\0'.encode() + raw).hexdigest()
    require(blob == record['git_blob_sha1'], 'Source Git blob mismatch: ' + record['path'])
    require('bytes' not in record or len(raw) == record['bytes'], 'Source length mismatch: ' + record['path'])
    return raw


def collection_order(raw):
    col, md = '{http://cnx.rice.edu/collxml}', '{http://cnx.rice.edu/mdml}'
    root = ET.fromstring(raw)
    require(root.tag == col + 'collection', 'Expected a book collection')
    license_node = root.find(col + 'metadata/' + md + 'license')
    require(license_node is not None and license_node.get('url', '').replace('http:', 'https:').rstrip('/')
            == 'https://creativecommons.org/licenses/by/4.0', 'Unexpected collection license')
    rows = []

    def walk(content, chapter):
        require(content is not None, 'Missing collection content')
        for node in content:
            if node.tag == col + 'module':
                rows.append((node.get('document'), chapter))
            elif node.tag == col + 'subcollection':
                title = node.find(md + 'title')
                require(title is not None, 'Untitled collection chapter')
                walk(node.find(col + 'content'), compact(''.join(title.itertext())))
            else:
                raise ValueError('Unreviewed collection structure: ' + node.tag)
    walk(root.find(col + 'content'), 'Front or back matter')
    return rows


def source_bundle(spec, cache, download=False):
    require(spec['status'] == 'research_only_not_admitted', 'This tool only accepts review specifications')
    require(spec['upstream_repository'] == 'https://github.com/openstax/osbooks-physics', 'Unexpected repository')
    require(re.fullmatch(r'[0-9a-f]{40}', spec['upstream_commit']), 'Expected a full commit pin')
    require(spec['license_url'] == 'https://creativecommons.org/licenses/by/4.0/', 'Unexpected source license')
    records = spec['modules']
    require(len({r['module'] for r in records}) == len(records), 'Duplicate module identity')
    require(len(spec['auxiliary_files']) == 3 and
            {r['name'] for r in spec['auxiliary_files']} == {'license', 'collection', 'preface'},
            'Expected exactly one license, collection and preface')
    prefix = f"https://raw.githubusercontent.com/openstax/osbooks-physics/{spec['upstream_commit']}/"

    def load(record, relative):
        path = cache / relative
        require(path.resolve().is_relative_to(cache.resolve()), 'Cache path leaves its directory')
        if not path.exists() and download:
            path.parent.mkdir(parents=True, exist_ok=True)
            fetch(prefix + record['path'], path)
        return authenticate(path.read_bytes(), record)

    auxiliary = {r['name']: load(r, r['cache_file']) for r in spec['auxiliary_files']}
    require(auxiliary['license'].startswith(b'Attribution 4.0 International'), 'Unexpected license text')
    require(collection_order(auxiliary['collection']) == [(r['module'], r['chapter']) for r in records],
            'Module order or chapter identity differs from the source collection')
    sources = {}
    for row in records:
        ident = row['module']
        require(re.fullmatch(r'm[0-9]+', ident) and row['path'] == f'modules/{ident}/index.cnxml',
                'Invalid module identity or source path')
        require(row['review_role'] in ('content_candidate', 'attribution_only'), 'Unknown review role')
        raw = load(row, f'raw/physics/{ident}.cnxml')
        root = ET.fromstring(raw)
        title = root.find(CN + 'title')
        require(root.tag == CN + 'document' and title is not None and
                compact(''.join(title.itertext())) == row['title'], 'Module title differs from source: ' + ident)
        sources[ident] = raw
    attribution = [r for r in records if r['review_role'] == 'attribution_only']
    require(len(attribution) == 1 and sources[attribution[0]['module']] == auxiliary['preface'],
            'The original preface must be preserved and excluded from candidate text')
    return auxiliary, sources


def media_inventory(root):
    """Flag missing descriptions in retained content, without validating images."""
    counts, missing = Counter(), []

    def walk(node):
        if node.tag == CN + 'note' and OMITTED_NOTES.intersection(node.get('class', '').split()):
            return
        if node.tag == CN + 'media':
            counts['media_elements'] += 1
            if not any(c.isalnum() for c in node.get('alt', '')):
                counts['missing_or_placeholder_descriptions'] += 1
                missing.append(dict(id=node.get('id'), alt=node.get('alt', ''),
                                    images=[n.get('src') for n in node.iter(CN + 'image')]))
            return
        if node.tag == CN + 'image':
            counts['images_outside_media'] += 1
        for child in node:
            walk(child)
    for name in ('content', 'glossary'):
        node = root.find(CN + name)
        if node is not None:
            walk(node)
    return dict(counts), missing


def review(spec_path, cache, out, download=False):
    require(not out.exists(), 'Use a fresh review directory')
    spec_raw = spec_path.read_bytes()
    spec = json.loads(spec_raw)
    auxiliary, sources = source_bundle(spec, cache, download)
    titles = {r['module']: r['title'] for r in spec['modules']}
    rows, candidates = [], {}
    totals, baseline_errors, formula_errors, structural_errors = Counter(), Counter(), Counter(), Counter()
    policy_totals, extraction_totals, punctuation_totals, media_totals = Counter(), Counter(), Counter(), Counter()
    for source in spec['modules']:
        ident, raw = source['module'], sources[source['module']]
        row = {k: source[k] for k in ('module', 'chapter', 'title', 'path', 'raw_sha256', 'git_blob_sha1', 'review_role')}
        row['source_url'] = f"{spec['upstream_repository']}/blob/{spec['upstream_commit']}/{source['path']}"
        root = ET.fromstring(raw)
        rendered, failures = [], []
        for index, formula in enumerate(root.iter(MATH + 'math')):
            try:
                rendered.append(math_text(formula, physics_extension))
            except ValueError as error:
                rendered.append(None)
                failures.append(dict(zero_based_index=index, error=str(error)))
                formula_errors[str(error)] += 1
        row.update(raw_formula_count=len(rendered), raw_formula_failures=failures,
                   raw_formula_text_sha256=digest(json.dumps(rendered, ensure_ascii=False).encode()))
        totals['raw_formulas'] += len(rendered)
        try:
            extract(raw, titles, math_extension=prealgebra_extension)
            totals['unmodified_pre_algebra_converter_modules'] += 1
        except ValueError as error:
            baseline_errors[str(error)] += 1
        if source['review_role'] == 'attribution_only':
            row.update(candidate_extracted=False, exclusion='preface_retained_for_attribution_only')
            rows.append(row)
            continue
        totals['content_modules'] += 1
        try:
            body, stats = extract_physics(raw, titles)
        except ValueError as error:
            row.update(candidate_extracted=False, rejection=str(error))
            structural_errors[str(error)] += 1
        else:
            require(not failures, 'A candidate contains unconverted source mathematics')
            transformed, _ = physics_document(raw)
            media_counts, missing = media_inventory(ET.fromstring(transformed))
            words = len(re.findall(r"[A-Za-z]+(?:['-][A-Za-z]+)*", body.decode('utf-8')))
            row.update(candidate_extracted=True, text_file=f'review-text/{ident}.txt', text_sha256=digest(body),
                       text_bytes=len(body), word_like_units=words, **stats, media=media_counts,
                       missing_media_descriptions=missing)
            candidates[ident] = body
            totals['candidate_modules'] += 1
            totals['candidate_bytes'] += len(body)
            totals['candidate_word_like_units'] += words
            policy_totals.update(stats['policy'])
            extraction_totals.update(stats['extraction'])
            punctuation_totals.update(stats['punctuation_repairs'])
            media_totals.update(media_counts)
        rows.append(row)
    # All bytes and collection identities were checked before creating output.
    (out / 'review-text').mkdir(parents=True)
    for ident, body in candidates.items():
        (out / 'review-text' / f'{ident}.txt').write_bytes(body)
    for name, target in [('license', 'LICENSE-source.txt'), ('collection', 'original-collection.xml'),
                         ('preface', 'original-preface.cnxml')]:
        (out / target).write_bytes(auxiliary[name])
    (out / 'ATTRIBUTION.txt').write_bytes((spec['attribution'] + '\n').encode('utf-8'))
    (out / 'review-spec.json').write_bytes(spec_raw)
    report = dict(complete=True, scope='Authenticated source extraction for review; no training admission',
        review_spec_sha256=digest(spec_raw), upstream_repository=spec['upstream_repository'],
        upstream_commit=spec['upstream_commit'], authenticated_modules=len(sources), authenticated_auxiliary_files=3,
        totals=dict(totals), baseline_rejections=dict(baseline_errors), raw_formula_rejections=dict(formula_errors),
        candidate_rejections=dict(structural_errors), policy_totals=dict(policy_totals),
        extraction_totals=dict(extraction_totals), punctuation_repairs=dict(punctuation_totals),
        media_totals=dict(media_totals), attribution=spec['attribution'], reference_urls=spec['reference_urls'],
        records=rows, training_admitted=False, source_files_modified=False, native_calls=0, models_trained=False,
        implementation_sha256={p.as_posix(): digest(p.read_bytes()) for p in
            (Path('scripts/review_physics.py'), Path('scripts/corpus/physics.py'),
             Path('scripts/corpus/openstax.py'), Path('scripts/corpus/prealgebra_math.py'))},
        limits=['Source identity and conversion checks do not establish factual correctness or completeness.',
                'Images are not loaded. Present descriptions can still omit essential diagram information.',
                'Teacher/activity notes and vocabulary grids are omitted by an explicit source-class policy.',
                'Math is text notation; spacing, font shape and size are not a facsimile of the source.',
                'Generic cross-references remain review flags; context-dependent passages need inspection.',
                'Word-like counts include repeated words and notation labels; they are not unique facts or model tokens.',
                'There is no passage admission, held-out split, overlap filtering, curriculum or model training.'])
    write(out / 'result.json', report)
    print(json.dumps(dict(authenticated_modules=len(sources), totals=dict(totals),
                         candidate_rejections=dict(structural_errors), media=dict(media_totals)), indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', type=Path, default=Path('data/physics-source-review-v1.json'))
    parser.add_argument('--cache', type=Path, default=Path('runs/science-source-review'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--download', action='store_true', help='Fetch missing pinned cache files')
    args = parser.parse_args()
    review(args.spec, args.cache, args.out, args.download)
