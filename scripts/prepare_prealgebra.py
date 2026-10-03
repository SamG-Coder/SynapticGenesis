"""Prepare only the individually reviewed, pinned arithmetic worked examples."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

from corpus.openstax import CN
from corpus.paired_selection import question, select_pairs
from corpus.selection import require_training_spec
from native_experiment import read, sha
from review_prealgebra_integrity import worked_candidate


def require(condition, message):
    if not condition:
        raise ValueError(message)


def reviewed_candidates(candidate_path, review_path, cache):
    review, bundle = read(review_path), read(candidate_path)
    require(review['complete'] and review['candidates_sha256'] == sha(candidate_path), 'Candidate review mismatch')
    require(bundle['status'] == 'review_only_not_admitted', 'Unexpected input bundle')
    candidates, decisions = bundle['candidates'], review['records']
    require(len(candidates) == len(decisions) == review['candidate_count'], 'Incomplete passage review')
    require([r['index'] for r in decisions] == list(range(len(candidates))), 'Review order changed')
    integrity_path = Path(review['source_integrity_report'])
    require(sha(integrity_path) == review['source_integrity_report_sha256'], 'Source-integrity report changed')
    integrity = read(integrity_path)
    require(integrity['complete'] and integrity['candidates_sha256'] == sha(candidate_path)
            and integrity['totals']['text_worked_candidates'] == len(candidates), 'Incomplete candidate inventory')
    audit_path = Path(review['notation_audit'])
    require(sha(audit_path) == review['notation_audit_sha256'], 'Notation/source audit changed')
    audit = read(audit_path)
    require(audit['passed'] and audit['upstream_commit'] == review['upstream_commit']
            and audit['upstream_repository'] == review['upstream_repository'], 'Source edition changed')
    require(integrity['notation_audit_sha256'] == sha(audit_path), 'Audits describe different sources')
    sources = {}
    for row in audit['records']:
        path = cache / 'raw/prealgebra' / f"{row['module']}.cnxml"
        require(sha(path) == row['raw_sha256'], 'Upstream module changed: ' + row['module'])
        sources[row['module']] = (row, ET.fromstring(path.read_bytes()))
    seen, selected = set(), []
    for row, decision in zip(candidates, decisions):
        key = (row['module'], row['exercise'])
        require(key not in seen, 'Duplicate reviewed source identity')
        seen.add(key)
        require(all(row[k] == decision[k] for k in ('module', 'exercise', 'text_sha256')),
                'Review identity/text changed')
        require(decision['decision'] in ('retain', 'exclude') and decision['reason'].strip(), 'Unreviewed decision')
        source, root = sources[row['module']]
        require(row['source_sha256'] == source['raw_sha256'] and row['chapter'] == source['chapter']
                and row['module_title'] == source['title'], 'Candidate provenance changed')
        matches = [e for e in root.iter(CN + 'exercise') if e.get('id') == row['exercise']]
        require(len(matches) == 1, 'Missing/duplicate source exercise')
        result, reason = worked_candidate(matches[0], row['module_title'])
        require(reason is None, 'Source pair is no longer independently extractable')
        body, _ = result
        require(body == row['text'].encode('utf-8')
                and hashlib.sha256(body).hexdigest() == row['text_sha256'], 'Reviewed extraction changed')
        question(row['text'])
        if decision['decision'] == 'retain':
            selected.append(dict(row, id=row['module'] + '-' + row['exercise']))
    return review, selected, audit


def protected_inputs(records):
    result = []
    for row in records:
        path = Path(row['path'])
        require(sha(path) == row['sha256'], 'Held-out source changed: ' + str(path))
        result.append((row['path'], path.read_bytes(), row['kind']))
    return result


def prepare(spec_path, candidate_path, cache, out):
    require(not out.exists(), 'Use a fresh prepared directory')
    spec = require_training_spec(spec_path)
    review_path = Path(spec['passage_review'])
    require(sha(review_path) == spec['passage_review_sha256'], 'Passage decisions changed')
    review, candidates, audit = reviewed_candidates(candidate_path, review_path, cache)
    require(spec['upstream_commit'] == review['upstream_commit']
            and spec['upstream_repository'] == review['upstream_repository'], 'Specification source mismatch')
    for row in candidates:
        policy = spec['chapter_policy'][row['chapter']]
        row.update(split=policy['split'], stage=policy['stage'])
    selected, omitted = select_pairs(candidates, protected_inputs(spec['protected_files']))
    require(omitted == spec['overlap_omissions'], 'Whole-lesson overlap decisions changed')
    expected = {r['id']: r for r in spec['sources']}
    require(len(expected) == len(spec['sources']) == len(selected), 'Selected lesson count changed')
    for row in selected:
        require(row['id'] in expected and re.fullmatch(r'm[0-9]+-fs-id[0-9]+', row['id']), 'Unexpected lesson identity')
        require(all(row[k] == value for k, value in expected[row['id']].items()), 'Selected lesson changed')
    auxiliaries = {}
    for row in spec['auxiliary_files']:
        raw = (cache / row['cache_path']).read_bytes()
        require(hashlib.sha256(raw).hexdigest() == row['sha256'], 'Attribution source changed')
        auxiliaries[row['name']] = raw
    require(auxiliaries['license'].startswith(b'Attribution 4.0 International'), 'Pinned license mismatch')
    require(hashlib.sha256(auxiliaries['license']).hexdigest() == audit['license_sha256']
            and hashlib.sha256(auxiliaries['collection']).hexdigest() == audit['collection_sha256'],
            'License/collection differs from audited edition')
    ns = {'c': 'http://cnx.rice.edu/collxml', 'm': 'http://cnx.rice.edu/mdml'}
    license_url = ET.fromstring(auxiliaries['collection']).find('c:metadata/m:license', ns).get('url')
    require(license_url.rstrip('/').endswith('creativecommons.org/licenses/by/4.0'), 'Collection license mismatch')
    bodies = {r['id']: r['text'].encode('utf-8') for r in selected}
    records = [dict(r, clean_bytes=len(bodies[r['id']]), clean_sha256=hashlib.sha256(bodies[r['id']]).hexdigest())
               for r in spec['sources']]
    out.mkdir(parents=True)
    (out / 'source-spec.json').write_bytes(spec_path.read_bytes())
    (out / 'passage-review.json').write_bytes(review_path.read_bytes())
    (out / 'LICENSE-source.txt').write_bytes(auxiliaries['license'])
    (out / 'original-preface.cnxml').write_bytes(auxiliaries['preface'])
    (out / 'original-collection.xml').write_bytes(auxiliaries['collection'])
    (out / 'ATTRIBUTION.txt').write_bytes((spec['attribution'] + '\n').encode('utf-8'))
    for ident, raw in bodies.items():
        (out / f'{ident}.txt').write_bytes(raw)
    outputs = {}
    for split in ('train', 'validation', 'test'):
        rows = [r for r in records if r['split'] == split]
        raw = b'\x1e'.join(bodies[r['id']] for r in rows)
        require(bool(raw), 'Every split must contain a complete lesson')
        (out / f'{split}.dat').write_bytes(raw)
        outputs[split] = dict(bytes=len(raw), documents=len(rows), sha256=hashlib.sha256(raw).hexdigest(),
            word_like_units=len(re.findall(r"[A-Za-z]+(?:['-][A-Za-z]+)*", raw.decode('utf-8'))))
    stages = []
    for stage in spec['stages']:
        raw = b'\x1e'.join(bodies[r['id']] for r in records if r['split'] == 'train' and r['stage'] == stage['id'])
        require(bool(raw), 'Empty training stage')
        name = f"stage-{stage['id']}.dat"
        (out / name).write_bytes(raw)
        stages.append(dict(stage, file=name, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()))
    manifest = dict(version=spec['version'], provider='reviewed-openstax-worked-examples',
        source_spec_sha256=sha(spec_path), passage_review_sha256=sha(review_path),
        tokenizer='UTF-8 bytes, fixed IDs 0..255', document_separator='0x1e, excluded from sampled windows',
        source_repository=spec['upstream_repository'], source_commit=spec['upstream_commit'],
        license='CC BY 4.0, retained from the pinned 2024 edition', attribution=spec['attribution'],
        review_counts=dict(Counter(r['decision'] for r in review['records'])),
        protected_files=spec['protected_files'], overlap_omissions=omitted,
        outputs=outputs, stages=stages, sources=records,
        source_text_modified=False, models_trained=False, curriculum_created=False,
        limitations=spec['limitations'])
    (out / 'manifest.json').write_bytes((json.dumps(manifest, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sources', type=Path, default=Path('data/sources-prealgebra-v1.json'))
    parser.add_argument('--candidates', type=Path, required=True)
    parser.add_argument('--cache', type=Path, default=Path('runs/science-source-review'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.sources, args.candidates, args.cache, args.out)
    print(json.dumps(result['outputs'], indent=2))
