"""Prepare the admitted, pinned Astronomy 2e selection; no model computation."""
import argparse
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

from corpus.openstax import extract
from corpus.selection import require_training_spec
from native_experiment import sha as file_sha, verified_book_manifest
from prepare_corpus import fetch, sha


def load_raw(spec, record, cache):
    relative = record['path']
    if not re.fullmatch(r'(LICENSE|collections/[a-z0-9-]+\.collection\.xml|modules/m[0-9]+/index\.cnxml)', relative):
        raise ValueError('Unexpected upstream text path')
    url = f"https://raw.githubusercontent.com/openstax/osbooks-astronomy/{spec['upstream_commit']}/{relative}"
    destination = cache / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    raw = fetch(url, destination)
    if sha(raw) != record['raw_sha256']:
        raise ValueError(f'Reviewed upstream bytes changed: {relative}')
    return raw


def paragraphs(raw):
    return [re.sub(r'\s+', ' ', p).strip() for p in raw.decode('utf-8').split('\n\n')]


def corrected_body(body, record):
    text = body.decode('utf-8')
    for correction in record.get('corrections', []):
        if not correction['before'] or text.count(correction['before']) != 1:
            raise ValueError('Reviewed correction must match exactly once')
        text = text.replace(correction['before'], correction['after'], 1)
    payload = text.encode('utf-8')
    if sha(payload) != record.get('corrected_sha256', record['extracted_sha256']):
        raise ValueError('Corrected text differs from reviewed selection')
    return payload


def prepare(source, cache, out):
    spec = require_training_spec(source)
    if (spec['upstream_repository'] != 'https://github.com/openstax/osbooks-astronomy'
            or not re.fullmatch('[0-9a-f]{40}', spec['upstream_commit'])):
        raise ValueError('Unexpected repository or unpinned commit')
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use a fresh output directory for a new preparation')
    titles = spec['module_titles']
    ids = [r['id'] for r in spec['sources']]
    assert len(ids) == len(set(ids)) and all(re.fullmatch('m[0-9]+', i) for i in ids)
    assert all(r['split'] in ('train', 'validation', 'test') and r['stage'] in (1, 2, 3, 4) for r in spec['sources'])
    for chapter in {r['chapter_number'] for r in spec['sources']}:
        assert len({r['split'] for r in spec['sources'] if r['chapter_number'] == chapter}) == 1
    auxiliaries = {r['name']: load_raw(spec, r, cache) for r in spec['auxiliary_files']}
    collection = ET.fromstring(auxiliaries['collection'])
    ns = {'c': 'http://cnx.rice.edu/collxml', 'm': 'http://cnx.rice.edu/mdml'}
    license_url = collection.find('c:metadata/m:license', ns).get('url')
    assert license_url.rstrip('/').endswith('creativecommons.org/licenses/by/4.0')
    assert auxiliaries['license'].startswith(b'Attribution 4.0 International')
    chapters = collection.findall('./c:content/c:subcollection', ns)
    for record in spec['sources']:
        assert record['id'] in {n.get('document') for n in chapters[record['chapter_number'] - 1].findall('.//c:module', ns)}
    protected = Path(spec['protected_edition']['prepared'])
    previous = verified_book_manifest(protected, Path(spec['protected_edition']['sources']))
    assert file_sha(protected / 'manifest.json') == spec['protected_edition']['manifest_sha256']
    seen = set()
    for record in previous['sources']:
        if record['split'] != 'train':
            seen.update(p for p in paragraphs((protected / f"{record['id']}.txt").read_bytes()) if len(p) >= 120)
    bodies, records = {}, []
    ordered = sorted(spec['sources'], key=lambda r: {'test': 0, 'validation': 1, 'train': 2}[r['split']])
    for record in ordered:
        raw = load_raw(spec, record, cache)
        root = ET.fromstring(raw)
        assert ''.join(root.find('{http://cnx.rice.edu/cnxml}title').itertext()) == record['title']
        body, stats = extract(raw, titles)
        if sha(body) != record['extracted_sha256']:
            raise ValueError(f'Extractor output differs from reviewed text: {record["id"]}')
        body = corrected_body(body, record)
        kept, removed = [], 0
        for paragraph in body.decode('utf-8').split('\n\n'):
            normalized = re.sub(r'\s+', ' ', paragraph).strip()
            if len(normalized) >= 120 and normalized in seen:
                removed += 1
                continue
            if len(normalized) >= 120:
                seen.add(normalized)
            kept.append(paragraph)
        payload = ('\n\n'.join(kept).strip() + '\n').encode('utf-8')
        bodies[record['id']] = payload
        records.append(dict(**record, clean_bytes=len(payload), clean_sha256=sha(payload),
                            removed_duplicate_paragraphs=removed, extraction=stats))
    out.mkdir(parents=True, exist_ok=True)
    (out / 'source-spec.json').write_bytes(source.read_bytes())
    (out / 'LICENSE-source.txt').write_bytes(auxiliaries['license'])
    (out / 'original-preface.cnxml').write_bytes(auxiliaries['preface'])
    (out / 'ATTRIBUTION.txt').write_text(spec['attribution'] + '\n', encoding='utf-8')
    for ident, payload in bodies.items():
        (out / f'{ident}.txt').write_bytes(payload)
    outputs = {}
    for split in ('train', 'validation', 'test'):
        selected = [r for r in spec['sources'] if r['split'] == split]
        payload = b'\x1e'.join(bodies[r['id']] for r in selected)
        (out / f'{split}.dat').write_bytes(payload)
        outputs[split] = dict(bytes=len(payload), documents=len(selected), sha256=sha(payload),
                             word_like_units=len(re.findall(r"[A-Za-z]+(?:['-][A-Za-z]+)*", payload.decode('utf-8'))))
    stages = []
    for stage in spec['stages']:
        payload = b'\x1e'.join(bodies[r['id']] for r in spec['sources'] if r['split'] == 'train' and r['stage'] == stage['id'])
        name = f"stage-{stage['id']}.dat"
        (out / name).write_bytes(payload)
        stages.append(dict(**stage, file=name, bytes=len(payload), sha256=sha(payload)))
    manifest = dict(version=spec['version'], provider='openstax-cnxml', source_spec_sha256=file_sha(source),
                    tokenizer='UTF-8 bytes, fixed IDs 0..255', document_separator='0x1e, excluded from sampled windows',
                    source_repository=spec['upstream_repository'], source_commit=spec['upstream_commit'],
                    license='CC BY 4.0, retained from the pinned 2024 edition', attribution=spec['attribution'],
                    protected_edition=spec['protected_edition'], outputs=outputs, stages=stages,
                    sources=records, models_trained=False, curriculum_created=False,
                    limitation=spec['limitations'])
    (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(outputs, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sources', type=Path, default=Path('data/sources-astronomy-v1.json'))
    parser.add_argument('--raw', type=Path, default=Path('data/raw/openstax-astronomy-2024'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.sources, args.raw, args.out)
