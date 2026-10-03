"""Prepare a small selected graded-text edition; this command trains no model."""
import argparse
import json
from pathlib import Path
import re

from corpus.storybooks import load, normalized
from corpus.selection import require_training_spec
from native_experiment import read, sha


def substantial_paragraphs(document):
    return {value for p in re.split(r'\n\s*\n', document.decode('utf-8'))
            if len(value := normalized(p)) >= 120}


def prepare(spec_path, cache, out, protected):
    spec = require_training_spec(spec_path)
    sources, stages = spec['sources'], spec['stages']
    identifiers, stage_ids = [s['id'] for s in sources], [s['id'] for s in stages]
    if not sources or len(identifiers) != len(set(identifiers)) or stage_ids != [1, 2, 3]:
        raise ValueError('Invalid selected source or stage identities')
    for source in sources:
        if (source['split'] not in ('train', 'validation', 'test') or source['stage'] not in stage_ids
                or not re.fullmatch(r'[0-9]{4}', source['id']) or not source['reason']):
            raise ValueError('Invalid selected source role')
    if out.exists():
        raise ValueError('Use a fresh output directory')
    # Read and authenticate the existing reserved editions; do not run a model
    # on them or place them in the new training data.
    old = read(protected/'manifest.json')
    reserved, protected_hashes = set(), {}
    for source in old['sources']:
        if source['split'] != 'train':
            path = protected/f'{source["id"]}.txt'
            if sha(path) != source['clean_sha256']:
                raise ValueError('Protected evaluation source changed')
            protected_hashes[path.name] = sha(path)
            reserved.update(substantial_paragraphs(path.read_bytes()))
    documents, records = {}, []
    for source in sources:
        document, record = load(source, spec['upstream_commit'], cache)
        documents[source['id']] = document
        records.append(record)
        if source['split'] != 'train':
            reserved.update(substantial_paragraphs(document))
    # Reject overlap instead of removing pages from these very short stories.
    training = [s for s in sources if s['split'] == 'train']
    for source in training:
        if substantial_paragraphs(documents[source['id']]) & reserved:
            raise ValueError('Selected training text overlaps a protected paragraph')
    if len(set(documents.values())) != len(documents):
        raise ValueError('Duplicate selected stories')
    outputs, stage_outputs = {}, []
    out.mkdir(parents=True)
    for identifier, document in documents.items():
        (out/f'{identifier}.txt').write_bytes(document)
    for split in ('train', 'validation', 'test'):
        selected = [documents[s['id']] for s in sources if s['split'] == split]
        (out/f'{split}.dat').write_bytes(b'\x1e'.join(selected))
        outputs[split] = dict(documents=len(selected), bytes=(out/f'{split}.dat').stat().st_size,
                              sha256=sha(out/f'{split}.dat'))
    cumulative = []
    for stage in stages:
        selected = [documents[s['id']] for s in training if s['stage'] == stage['id']]
        if not selected:
            raise ValueError('Empty training stage')
        cumulative.extend(selected)
        file, through = f'stage-{stage["id"]}.dat', f'through-stage-{stage["id"]}.dat'
        (out/file).write_bytes(b'\x1e'.join(selected))
        (out/through).write_bytes(b'\x1e'.join(cumulative))
        stage_outputs.append(dict(**stage, file=file, documents=len(selected), sha256=sha(out/file),
                                  cumulative_file=through, cumulative_sha256=sha(out/through)))
    (out/'source-spec.json').write_bytes(spec_path.read_bytes())
    manifest = dict(version=spec['version'], source_spec_sha256=sha(spec_path),
                    upstream_commit=spec['upstream_commit'], sources=records, outputs=outputs, stages=stage_outputs,
                    protected_manifest_sha256=sha(protected/'manifest.json'), protected_sources=protected_hashes,
                    model_training_performed=False, reserved_tests_scored=False,
                    transform='Remove Markdown title, page markers and attribution from model text; '
                              'join unedited page bodies with blank lines. Preserve full attribution separately. '
                              'Images and audio are not downloaded.',
                    limitations=spec['limitations'])
    (out/'manifest.json').write_bytes((json.dumps(manifest, ensure_ascii=False, indent=2)+'\n').encode('utf-8'))
    credits = ['# Selected early-reader source attribution', '', spec['policy'], '',
               'Changes: '+manifest['transform'], '',
               f'Upstream revision: `{spec["upstream_commit"]}`. Reading levels are publisher labels.', '']
    for r in records:
        credits += [f'## {r["title"]}', '', f'[Source]({r["catalog_url"]}) · [License]({r["license_url"]})', '']
        credits += [f'- {key}: {value}' for key, value in r['attribution'].items()]
        credits += ['', f'Role: {r["split"]}; level {r["reading_level"]}.', '']
    (out/'ATTRIBUTION.md').write_bytes(('\n'.join(credits)).encode('utf-8'))
    print(json.dumps(outputs, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--sources', type=Path, default=Path('data/sources-early-readers-v1.json'))
    p.add_argument('--raw', type=Path, default=Path('data/raw/sbc-be28dab'))
    p.add_argument('--out', type=Path, default=Path('data/prepared/early-readers-v1'))
    p.add_argument('--protected', type=Path, default=Path('data/prepared/stories-v1'))
    a = p.parse_args()
    prepare(a.sources, a.raw, a.out, a.protected)
