"""Prepare a selected reading expansion without training or scoring any model."""
import argparse
import json
from pathlib import Path
import re

from corpus.african_storybook import index_entries, load
from corpus.overlap import reject_overlap
from native_experiment import sha
from prepare_corpus import fetch


DEFAULT_PROTECTED = {'early-readers': Path('data/prepared/early-readers-v1'),
                     'stories': Path('data/prepared/stories-v1')}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def protected_documents(spec, roots):
    if set(roots) != set(spec['protected_editions']):
        raise ValueError('Missing protected edition')
    documents, records, previous_story_ids = {}, {}, set()
    for name, root in roots.items():
        expected = spec['protected_editions'][name]
        if sha(root/'manifest.json') != expected['manifest_sha256']:
            raise ValueError('Protected edition manifest changed')
        manifest = read(root/'manifest.json')
        if manifest['version'] != expected['version']:
            raise ValueError('Protected edition version changed')
        for source in manifest['sources']:
            filename = str(source['id'])+'.txt'
            if not re.fullmatch(r'[0-9]+\.txt', filename):
                raise ValueError('Invalid protected source identity')
            path = root/filename
            digest = source.get('body_sha256', source.get('clean_sha256'))
            if sha(path) != digest:
                raise ValueError('Protected source bytes changed')
            identity = name+'/'+filename
            if identity in documents:
                raise ValueError('Duplicate protected identity')
            documents[identity] = path.read_bytes()
            records[identity] = dict(sha256=digest, split=source['split'], title=source['title'])
            if name == 'early-readers':
                previous_story_ids.add(source['id'])
    return documents, records, previous_story_ids


def prepare(spec_path, cache, out, protected_roots=None):
    if out.exists():
        raise ValueError('Use a fresh output directory')
    spec = read(spec_path)
    sources, stages = spec['sources'], spec['stages']
    ids = [source['id'] for source in sources]
    stage_ids = [stage['id'] for stage in stages]
    if not sources or len(ids) != len(set(ids)) or stage_ids != [1, 2, 3]:
        raise ValueError('Invalid source or stage identities')
    if set(source['split'] for source in sources) != {'train', 'validation', 'test'}:
        raise ValueError('Require all three declared source roles')
    for source in sources:
        if (not re.fullmatch(r'[0-9]{4}', source['id']) or source['stage'] not in stage_ids
                or not source['reason'] or not source['review_note']):
            raise ValueError('Invalid selected source role or review')
    if any(not any(s['split'] == 'train' and s['stage'] == i for s in sources) for i in stage_ids):
        raise ValueError('Empty training stage')
    old, protected, previous_ids = protected_documents(spec, protected_roots or DEFAULT_PROTECTED)
    if set(ids) & previous_ids:
        raise ValueError('Previously selected story identity cannot enter the expansion')
    commit = spec['upstream_commit']
    if not re.fullmatch(r'[0-9a-f]{40}', commit):
        raise ValueError('Require a pinned upstream commit')
    cache.mkdir(parents=True, exist_ok=True)
    index_url = f'https://raw.githubusercontent.com/global-asp/asp-source/{commit}/en/README.md'
    catalog = fetch(index_url, cache/'README.md')
    entries = index_entries(catalog, spec['upstream_index_sha256'])
    documents, records = {}, []
    for source in sources:
        document, record = load(source, commit, entries, cache)
        documents[source['id']] = document
        records.append(record)
    # All old roles are excluded from this expansion, and whole new works are
    # assigned one role. No evaluation text is used as a model learning target.
    reject_overlap(documents, old)
    outputs, stage_outputs = {}, []
    out.mkdir(parents=True)
    for identifier, document in documents.items():
        (out/f'{identifier}.txt').write_bytes(document)
    for split in ('train', 'validation', 'test'):
        selected = [documents[s['id']] for s in sources if s['split'] == split]
        path = out/f'{split}.dat'
        path.write_bytes(b'\x1e'.join(selected))
        outputs[split] = dict(documents=len(selected), bytes=path.stat().st_size, sha256=sha(path),
                              words=sum(len(d.decode('utf-8').split()) for d in selected),
                              within_document_next_byte_pairs=sum(len(d)-1 for d in selected))
    cumulative = []
    for stage in stages:
        selected = [documents[s['id']] for s in sources if s['split'] == 'train' and s['stage'] == stage['id']]
        cumulative.extend(selected)
        name, through = f'stage-{stage["id"]}.dat', f'through-stage-{stage["id"]}.dat'
        (out/name).write_bytes(b'\x1e'.join(selected))
        (out/through).write_bytes(b'\x1e'.join(cumulative))
        stage_outputs.append(dict(**stage, file=name, documents=len(selected), sha256=sha(out/name),
                                  cumulative_file=through, cumulative_sha256=sha(out/through)))
    (out/'source-spec.json').write_bytes(spec_path.read_bytes())
    manifest = dict(version=spec['version'], source_spec_sha256=sha(spec_path), upstream_commit=commit,
                    upstream_index_sha256=spec['upstream_index_sha256'], source_index_url=index_url,
                    sources=records, outputs=outputs, stages=stage_outputs, protected_sources=protected,
                    protected_editions=spec['protected_editions'], model_training_performed=False,
                    reserved_tests_scored=False, publisher_pages_verified=False,
                    transform='Remove Markdown title, page delimiters, attribution and explicitly pinned '
                              'editorial footers after attribution from model text; '
                              'join otherwise unedited page bodies with blank lines and one final newline. '
                              'Preserve internal line breaks, source spelling and story notes. '
                              'Preserve attribution separately; no images, audio or model-generated text.',
                    limitations=spec['limitations'])
    (out/'manifest.json').write_bytes((json.dumps(manifest, ensure_ascii=False, indent=2)+'\n').encode('utf-8'))
    credits = ['# Selected reading expansion: source attribution', '', spec['policy'], '',
               'Changes: '+manifest['transform'], '',
               f'Upstream revision: `{commit}`. [Pinned source index]({index_url}).', '',
               'Source/index agreement is checked; current publisher pages are not independently verified.', '']
    for r in records:
        entry = r['index_entry']
        credits += [f'## {r["title"]}', '', f'[Selected source]({r["source_url"]}) | '
                    f'[Original reference]({entry["publisher_url"]}) | [License]({entry["license_url"]})', '']
        credits += [f'- {key}: {value}' for key, value in r['attribution'].items()]
        if 'source_footer' in r:
            credits += ['', 'Editorial footer (excluded from model text): '+r['source_footer']]
        credits += ['', f'Role: {r["split"]}; content group {r["stage"]}.', '', r['review_note'], '']
    (out/'ATTRIBUTION.md').write_bytes(('\n'.join(credits)).encode('utf-8'))
    print(json.dumps(outputs, indent=2))
    return manifest


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--sources', type=Path, default=Path('data/sources-reading-breadth-v1.json'))
    p.add_argument('--raw', type=Path, default=Path('data/raw/asp-b5c3d5b'))
    p.add_argument('--out', type=Path, default=Path('data/prepared/reading-breadth-v1'))
    p.add_argument('--early-readers', type=Path, default=DEFAULT_PROTECTED['early-readers'])
    p.add_argument('--stories', type=Path, default=DEFAULT_PROTECTED['stories'])
    a = p.parse_args()
    prepare(a.sources, a.raw, a.out, {'early-readers': a.early_readers, 'stories': a.stories})
