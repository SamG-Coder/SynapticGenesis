"""Audit selected modern readers, publication agreement and exclusion boundaries."""
import argparse
import copy
import hashlib
from pathlib import Path
import re
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from corpus.storybooks import parse, verify_publication
from native_experiment import read, sha, write
from prepare_binding import examples
from prepare_early_readers import prepare, substantial_paragraphs


def shingles(document, width=20):
    words = document.decode('utf-8').lower().split()
    return {tuple(words[i:i+width]) for i in range(len(words)-width+1)}


def audit(out, prepared, cache):
    out.mkdir(parents=True, exist_ok=False)
    spec_path = Path('data/sources-early-readers-v1.json')
    spec = read(spec_path)
    protected = Path('data/prepared/stories-v1')
    prepare(spec_path, cache, out/'reproduced', protected)
    files = sorted(p for p in prepared.iterdir() if p.is_file())
    assert len(files) == 28
    for path in files:
        assert path.read_bytes() == (out/'reproduced'/path.name).read_bytes(), path.name
    manifest = read(prepared/'manifest.json')
    assert sha(spec_path) == manifest['source_spec_sha256']
    assert manifest['upstream_commit'] == spec['upstream_commit']
    expected_ids = {'train': {'0008', '0087', '0001', '0027', '0089', '0095', '0110', '0315', '0243', '0294'},
                    'validation': {'0156', '0296', '0324'}, 'test': {'0327', '0004', '0291'}}
    groups = {split: [s for s in spec['sources'] if s['split'] == split] for split in expected_ids}
    documents = {}
    for split, sources in groups.items():
        assert {s['id'] for s in sources} == expected_ids[split]
        selected = []
        for source in sources:
            identifier = source['id']
            raw = (cache/source['source_file']).read_bytes()
            # Independently extract known pinned pages without the production
            # parser, then compare every byte, including internal paragraphs.
            chunks = raw.decode('utf-8').split('\n##\n')
            assert chunks[0].strip() == '# '+source['title']
            expected = ('\n\n'.join(chunk.strip() for chunk in chunks[1:-1])+'\n').encode('utf-8')
            document = (prepared/f'{identifier}.txt').read_bytes()
            assert document == expected and sha(prepared/f'{identifier}.txt') == source['body_sha256']
            assert b'##' not in document and b'* License:' not in document and b'\x1e' not in document
            assert source['attribution']['Text'] in (prepared/'ATTRIBUTION.md').read_text(encoding='utf-8')
            documents[identifier] = document
            selected.append(document)
        payload = b'\x1e'.join(selected)
        assert payload == (prepared/f'{split}.dat').read_bytes()
        assert sha(prepared/f'{split}.dat') == manifest['outputs'][split]['sha256']
    cumulative = []
    for stage in manifest['stages']:
        selected = [documents[s['id']] for s in groups['train'] if s['stage'] == stage['id']]
        assert selected and b'\x1e'.join(selected) == (prepared/stage['file']).read_bytes()
        cumulative.extend(selected)
        assert b'\x1e'.join(cumulative) == (prepared/stage['cumulative_file']).read_bytes()
    old = read(protected/'manifest.json')
    heldout = [documents[s['id']] for split in ('validation', 'test') for s in groups[split]]
    for source in old['sources']:
        if source['split'] != 'train':
            path = protected/f'{source["id"]}.txt'
            assert sha(path) == source['clean_sha256'] == manifest['protected_sources'][path.name]
            heldout.append(path.read_bytes())
    reserved_paragraphs = set().union(*(substantial_paragraphs(d) for d in heldout))
    reserved_shingles = set().union(*(shingles(d) for d in heldout))
    for source in groups['train']:
        doc = documents[source['id']]
        assert not substantial_paragraphs(doc) & reserved_paragraphs
        assert not shingles(doc) & reserved_shingles
    _, _, probes, _ = examples(read('data/lessons-binding-v2.json'))
    contexts = {re.sub(r'\s+', ' ', r['context']).strip()
                for split in ('development', 'test') for r in probes[split]}
    train = re.sub(r'\s+', ' ', (prepared/'train.dat').read_text(encoding='utf-8'))
    assert not any(context in train for context in contexts)

    rejected = []
    def rejection(name, operation):
        try:
            operation()
        except ValueError:
            rejected.append(name)
        else:
            raise AssertionError('Negative control accepted: '+name)

    selected = copy.deepcopy(spec['sources'][0])
    raw = (cache/selected['source_file']).read_bytes()
    rejection('changed-pinned-text', lambda: parse(raw+b'changed', selected))
    for label, candidate, changes in [
        ('wrong-title', raw.replace(b'# What are you doing?', b'# Wrong story', 1), {}),
        ('noncommercial-edition', raw.replace(b'[CC-BY]', b'[CC-BY-NC]'), {'License': '[CC-BY-NC]'}),
        ('wrong-language', raw.replace(b'Language: en', b'Language: fr'), {'Language': 'fr'}),
        ('missing-page', raw.replace(b'##\nI am singing.\n\n', b'', 1), {}),
        ('duplicated-credit', raw.replace(b'* Language: en', b'* Text: Someone\n* Language: en'), {})]:
        edited = copy.deepcopy(selected)
        edited['raw_sha256'] = hashlib.sha256(candidate).hexdigest()
        edited['attribution'].update(changes)
        rejection(label, lambda candidate=candidate, edited=edited: parse(candidate, edited))
    catalog = (cache/(selected['id']+'.html')).read_bytes()
    _, pages = parse(raw, selected)
    for label, candidate in [
        ('publisher-license-change', catalog.replace(b'/licenses/by/3.0/', b'/licenses/by-nc/3.0/')),
        ('publisher-language-body-change', catalog.replace(b'I am singing.', b'I am sleeping.'))]:
        edited = {**selected, 'catalog_sha256': hashlib.sha256(candidate).hexdigest()}
        rejection(label, lambda candidate=candidate, edited=edited: verify_publication(candidate, edited, pages))
    rejection('existing-output', lambda: prepare(spec_path, cache, prepared, protected))
    # Substitute one already-validated document at the assembly boundary. This
    # exercises the actual corpus rejection path rather than just its set helper.
    import prepare_early_readers as assembly
    actual_load = assembly.load
    injected = next(doc for doc in heldout if substantial_paragraphs(doc))
    def contaminated(source, commit, cache_path):
        document, record = actual_load(source, commit, cache_path)
        if source['id'] == selected['id']:
            document += b'\n'+injected
        return document, record
    with patch.object(assembly, 'load', contaminated):
        rejection('heldout-document-in-training', lambda: prepare(spec_path, cache, out/'contaminated', protected))
    assert not (out/'contaminated').exists()
    evidence = dict(passed=True, reproduced_files=len(files), source_count=len(documents),
                    outputs=manifest['outputs'], training_words=sum(len(documents[s['id']].decode().split()) for s in groups['train']),
                    training_word_counts_by_stage={str(i): sum(len(documents[s['id']].decode().split())
                        for s in groups['train'] if s['stage'] == i) for i in (1, 2, 3)},
                    protected_existing_evaluation_books=len(manifest['protected_sources']),
                    heldout_paragraph_overlap=0, heldout_20_word_overlap=0, heldout_binding_contexts_checked=len(contexts),
                    rejected_negative_controls=rejected, training_text_matches_pinned_source_pages=True,
                    published_english_pages_verified=True, publisher_typography_normalized_for_comparison_only=True,
                    attribution_preserved=True, cumulative_training_prefixes_verified=True,
                    model_training_performed=False, reserved_tests_scored=False,
                    source_spec_sha256=sha(spec_path), source_manifest_sha256=sha(prepared/'manifest.json'),
                    code_sha256={p: sha(p) for p in ('scripts/corpus/storybooks.py', 'scripts/prepare_early_readers.py',
                                                   'tests/early_reader_corpus.py')},
                    limits='A small preparation result, not evidence of improved model quality. Exact paragraph and '
                           '20-whitespace-word overlap checks do not detect paraphrases or shorter shared phrases.')
    write(out/'verification.json', evidence)
    print(evidence)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--prepared', type=Path, default=Path('data/prepared/early-readers-v1'))
    p.add_argument('--raw', type=Path, default=Path('data/raw/sbc-be28dab'))
    a = p.parse_args()
    audit(a.out, a.prepared, a.raw)
