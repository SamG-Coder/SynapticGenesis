"""Audit immutable text selection, attribution, whole-work splits and exclusions."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import prepare_reading_breadth as assembly
from corpus.african_storybook import index_entries, load, verify_index_entry
from corpus.storybooks import parse
from native_experiment import sha
from prepare_binding import examples


def independent_signatures(document):
    # Character classification is independent of the production regex tokenizer.
    def tokens(text):
        return ''.join(c if c.isalnum() or c == '_' else ' ' for c in text.casefold()).split()
    text = document.decode('utf-8')
    words = tokens(text)
    paragraphs = {' '.join(tokens(p)) for p in re.split(r'\n\s*\n', text)}
    return (tuple(words), {tuple(words[i:i+20]) for i in range(len(words)-19)},
            {p for p in paragraphs if len(p) >= 120})


def audit(out, prepared, cache):
    out.mkdir(parents=True, exist_ok=False)
    spec_path = Path('data/sources-reading-breadth-v1.json')
    spec = assembly.read(spec_path)
    manifest = assembly.read(prepared/'manifest.json')
    assert manifest['source_spec_sha256'] == sha(spec_path)
    assert (prepared/'source-spec.json').read_bytes() == spec_path.read_bytes()
    assembly.prepare(spec_path, cache, out/'reproduced')
    files = sorted(path.name for path in prepared.iterdir() if path.is_file())
    assert len(files) == 54 and files == sorted(p.name for p in (out/'reproduced').iterdir())
    for filename in files:
        assert (prepared/filename).read_bytes() == (out/'reproduced'/filename).read_bytes(), filename

    documents = {}
    credits = (prepared/'ATTRIBUTION.md').read_text(encoding='utf-8')
    entries = index_entries((cache/'README.md').read_bytes(), spec['upstream_index_sha256'])
    expected_roles = {'validation': {'0011', '0028', '0108', '0276'},
                      'test': {'0014', '0033', '0200', '0252'}}
    assert len(spec['sources']) == 42 and len(manifest['protected_sources']) == 28
    for role, ids in expected_roles.items():
        assert {s['id'] for s in spec['sources'] if s['split'] == role} == ids
    assert set(expected_roles['validation']).isdisjoint(expected_roles['test'])
    for source in spec['sources']:
        raw = (cache/source['source_file']).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == source['raw_sha256']
        # These exact upstream editions use LF and the literal page separator.
        chunks = raw.decode('utf-8').split('\n##\n')
        assert chunks[0].strip() == '# '+source['title']
        expected = ('\n\n'.join(page.strip() for page in chunks[1:-1])+'\n').encode('utf-8')
        document = (prepared/(source['id']+'.txt')).read_bytes()
        assert document == expected and hashlib.sha256(document).hexdigest() == source['body_sha256']
        assert len(chunks)-2 == source['pages']
        assert b'* License:' not in document and b'\x1e' not in document
        assert source['attribution']['License'] == '[CC-BY]'
        assert source['index_entry']['license'] == 'CC-BY'
        for key, value in source['attribution'].items():
            assert '- '+key+': '+value in credits
        if 'source_footer' in source:
            assert source['source_footer'] in credits
            assert source['source_footer'] not in document.decode('utf-8')
        assert source['index_entry']['license_url'] in credits
        assert source['source_file'] in credits
        documents[source['id']] = document
    assert 'Jo\u00e3o Carlos Brito' in credits  # Explicit Windows UTF-8 regression.
    for split in ('train', 'validation', 'test'):
        selected = [documents[s['id']] for s in spec['sources'] if s['split'] == split]
        payload = b'\x1e'.join(selected)
        record = manifest['outputs'][split]
        assert payload == (prepared/(split+'.dat')).read_bytes()
        assert sha(prepared/(split+'.dat')) == record['sha256']
        assert len(selected) == record['documents'] and len(payload) == record['bytes']
        assert sum(len(d)-1 for d in selected) == record['within_document_next_byte_pairs']
    assert manifest['outputs']['train']['words'] == 9827
    assert manifest['outputs']['train']['documents'] == 34
    cumulative = []
    for stage in manifest['stages']:
        selected = [documents[s['id']] for s in spec['sources']
                    if s['split'] == 'train' and s['stage'] == stage['id']]
        assert selected and b'\x1e'.join(selected) == (prepared/stage['file']).read_bytes()
        cumulative.extend(selected)
        assert b'\x1e'.join(cumulative) == (prepared/stage['cumulative_file']).read_bytes()
    old, protected, old_ids = assembly.protected_documents(spec, assembly.DEFAULT_PROTECTED)
    assert protected == manifest['protected_sources'] and not (old_ids & set(documents))
    all_documents = list(old.items())+list(documents.items())
    old_count = len(old)
    seen_full, seen_sequences, seen_paragraphs = set(), set(), set()
    for n, (identifier, document) in enumerate(all_documents):
        full, sequences, paragraphs = independent_signatures(document)
        if n >= old_count:
            assert full not in seen_full, identifier
            assert not sequences & seen_sequences, identifier
            assert not paragraphs & seen_paragraphs, identifier
        seen_full.add(full)
        seen_sequences.update(sequences)
        seen_paragraphs.update(paragraphs)
    _, _, probes, _ = examples(assembly.read('data/lessons-binding-v2.json'))
    contexts = {re.sub(r'\s+', ' ', r['context']).strip()
                for split in ('development', 'test') for r in probes[split]}
    train_text = re.sub(r'\s+', ' ', (prepared/'train.dat').read_text(encoding='utf-8'))
    assert not any(context in train_text for context in contexts)

    rejected = []
    def reject(label, message, operation):
        try:
            operation()
        except ValueError as error:
            assert message in str(error), (label, str(error))
            rejected.append(label)
        else:
            raise AssertionError('Accepted negative control: '+label)
    selected = copy.deepcopy(spec['sources'][0])
    raw = (cache/selected['source_file']).read_bytes()
    reject('source-byte-change', 'edition changed', lambda: parse(raw+b'changed', selected))
    reject('index-byte-change', 'index changed', lambda: index_entries((cache/'README.md').read_bytes()+b'changed', spec['upstream_index_sha256']))
    duplicate_index = b'0005 | [Anansi and Turtle](http://example.invalid/source) | [CC-BY](https://creativecommons.org/licenses/by/3.0/)\n'*2
    reject('duplicate-index-identity', 'duplicated', lambda: index_entries(duplicate_index, hashlib.sha256(duplicate_index).hexdigest()))
    edited = copy.deepcopy(selected)
    edited['attribution']['License'] = '[CC-BY-NC]'
    reject('conflicting-source-license', 'agree', lambda: verify_index_entry(edited, entries))
    for label, field, value in [('unsupported-license-version', 'license_url', 'https://creativecommons.org/licenses/by/2.0/'),
                               ('noncommercial-index', 'license', 'CC-BY-NC')]:
        candidate = copy.deepcopy(selected)
        candidate['index_entry'][field] = value
        changed = {**entries, candidate['id']: candidate['index_entry']}
        reject(label, 'agree', lambda candidate=candidate, changed=changed: verify_index_entry(candidate, changed))
    edited = {**selected, 'title': 'Unselected title'}
    reject('wrong-source-title', 'metadata differs', lambda: verify_index_entry(edited, entries))
    edited = {**selected, 'body_sha256': '0'*64}
    reject('changed-body-boundary', 'body changed', lambda: parse(raw, edited))
    reject('unpinned-revision', 'pinned', lambda: load(selected, 'master', entries, cache))
    edited = {**selected, 'source_file': '../0005.md'}
    reject('source-path-escape', 'filename', lambda: load(edited, spec['upstream_commit'], entries, cache))
    footered = next(s for s in spec['sources'] if 'source_footer' in s)
    footer_raw = (cache/footered['source_file']).read_bytes()
    changed_raw = footer_raw.replace(footered['source_footer'].encode('utf-8'), b'A changed editorial note')
    edited = {**footered, 'raw_sha256': hashlib.sha256(changed_raw).hexdigest()}
    reject('resigned-editorial-footer', 'footer changed', lambda: parse(changed_raw, edited))
    reject('existing-output', 'fresh', lambda: assembly.prepare(spec_path, cache, prepared))

    # Inject at the validated-document assembly boundary to prove that the
    # production whole-corpus exclusion gate stops actual contaminated output.
    actual_load = assembly.load
    first_id = selected['id']
    for label, payload in [
        ('old-heldout-in-training', old['early-readers/0156.txt']),
        ('new-test-in-training', documents['0252']),
        ('new-training-duplicate', documents['0013']),
        ('normalized-protected-sequence', (' '.join(independent_signatures(old['stories/11757.txt'])[0][:24])).upper().encode('utf-8'))]:
        def contaminated(source, commit, catalog, raw_cache, payload=payload):
            document, record = actual_load(source, commit, catalog, raw_cache)
            return (document+b'\n\n'+payload if source['id'] == first_id else document), record
        destination = out/label
        with patch.object(assembly, 'load', contaminated):
            reject(label, 'overlaps', lambda: assembly.prepare(spec_path, cache, destination))
        assert not destination.exists()
    for label, mutate, expected_error in [
        ('previously-selected-story-id', lambda s: s['sources'][0].update(id='0008'), 'Previously selected'),
        ('duplicate-selected-id', lambda s: s['sources'].append(s['sources'][0]), 'identities'),
        ('changed-protected-manifest', lambda s: s['protected_editions']['stories'].update(manifest_sha256='0'*64), 'manifest changed')]:
        altered = copy.deepcopy(spec)
        mutate(altered)
        path = out/(label+'.json')
        path.write_bytes((json.dumps(altered, ensure_ascii=False)+'\n').encode('utf-8'))
        destination = out/label
        reject(label, expected_error, lambda: assembly.prepare(path, cache, destination))
        assert not destination.exists()
    real_sha = assembly.sha
    changed_path = assembly.DEFAULT_PROTECTED['early-readers']/'0008.txt'
    with patch.object(assembly, 'sha', lambda p: '0'*64 if Path(p) == changed_path else real_sha(p)):
        reject('changed-protected-body', 'source bytes changed', lambda: assembly.prepare(spec_path, cache, out/'changed-protected-body'))
    assert not (out/'changed-protected-body').exists()
    evidence = dict(passed=True, source_spec_sha256=sha(spec_path), source_manifest_sha256=sha(prepared/'manifest.json'),
                    reproduced_files=len(files), source_count=42, outputs=manifest['outputs'],
                    protected_existing_sources=len(old), protected_existing_evaluation_sources=sum(r['split'] != 'train' for r in protected.values()),
                    old_or_selected_whole_text_overlap=0, normalized_20_word_overlap=0, normalized_substantial_paragraph_overlap=0,
                    heldout_binding_contexts_checked=len(contexts), pinned_source_pages_reproduced=True,
                    source_and_index_licenses_agree=True, publisher_pages_verified=False,
                    utf8_credits_verified=True, editorial_footers_excluded_from_model_text_and_preserved_in_credits=3,
                    split_and_stage_bytes_verified=True, rejected_negative_controls=rejected,
                    model_training_performed=False, reserved_tests_scored=False,
                    code_sha256={p: sha(p) for p in ('scripts/corpus/storybooks.py', 'scripts/corpus/african_storybook.py',
                                                   'scripts/corpus/overlap.py', 'scripts/prepare_reading_breadth.py',
                                                   'tests/reading_breadth_corpus.py')},
                    limits=spec['limitations'])
    (out/'verification.json').write_bytes((json.dumps(evidence, indent=2)+'\n').encode('utf-8'))
    print(json.dumps({k:v for k,v in evidence.items() if k not in ('limits', 'code_sha256')}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--prepared', type=Path, default=Path('data/prepared/reading-breadth-v1'))
    p.add_argument('--raw', type=Path, default=Path('data/raw/asp-b5c3d5b'))
    a = p.parse_args()
    audit(a.out, a.prepared, a.raw)
