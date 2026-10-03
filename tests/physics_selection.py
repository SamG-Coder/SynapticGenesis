"""Verify the Physics selection, complete modules, and evaluation reservations."""
import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from corpus.paired_selection import select_documents
from native_experiment import read, sha
from prepare_physics import compiled_selection, prepare
from prose_founder import write
from review_physics_edition import compile_edition


def rejected(call):
    try:
        call()
    except ValueError:
        return
    raise AssertionError('Invalid Physics input was accepted')


def fixtures():
    def row(ident, text, split='train'):
        return dict(id=ident, text=text, split=split)

    held = row('held', 'Whole text\n\nA small independent passage.', 'test')
    duplicate = row('duplicate', 'Whole  text\n\nA small independent passage.')
    kept, omitted = select_documents([duplicate, held], [])
    assert kept == [held] and omitted == [dict(id='duplicate', split='train',
        kind='whole_document', reserved_by='held')]
    earlier = 'A complete paragraph belongs to a previously reserved evaluation source. ' * 3
    later = 'This other paragraph occurs in a rejected held-out module and must still be reserved. ' * 3
    held = row('rejected-heldout', 'Held-out title\n\n' + earlier + '\n\n' + later, 'validation')
    train = row('rejected-train', 'Different title\n\n' + later)
    kept, omitted = select_documents([train, held], [('previous-test', earlier.encode(), 'documents')])
    assert not kept and [r['reserved_by'] for r in omitted] == ['previous-test', 'rejected-heldout']
    assert all(r['kind'] == 'paragraph_at_least_120_characters' for r in omitted)
    # The second paragraph cannot leak when the first one rejected its whole held-out module.
    untouched = row('intact', 'Lesson\n\nQuestion: What changes?\n\nAnswer: Nothing is cut out.')
    assert select_documents([untouched], [])[0] == [untouched]
    test = row('test', earlier, 'test')
    validation = row('validation', earlier, 'validation')
    assert select_documents([validation, test], [])[0] == [test]
    context = 'An authored evaluation context lives here.'
    probe = ('SGPROBE2 1 "id" "pair" "skill" 0 "' + context + '" "Query" "a" "b"').encode()
    probe_copy = row('probe-copy', 'Opening.\n\n' + context + '\n\nA conclusion.')
    assert select_documents([probe_copy], [('probe', probe, 'probe')])[1] == [
        dict(id='probe-copy', split='train', kind='authored_probe_context', reserved_by='probe')]
    distinct = [row(str(i), text) for i, text in enumerate(('x = -2', 'x = 2', 'X = 2', 'x = 0.2'))]
    assert select_documents(distinct, [])[0] == distinct
    for rows in ([row('empty', '  ')], [row('boundary', 'First\x1eSecond')],
                 [row('unknown', 'Text', 'holdout')], [untouched, untouched]):
        rejected(lambda rows=rows: select_documents(rows, []))
    rejected(lambda: select_documents([untouched], [('bad', b'\x1e', 'documents')]))
    rejected(lambda: select_documents([untouched], [('bad', b'Text', 'unknown')]))
    return dict(whole_document_omission=True, normalized_whole_text_protected=True,
        rejected_holdout_other_paragraph_still_reserved=True, test_precedes_validation=True,
        question_and_answer_remain_intact=True, authored_probe_context_protected=True,
        mathematical_signs_decimals_and_case_distinct=True, malformed_documents_rejected=6)


def altered_inputs(spec, cache):
    failures = []
    with TemporaryDirectory() as folder:
        root = Path(folder)
        review_path = root / 'passage.json'
        review = read(spec['passage_review'])
        cases = (
            ('missing_sample', lambda p: p['reviewed_samples'].pop()),
            ('duplicate_sample', lambda p: p['reviewed_samples'].__setitem__(1, copy.deepcopy(p['reviewed_samples'][0]))),
            ('empty_sample', lambda p: p['reviewed_samples'][0].update(sample='')),
            ('changed_sample', lambda p: p['reviewed_samples'][0].update(sample='A passage that is not in this edition.')),
            ('changed_sample_hash', lambda p: p['reviewed_samples'][0].update(text_sha256='0'*64)),
            ('changed_identity', lambda p: p['reviewed_samples'][0].update(title='Different lesson')),
            ('unknown_decision', lambda p: p['reviewed_samples'][0].update(decision='unreviewed')),
            ('missing_correction_decision', lambda p: p['reviewed_samples'][3].update(correction_ids=[])),
            ('different_extraction', lambda p: p.update(source_review_sha256='0'*64)),
            ('missing_correction', lambda p: p['text_corrections'].pop()),
            ('ambiguous_correction', lambda p: p['text_corrections'][0].update(before='the')),
            ('empty_correction', lambda p: p['text_corrections'][0].update(after='')),
            ('absent_correction_target', lambda p: p['text_corrections'][0].update(before='Not present in the source')),
            ('different_image', lambda p: p['additional_inspected_media'][0].update(raw_sha256='0'*64)),
            ('changed_image_blob', lambda p: p['additional_inspected_media'][0].update(git_blob_sha1='0'*40)),
            ('untrusted_image_url', lambda p: p['additional_inspected_media'][0].update(source_url='https://example.com/diagram.jpg')),
        )
        for name, mutate in cases:
            damaged = copy.deepcopy(review)
            mutate(damaged)
            write(review_path, damaged)
            changed = copy.deepcopy(spec)
            changed.update(passage_review=str(review_path), passage_review_sha256=sha(review_path))
            rejected(lambda: compiled_selection(changed, cache))
            failures.append(name)
        cases = (
            ('changed_review_pin', lambda p: p.update(passage_review_sha256='0'*64)),
            ('changed_upstream', lambda p: p.update(upstream_commit='0'*40)),
            ('heldout_chapter_in_train', lambda p: p['chapter_policy']['Momentum'].update(split='train')),
            ('incorrect_topic_stage', lambda p: p['chapter_policy']['Momentum'].update(stage=1)),
            ('duplicate_stage', lambda p: p['stages'].__setitem__(1, copy.deepcopy(p['stages'][0]))),
            ('missing_module', lambda p: p['sources'].pop()),
            ('changed_module_hash', lambda p: p['sources'][0].update(corrected_sha256='0'*64)),
            ('changed_module_identity', lambda p: p['sources'][0].update(id='m00000')),
            ('changed_overlap_decision', lambda p: p['overlap_omissions'].append(dict(id='unexpected'))),
            ('changed_protected_text_pin', lambda p: p['protected_files'][0].update(sha256='0'*64)),
        )
        for name, mutate in cases:
            changed = copy.deepcopy(spec)
            mutate(changed)
            rejected(lambda: compiled_selection(changed, cache))
            failures.append(name)
        # Admission is checked before output creation, even if a caller renames a modified source spec.
        changed = copy.deepcopy(spec)
        changed['sources'].pop()
        source_path = root / 'renamed-edition.json'
        write(source_path, changed)
        rejected(lambda: prepare(source_path, cache, root / 'output'))
        assert not (root / 'output').exists()
        failures.append('altered_edition_rejected_before_output')
    return failures


def prepared_outputs(prepared, spec_path, spec, bodies, auxiliary, edits, passage):
    manifest = read(prepared / 'manifest.json')
    assert manifest['source_spec_sha256'] == sha(spec_path)
    for name, source in [('source-spec.json', spec_path), ('source-edits.json', spec['source_edits']),
                         ('passage-review.json', spec['passage_review'])]:
        assert (prepared / name).read_bytes() == Path(source).read_bytes()
    assert len(manifest['sources']) == len(bodies) == 99
    for row, expected in zip(manifest['sources'], spec['sources']):
        body = bodies[row['id']]
        assert row == dict(expected, clean_bytes=len(body), clean_sha256=sha(prepared / f"{row['id']}.txt"))
        assert (prepared / f"{row['id']}.txt").read_bytes() == body
        assert row['clean_sha256'] == row['corrected_sha256']
    for split, count in [('train', 90), ('validation', 4), ('test', 5)]:
        rows = [r for r in spec['sources'] if r['split'] == split]
        raw = b'\x1e'.join(bodies[r['id']] for r in rows)
        path = prepared / f'{split}.dat'
        assert path.read_bytes() == raw and len(rows) == count
        assert manifest['outputs'][split] == dict(documents=count, bytes=len(raw), sha256=sha(path),
            word_like_units=len(re.findall(r"[A-Za-z]+(?:['-][A-Za-z]+)*", raw.decode('utf-8'))))
    assert len(manifest['stages']) == 4
    for expected, stage in zip(spec['stages'], manifest['stages']):
        raw = b'\x1e'.join(bodies[r['id']] for r in spec['sources'] if r['split'] == 'train' and r['stage'] == stage['id'])
        path = prepared / f"stage-{stage['id']}.dat"
        assert path.read_bytes() == raw
        assert stage == dict(expected, file=path.name, bytes=len(raw), sha256=sha(path))
    for key, name in [('license', 'LICENSE-source.txt'), ('preface', 'original-preface.cnxml'),
                      ('collection', 'original-collection.xml')]:
        assert (prepared / name).read_bytes() == auxiliary[key]
    assert (prepared / 'ATTRIBUTION.txt').read_text(encoding='utf-8') == spec['attribution'] + '\n'
    assert not manifest['models_trained'] and not manifest['curriculum_created']
    assert manifest['protected_files'] == spec['protected_files'] and manifest['overlap_omissions'] == []
    rejected(lambda: prepare(spec_path, Path('unused-cache'), prepared))
    return manifest


def audit(spec_path, cache, prepared=None):
    spec = read(spec_path)
    fixture_result = fixtures()
    bodies, auxiliary, edits, passage = compiled_selection(spec, cache)
    _, _, _, original, records = compile_edition(Path(spec['source_review_spec']), Path(spec['source_edits']), cache)
    assert len(bodies) == len(records) == 99 and len(spec['protected_files']) == 18
    expected_counts = Counter(train=90, validation=4, test=5)
    assert Counter(r['split'] for r in spec['sources']) == expected_counts
    assert {r['chapter_number'] for r in spec['sources'] if r['split']=='validation'} == {8}
    assert {r['chapter_number'] for r in spec['sources'] if r['split']=='test'} == {19}
    assert all(r['chapter_number'] not in (8,19) for r in spec['sources'] if r['split']=='train')
    corrections = {r['module']: r for r in passage['text_corrections']}
    assert len(corrections) == 5
    for ident, body in bodies.items():
        if ident in corrections:
            change = corrections[ident]
            before, after = change['before'].encode(), change['after'].encode()
            assert original[ident].count(before) == 1 and body.count(before) == 0 and body.count(after) == 1
            assert body == original[ident].replace(before, after, 1)
        else:
            assert body == original[ident]
        assert b'\x1e' not in body
    assert 'A net force changes an object’s velocity' in bodies['m54135'].decode()
    assert 'along the negative direction of the chosen axis' in bodies['m54215'].decode()
    assert b'radiation across empty space' in bodies['m54302']
    assert b'material rather than visible color alone' in bodies['m54566']
    assert b'the metal mercury and the nonmetal bromine as liquids' in bodies['m54602']
    failures = altered_inputs(spec, cache)
    manifest = prepared_outputs(prepared, spec_path, spec, bodies, auxiliary, edits, passage) if prepared else None
    return dict(passed=True, source_spec_sha256=sha(spec_path), complete_modules=99,
        split_modules=dict(expected_counts), protected_files=18, representative_passages=30,
        reviewed_chapters=23, all_module_bytes_match_declared_edits=True, text_adaptations=5,
        earlier_extraction_preserved_in_other_modules=94, complete_chapters_reserved=True,
        fixtures=fixture_result, altered_inputs_rejected=failures,
        prepared_manifest_sha256=sha(prepared/'manifest.json') if prepared else None,
        outputs=manifest['outputs'] if manifest else None, stages=manifest['stages'] if manifest else None,
        implementation_sha256={str(p):sha(p) for p in (Path(__file__),Path('scripts/prepare_physics.py'),
                               Path('scripts/corpus/paired_selection.py'))},
        native_calls=0, models_trained=False, limitations=spec['limitations'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sources', type=Path, default=Path('data/sources-physics-v1.json'))
    parser.add_argument('--cache', type=Path, default=Path('runs/science-source-review'))
    parser.add_argument('--prepared', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists(), 'Use a fresh report path'
    result = audit(args.sources, args.cache, args.prepared)
    write(args.out, result)
    print(json.dumps({k:result[k] for k in ('passed','complete_modules','split_modules','text_adaptations','outputs')}, indent=2))
    print(f"{len(result['altered_inputs_rejected'])} altered inputs rejected; no native model work.")
