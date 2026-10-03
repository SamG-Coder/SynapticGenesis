"""Check paired-data integrity, held-out protection and the actual prepared edition."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from corpus.paired_selection import normalized, probe_contexts, question, select_pairs
from corpus.selection import require_training_spec
from native_experiment import read, sha
from prepare_lessons import write_probes
from prepare_prealgebra import protected_inputs, reviewed_candidates


def fixtures():
    def row(ident, prompt, answer, split='train'):
        return dict(id=ident, split=split, text=f'Arithmetic\n\nQuestion: {prompt}\n\nAnswer: Solution\n\n{answer}\n')

    first = row('held', 'What is 2 + 3?', 'The sum is 5.', 'test')
    changed_answer = row('changed', 'What is 2 + 3?', 'Incorrect alternate answer.')
    kept, omitted = select_pairs([changed_answer, first], [])
    assert kept == [first] and omitted == [dict(id='changed', split='train', kind='question', reserved_by='held')]
    assert kept[0]['text'] == first['text']
    long_answer = 'A complete answer paragraph is reserved as a single unit. ' * 3
    source = row('source', 'First question?', long_answer, 'validation')
    overlap = row('overlap', 'Different question?', 'Opening explanation.\n\n' + long_answer)
    kept, omitted = select_pairs([overlap, source], [])
    assert kept == [source] and omitted[0]['kind'] == 'paragraph_at_least_120_characters'
    kept, omitted = select_pairs([overlap], [('earlier-heldout', long_answer.encode(), 'documents')])
    assert not kept and omitted[0]['reserved_by'] == 'earlier-heldout'
    distinct = [row('minus', '-2 + 3?', '1'), row('plus', '2 + 3?', '5'),
                row('decimal', '0.5 + 1?', '1.5'), row('integer', '5 + 1?', '6'),
                row('lower', 'Value of x?', 'Unknown'), row('upper', 'Value of X?', 'Unknown')]
    assert select_pairs(distinct, [])[0] == distinct
    spaced = row('spaced', 'What is 2  +\n3?', '5')
    assert select_pairs([first, spaced], [])[1][0]['kind'] == 'question'
    rejected = 0
    for text in ('No markers', first['text'].replace('Question:', 'Prompt:'),
                 first['text'].replace('The sum is 5.', ''), first['text'] + '\x1e',
                 first['text'] + '\n\nAnswer: Another answer'):
        try:
            question(text)
        except ValueError:
            rejected += 1
        else:
            raise AssertionError('Malformed pair accepted')
    with TemporaryDirectory() as folder:
        path = Path(folder) / 'held.sgprobe'
        context = 'The "key" is in the box. A path uses \\ separators.\n'
        write_probes(path, [dict(id='a', pair='a', skill='a', correct=0, context=context,
            query='Where?\nAnswer: ', choice0='box', choice1='bag')], 'SGPROBE2')
        raw = path.read_bytes()
        assert probe_contexts(raw) == [normalized(context)]
        probe_pair = row('probe-copy', context + ' Where is the key?', 'In the box.')
        assert select_pairs([probe_pair], [('probe', raw, 'probe')])[1][0]['kind'] == 'authored_probe_context'
        for bad in (raw.replace(b'SGPROBE2 1', b'SGPROBE2 2'), b'SGPROBE9 0', b'SGPROBE1 1'):
            try:
                probe_contexts(bad)
            except ValueError:
                rejected += 1
            else:
                raise AssertionError('Malformed probe accepted')
    return dict(whole_pair_omission=True, duplicate_question_with_changed_answer_rejected=True,
        reserved_answer_and_old_paragraph_protected=True, signs_decimals_and_variable_case_preserved=True,
        probe_quote_backslash_newline_roundtrip=True, authored_context_protected=True,
        malformed_inputs_rejected=rejected)


def audit(prepared, candidates, cache):
    fixture_result = fixtures()
    spec_path = Path('data/sources-prealgebra-v1.json')
    spec = require_training_spec(spec_path)
    review, rows, _ = reviewed_candidates(candidates, Path(spec['passage_review']), cache)
    manifest = read(prepared / 'manifest.json')
    assert (prepared / 'source-spec.json').read_bytes() == spec_path.read_bytes()
    assert (prepared / 'passage-review.json').read_bytes() == Path(spec['passage_review']).read_bytes()
    assert manifest['source_spec_sha256'] == sha(spec_path)
    assert len(review['records']) == 261 and len(rows) == 221
    assert sum(r['decision'] == 'exclude' for r in review['records']) == 40
    for row in rows:
        row.update(spec['chapter_policy'][row['chapter']])
    selected, omitted = select_pairs(rows, protected_inputs(spec['protected_files']))
    assert omitted == spec['overlap_omissions'] == []
    by_id = {r['id']: r for r in selected}
    assert len(by_id) == len(spec['sources']) == len(manifest['sources']) == 221
    for record in manifest['sources']:
        body = (prepared / f"{record['id']}.txt").read_bytes()
        assert body == by_id[record['id']]['text'].encode('utf-8')
        assert hashlib.sha256(body).hexdigest() == record['clean_sha256'] == record['text_sha256']
        assert len(body) == record['clean_bytes']
        assert record['split'] == spec['chapter_policy'][record['chapter']]['split']
        question(body.decode('utf-8'))
    for split, expected_count in [('train', 168), ('validation', 29), ('test', 24)]:
        rows = [r for r in spec['sources'] if r['split'] == split]
        assert len(rows) == expected_count
        body = b'\x1e'.join((prepared / f"{r['id']}.txt").read_bytes() for r in rows)
        output = manifest['outputs'][split]
        assert body == (prepared / f'{split}.dat').read_bytes()
        assert sha(prepared / f'{split}.dat') == output['sha256']
        assert len(body) == output['bytes'] and len(rows) == output['documents']
        assert len(re.findall(r"[A-Za-z]+(?:['-][A-Za-z]+)*", body.decode())) == output['word_like_units']
    for stage in manifest['stages']:
        body = b'\x1e'.join((prepared / f"{r['id']}.txt").read_bytes() for r in spec['sources']
                            if r['split'] == 'train' and r['stage'] == stage['id'])
        assert body == (prepared / stage['file']).read_bytes()
        assert sha(prepared / stage['file']) == stage['sha256'] and len(body) == stage['bytes']
    for name, output in [('license', 'LICENSE-source.txt'), ('preface', 'original-preface.cnxml'),
                         ('collection', 'original-collection.xml')]:
        assert sha(prepared / output) == next(r['sha256'] for r in spec['auxiliary_files'] if r['name'] == name)
    assert (prepared / 'ATTRIBUTION.txt').read_text(encoding='utf-8') == spec['attribution'] + '\n'
    negative = []
    with TemporaryDirectory() as folder:
        path = Path(folder) / 'review.json'
        for label in ('missing_decision', 'unknown_decision', 'changed_text_hash', 'changed_identity', 'changed_source_pin'):
            damaged = copy.deepcopy(review)
            if label == 'missing_decision': damaged['records'].pop()
            if label == 'unknown_decision': damaged['records'][0]['decision'] = 'unreviewed'
            if label == 'changed_text_hash': damaged['records'][0]['text_sha256'] = '0' * 64
            if label == 'changed_identity': damaged['records'][0]['exercise'] = 'fs-id0'
            if label == 'changed_source_pin': damaged['upstream_commit'] = '0' * 40
            path.write_text(json.dumps(damaged), encoding='utf-8')
            try:
                reviewed_candidates(candidates, path, cache)
            except ValueError:
                negative.append(label)
            else:
                raise AssertionError('Altered review accepted: ' + label)
        heldout = Path(folder) / 'heldout.dat'
        heldout.write_bytes(b'Original protected text.')
        record = dict(path=str(heldout), sha256=sha(heldout), kind='documents')
        heldout.write_bytes(b'Changed protected text.')
        try:
            protected_inputs([record])
        except ValueError:
            negative.append('changed_protected_source')
        else:
            raise AssertionError('Altered held-out bytes accepted')
    return dict(passed=True, reviewed_candidates=261, excluded_by_passage_review=40,
        retained_complete_lessons=221, outputs=manifest['outputs'], protected_files=len(spec['protected_files']),
        fixtures=fixture_result, altered_reviews_rejected=negative, all_prepared_pairs_match_reviewed_source_bytes=True,
        all_source_modules_authenticated=75, complete_chapters_reserved=True,
        examples=[dict(index=i, **{k: read(candidates)['candidates'][i][k] for k in
                   ('module', 'exercise', 'source_url', 'text', 'text_sha256')},
                   review=review['records'][i]) for i in (24, 126, 130, 142, 159, 185, 203, 258, 260)],
        attribution=spec['attribution'], source_spec_sha256=sha(spec_path),
        prepared_manifest_sha256=sha(prepared / 'manifest.json'), passage_review_sha256=sha(spec['passage_review']),
        implementation_sha256={str(p): sha(p) for p in (Path(__file__), Path('scripts/prepare_prealgebra.py'),
                               Path('scripts/corpus/paired_selection.py'))},
        native_calls=0, models_trained=False, limitations=spec['limitations'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--candidates', type=Path, required=True)
    parser.add_argument('--cache', type=Path, default=Path('runs/science-source-review'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists(), 'Use a fresh report path'
    result = audit(args.prepared, args.candidates, args.cache)
    args.out.write_bytes((json.dumps(result, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))
    print(json.dumps({k: v for k, v in result.items() if k in
        ('passed', 'retained_complete_lessons', 'outputs', 'altered_reviews_rejected', 'fixtures')}, indent=2))
