"""Selected correction targets, held-out questions and append-only curriculum."""
from pathlib import Path

from corpus.paired_selection import PairReservations, normalized
from corpus.selection import SELECTION, require_training_spec, selection
from extend_curriculum import prepare as extend, read_schedule
from membrane_study_state import require
from native_experiment import read, verified_book_manifest
from prose_founder import file_hash, write
from quantitative_assessment_inputs import admit, authenticate
from quantitative_probes import build, protect

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / 'data/conversation-experiment-v1.json'


def prompt(question):
    return 'Question: ' + question + '\nAnswer: '


def material(source):
    lessons, evaluation = source['lessons'], source['evaluation_only']
    require(source['version'] == 'conversation-corrections-v1' and len(lessons) == 6 and
            len(evaluation) == 12, 'Unexpected correction edition')
    indexed = {r['id']: r for r in lessons}
    require(len(indexed) == 6, 'Duplicate correction identity')
    documents, questions = [], []
    for lesson in lessons:
        raw = (prompt(lesson['question']) + lesson['answer'] + '\n').encode('ascii')
        require(lesson['answer'] in lesson['accepted'] and 2 <= len(raw) <= 129 and
                raw.count(b'\nAnswer: ') == 1 and b'\x1e' not in raw,
                'A correction must fit one native source window with one answer')
        documents.append(raw)
        questions.append(dict(id=lesson['id'], group='practised', question=lesson['question'],
                              accepted=lesson['accepted']))
    for row in evaluation:
        accepted = indexed[row['accepted_from']]['accepted'] if 'accepted_from' in row else row['accepted']
        require(row['group'] in ('reworded', 'new_arithmetic', 'unpractised_facts'), 'Unknown question role')
        require(all(normalized(row['question']) not in normalized(doc.decode('ascii')) for doc in documents),
                'An evaluation-only question appears in a correction')
        questions.append(dict(id=row['id'], group=row['group'], question=row['question'], accepted=accepted))
    require(len({r['id'] for r in questions}) == len({r['question'] for r in questions}) == 18,
            'Duplicate evaluation identity or question')
    require(len(set(documents)) == 6, 'Duplicate correction text')
    for row in questions:
        require(row['question'] and '\n' not in row['question'] and '\x1e' not in row['question'] and
                row['accepted'] and all(answer.strip() and '\n' not in answer for answer in row['accepted']),
                'Malformed conversation question or accepted answer')
    return b'\x1e'.join(documents), questions, sum(len(doc) - 1 for doc in documents)


def prepare(out):
    out = Path(out).resolve()
    require(not out.exists(), 'Use a fresh correction edition directory')
    spec, admitted = read(SPEC), admit()
    require(spec['version'] == 'conversation-experiment-v1' and spec['passes'] == [1, 16, 256] and
            (spec['rate_scale'], spec['answer_scale']) == (.25, 8), 'Unexpected correction policy')
    source_path = ROOT / spec['source']
    source = require_training_spec(source_path)
    base = admitted['cases'][0]
    require(base['profile'] == '105m' and base['checkpoint'] == spec['base'] and
            base['checkpoint_sha256'] == spec['base_sha256'] and
            base['state']['counters']['online_updates'] == spec['base_online_updates'],
            'Correction base differs from the published complete prose checkpoint')
    allowed = selection()['conversation_diagnostic_bases']
    require(len(allowed) == 1 and all(allowed[0][key] == spec[key] for key in ('base', 'base_sha256')) and
            allowed[0]['correction_edition'] == source['version'], 'Correction continuation is not admitted')
    original = Path(spec['original_curriculum'])
    require(file_hash(original) == spec['original_curriculum_sha256'], 'Original curriculum changed')
    old_raw, old_stages = read_schedule(original)
    require(len(old_stages) == 4 and old_stages[-1]['end_update'] == spec['base_online_updates'],
            'Expected the complete four-stage prose curriculum')
    targets, questions, pairs = material(source)
    reservations = PairReservations()
    prepared = Path(spec['retention']['prepared'])
    pins = dict(admitted['authenticated_inputs'])
    manifest = verified_book_manifest(prepared, Path('data/sources-prose-scale-v1.json'))
    roles = {r['id']: r['split'] for r in manifest['sources']}
    retention = spec['retention']
    require(all(roles[n] == 'validation' for n in retention['validation_books']) and
            all(roles[n] == 'train' for n in retention['training_books']) and
            len(set(retention['validation_books'] + retention['training_books'])) == 4,
            'Unexpected retention book selection')
    pins[(prepared / 'manifest.json').as_posix()] = file_hash(prepared / 'manifest.json')
    for record in manifest['sources']:
        path = prepared / f"{record['id']}.txt"
        pins[path.as_posix()] = file_hash(path)
        if record['split'] in ('validation', 'test'):
            reservations.add(path.read_text(encoding='utf-8'), str(path))
    for doc in targets.split(b'\x1e'):
        require(reservations.conflict(doc.decode('ascii'), paired=False) is None,
                'Correction conflicts with a reserved prose document or long paragraph')
    out.mkdir(parents=True)
    (out / 'corrections.dat').write_bytes(targets)
    write(out / 'questions.json', questions)
    # Exact full contexts and single-trial statements remain out of new learning.
    qspec = read(ROOT / 'data/quantitative-probes-v2.json')
    rows, statements, _ = build(qspec)
    protected_source = out / 'corrections.dat'
    protect(rows, statements, [dict(path=str(protected_source), sha256=file_hash(protected_source))])
    extension = extend(original, out / 'corrections.dat', out / 'curriculum',
        spec['base_online_updates'] + 6 * spec['passes'][-1], spec['rate_scale'], 'new', spec['answer_scale'])
    _, new_stages = read_schedule(out / 'curriculum/curriculum.sg')
    require([s['content'] for s in new_stages[:4]] == [s['content'] for s in old_stages] and
            new_stages[-1]['content'] == old_stages[-1]['content'] + b'\x1e' + targets,
            'Correction extension rewrote earlier source material')
    protected_source = out / 'curriculum/edition-5.dat'
    protected = protect(rows, statements, [dict(path=str(protected_source), sha256=file_hash(protected_source))])
    pins.update(protected)
    pins.update({p.as_posix(): file_hash(p) for p in (SPEC, SELECTION, source_path, original,
        ROOT / 'scripts/conversation_material.py', ROOT / 'scripts/extend_curriculum.py',
        ROOT / 'scripts/corpus/paired_selection.py')})
    pins.update({str(s['source']): file_hash(s['source']) for s in old_stages})
    pins.update({p.as_posix(): file_hash(p) for p in out.rglob('*') if p.is_file()})
    result = dict(version='conversation-material-v1', specification=spec, source=source,
        base=base, questions=questions, target_bytes=len(targets), target_pairs_per_pass=pairs,
        extension=extension, source_windows_per_pass=6, authenticated_inputs=pins,
        training_targets_are_generated=False, reserved_tests_scored=False,
        protected_quantitative_contexts=len({r['context'] for r in rows}),
        protected_quantitative_statements=len(statements), native_commands=0)
    authenticate(pins)
    write(out / 'manifest.json', result)
    return result
