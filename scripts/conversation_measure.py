"""Raw native answers and deliberately narrow first-line correction scoring."""
import math
from pathlib import Path
import time

from conversation_material import prompt
from membrane_study_state import read_state, require
from native_experiment import NativeCommands, read
from prose_founder import file_hash, write


def normalize_answer(value):
    return value.strip().rstrip('.!?').strip().casefold()


def score(raw, accepted):
    try:
        text = raw.decode('utf-8', errors='strict')
    except UnicodeDecodeError:
        return dict(first_line_utf8=None, first_line_matches=False, valid_utf8=False)
    lines = text.lstrip(' \t\r\n').split('\n', 1)
    first = lines[0].rstrip('\r')
    return dict(first_line_utf8=first, valid_utf8=True,
                first_line_matches=normalize_answer(first) in {normalize_answer(x) for x in accepted})


def ask(native, checkpoint, row, generation, out, *, context=''):
    prefix = context + prompt(row['question'])
    output = out / (row['id'] + '.txt')
    start = time.perf_counter()
    native('sample', '--checkpoint', checkpoint, '--prompt', prefix, '--tokens', generation['bytes'],
           '--seed', generation['seed'], '--temperature', generation['temperature'],
           '--top-k', generation['top_k'], '--graph', '--output', output)
    seconds = time.perf_counter() - start
    raw, expected = output.read_bytes(), prefix.encode('ascii')
    require(raw.startswith(expected) and len(raw) == len(expected) + generation['bytes'],
            'Native answer prompt or byte budget differs')
    answer = raw[len(expected):]
    return dict(id=row['id'], group=row['group'], question=row['question'], prompt=prefix,
        answer_hex=answer.hex(), answer_utf8=answer.decode('utf-8', errors='replace'),
        sha256=file_hash(output), generated_bytes=len(answer), process_elapsed_seconds=seconds,
        generated_bytes_per_process_second=len(answer)/seconds, **score(answer, row['accepted']))


def assess(runtime, checkpoint, material, out):
    out = Path(out); out.mkdir()
    before, state = file_hash(checkpoint), read_state(checkpoint)
    native = NativeCommands(runtime, out)
    g, r = material['specification']['generation'], material['specification']['retention']
    answers = [ask(native, checkpoint, row, g, out) for row in material['questions']]
    groups = {}
    for row in answers:
        entry = groups.setdefault(row['group'], dict(matches=0, questions=0))
        entry['questions'] += 1
        entry['matches'] += int(row['first_line_matches'])
    books = []
    for role, ids in (('validation', r['validation_books']), ('training_retention', r['training_books'])):
        for ident in ids:
            output = out / f'book-{ident}.json'
            native('evaluate', '--checkpoint', checkpoint, '--data', Path(r['prepared']) / f'{ident}.txt',
                '--batch', r['batch'], '--context', r['context'], '--batches', r['batches'], '--output', output)
            record = read(output)
            require(record['evaluated_bytes'] == r['batch'] * r['context'] * r['batches'] and
                    record['step'] == state['counters']['global_updates'] and
                    math.isfinite(record['loss_nats_per_byte']) and record['loss_nats_per_byte'] >= 0,
                    'Invalid retention assessment')
            books.append(dict(book=ident, role=role, **record))
    require(file_hash(checkpoint) == before, 'Read-only conversation assessment changed its checkpoint')
    result = dict(complete=True, checkpoint=str(checkpoint), checkpoint_sha256=before,
        checkpoint_unchanged=True, state=state, answers=answers, first_line_scores=groups, books=books,
        native_commands=len(native.commands), reserved_tests_scored=False,
        artifact_sha256={p.name: file_hash(p) for p in out.iterdir() if p.is_file()})
    write(out / 'result.json', result)
    return result


def contextual_control(runtime, checkpoint, material, out):
    """Show a correction in the prompt without making a weight update."""
    out = Path(out); out.mkdir()
    native, before = NativeCommands(runtime, out), file_hash(checkpoint)
    results = []
    for lesson, row in zip(material['source']['lessons'], material['questions'][:6]):
        context = prompt(lesson['question']) + lesson['answer'] + '\n\n'
        results.append(ask(native, checkpoint, row, material['specification']['generation'], out, context=context))
    require(file_hash(checkpoint) == before, 'Context-only control changed the checkpoint')
    result = dict(complete=True, weight_updates=0, checkpoint_sha256=before, answers=results,
                  first_line_matches=sum(r['first_line_matches'] for r in results), questions=6)
    write(out / 'result.json', result)
    return result
