"""Shared native book losses and raw generations for checkpoint experiments."""
from pathlib import Path
import math

from native_experiment import read, sha


def measure(native, checkpoint, spec, out, additional_books=()):
    """The caller authenticates sources/checkpoint and owns the command journal."""
    prepared = Path(spec['prepared'])
    book_rows, generated = [], []
    e, g = spec['evaluation'], spec['generation']
    groups = [('validation', spec['validation_books']),
              ('training_retention', spec['training_retention_books']), *additional_books]
    ids = [ident for _, selected in groups for ident in selected]
    if len(ids) != len(set(ids)):
        raise ValueError('Assessment book roles must be disjoint')
    for role, selected in groups:
        for ident in selected:
            report = out / f'book-{ident}.json'
            native('evaluate', '--checkpoint', checkpoint, '--data', prepared / f'{ident}.txt',
                   '--batch', e['batch'], '--context', e['context'], '--batches', e['batches'],
                   '--output', report)
            result = read(report)
            assert result['evaluated_bytes'] == e['target_bytes_per_book']
            book_rows.append(dict(book=ident, role=role, **result))
    for i, prompt in enumerate(g['prompts']):
        output = out / f'sample-{i}.txt'
        native('sample', '--checkpoint', checkpoint, '--prompt', prompt, '--tokens', g['bytes'],
               '--seed', g['seed'], '--temperature', g['temperature'], '--top-k', g['top_k'],
               '--graph', '--output', output)
        raw = output.read_bytes()
        assert raw.startswith(prompt.encode()) and len(raw) == len(prompt.encode()) + g['bytes']
        generated.append(dict(prompt=prompt, output_utf8=raw.decode('utf-8', errors='replace'),
                              output_hex=raw.hex(), sha256=sha(output), generated_bytes=g['bytes']))
    validation = [r['loss_nats_per_byte'] for r in book_rows if r['role'] == 'validation']
    return dict(books=book_rows, samples=generated,
                validation_mean_nats_per_byte=sum(validation) / len(validation))


def verify_saved(directory, row, spec, global_update, additional_books=()):
    """Verify recorded assessment bytes and commands without running a model."""
    directory, checkpoint = Path(directory), Path(row['checkpoint'])
    e, g = spec['evaluation'], spec['generation']
    groups = [('validation', spec['validation_books']),
              ('training_retention', spec['training_retention_books']), *additional_books]
    roles = [(role, ident) for role, ids in groups for ident in ids]
    assert len(roles) == len({ident for _, ident in roles}), 'Duplicate assessment book role'
    assert [(b['role'], b['book']) for b in row['books']] == roles, 'Assessment book order or role differs'
    wanted, hashes = [], {}
    for book, (_, ident) in zip(row['books'], roles):
        output = directory / f'book-{ident}.json'
        assert {k: v for k, v in book.items() if k not in ('book', 'role')} == read(output), 'Saved book result differs'
        assert book['evaluated_bytes'] == e['target_bytes_per_book'] and book['step'] == global_update
        assert all(math.isfinite(book[k]) and book[k] >= 0 for k in ('loss_nats_per_byte', 'bits_per_byte'))
        wanted.append(list(map(str, ['evaluate', '--checkpoint', checkpoint, '--data',
            Path(spec['prepared']) / f'{ident}.txt', '--batch', e['batch'], '--context', e['context'],
            '--batches', e['batches'], '--output', output])))
        hashes[output.as_posix()] = sha(output)
    assert len(row['samples']) == len(g['prompts']), 'Missing or extra generated sample'
    for index, (sample, prompt) in enumerate(zip(row['samples'], g['prompts'])):
        output = directory / f'sample-{index}.txt'
        raw, prefix = output.read_bytes(), prompt.encode('utf-8')
        assert sample['prompt'] == prompt and raw.startswith(prefix), 'Generated prompt differs'
        assert len(raw) == len(prefix) + g['bytes'] and sample['generated_bytes'] == g['bytes']
        assert raw.hex() == sample['output_hex'] and sha(output) == sample['sha256'], 'Raw generation differs'
        assert raw.decode('utf-8', errors='replace') == sample['output_utf8'], 'Displayed generation differs'
        wanted.append(list(map(str, ['sample', '--checkpoint', checkpoint, '--prompt', prompt,
            '--tokens', g['bytes'], '--seed', g['seed'], '--temperature', g['temperature'],
            '--top-k', g['top_k'], '--graph', '--output', output])))
        hashes[output.as_posix()] = sha(output)
    commands_path = directory / 'commands.json'
    commands = read(commands_path)
    assert len(commands) == len(wanted) and [c[1:] for c in commands] == wanted, 'Assessment command arguments differ'
    assert all(Path(c[0]).resolve() == Path('build/synapticgenesis.exe').resolve() for c in commands)
    hashes[commands_path.as_posix()] = sha(commands_path)
    for index in range(1, len(commands) + 1):
        log = directory / f'command-{index:03d}.log'
        hashes[log.as_posix()] = sha(log)
    values = [b['loss_nats_per_byte'] for b in row['books'] if b['role'] == 'validation']
    assert values and sum(values) / len(values) == row['validation_mean_nats_per_byte'], 'Validation mean differs'
    assert row['checkpoint_unchanged'] and not row['reserved_tests_scored']
    return commands, hashes
