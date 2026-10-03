"""Shared native book losses and raw generations for checkpoint experiments."""
from pathlib import Path

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
