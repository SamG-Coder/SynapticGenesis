"""Source authentication and fixed window schedules for adaptation diagnostics."""
import hashlib
from pathlib import Path
import random

from native_experiment import read, sha, write


def fnv(raw, value=14695981039346656037):
    for byte in raw:
        value = ((value ^ byte) * 1099511628211) & ((1 << 64) - 1)
    return value


def verified_edition(directory, specification):
    manifest, selection = read(directory / 'manifest.json'), read(specification)
    if (sha(specification) != manifest['source_spec_sha256']
            or (directory / 'source-spec.json').read_bytes() != specification.read_bytes()
            or manifest['upstream_commit'] != selection['upstream_commit']):
        raise ValueError('Early-reader source selection changed')
    expected = [(s['id'], s['split'], s['stage']) for s in selection['sources']]
    if expected != [(s['id'], s['split'], s['stage']) for s in manifest['sources']]:
        raise ValueError('Early-reader source roles changed')
    for source in manifest['sources']:
        path = directory / (source['id'] + '.txt')
        if sha(path) != source['body_sha256'] or path.stat().st_size != source['clean_bytes']:
            raise ValueError('Early-reader document changed')
    for split in ('train', 'validation', 'test'):
        path, output = directory / (split + '.dat'), manifest['outputs'][split]
        documents = [(directory / (s['id'] + '.txt')).read_bytes()
                     for s in manifest['sources'] if s['split'] == split]
        if (b'\x1e'.join(documents) != path.read_bytes() or sha(path) != output['sha256']
                or path.stat().st_size != output['bytes'] or len(documents) != output['documents']):
            raise ValueError('Early-reader assembled split changed')
    return manifest


def prepare_schedule(source, seed, steps, out):
    raw = source.read_bytes()
    documents = raw.split(b'\x1e')
    if not documents or min(map(len, documents)) < 129 or not 0 < steps <= 65536 or seed <= 0:
        raise ValueError('Schedule requires a positive seed/budget and all documents of at least 129 bytes')
    if out.exists() or out.with_suffix('.json').exists():
        raise ValueError('Use fresh schedule files')
    rng, records = random.Random(seed), []
    lines = [f'SGADAPT1 128 {steps} {fnv(raw)}']
    counts = [0] * len(documents)
    for update in range(steps):
        document = rng.randrange(len(documents))
        offset = rng.randrange(len(documents[document]) - 128)
        window = documents[document][offset:offset + 129]
        counts[document] += 1
        lines.append(f'{document} {offset} {fnv(window)}')
        records.append(dict(update=update + 1, document=document, offset=offset,
                            sha256=hashlib.sha256(window).hexdigest()))
    out.write_bytes(('\n'.join(lines) + '\n').encode())
    result = dict(seed=seed, updates=steps, source_sha256=sha(source), schedule_sha256=sha(out),
                  context=128, input_bytes_per_window=129, presented_pairs=steps * 128,
                  selection='Python random.Random(seed): uniform document then uniform valid byte start.',
                  document_updates=counts, windows=records)
    write(out.with_suffix('.json'), result)
    return result
