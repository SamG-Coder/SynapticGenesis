"""Prepare the explicitly selected original lessons and context-reversal probes.

This is data preparation only. All model computation remains native C++/CUDA.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import random
import shutil


def quoted(value):
    # C++ std::quoted strings preserve actual newlines and escape quotes/slashes.
    return '"' + value.replace('\\', '\\\\').replace('"', '\\"') + '"'


def write_probes(path, rows):
    lines = [f'SGPROBE1 {len(rows)}\n']
    for row in rows:
        lines.append(' '.join([quoted(row[k]) for k in ('id', 'pair', 'skill')] +
                              [str(row['correct'])] +
                              [quoted(row[k]) for k in ('context', 'query', 'choice0', 'choice1')]) + '\n')
    path.write_bytes(''.join(lines).encode('ascii'))


def prepare(spec_path, out, reading=None):
    spec_bytes = spec_path.read_bytes()
    spec = json.loads(spec_bytes)
    assert spec['version'] == 'relations-v1'
    objects, locations, templates = spec['objects'], spec['locations'], spec['templates']
    assert len(objects) == 6 and len(set(objects)) == 6
    assert len(locations) == 4 and len(set(locations)) == 4
    assert all(word.isascii() and word.isalpha() and len(word) == 3 for word in objects + locations)
    assert spec['skills'] == ['single_fact', 'two_facts', 'changed_fact']
    split = spec['split']
    assert sorted([split['development_distance'], split['test_distance'], *split['train_distances']]) == list(range(1, 6))
    train = set()
    probes = {'development': [], 'test': [], 'train': []}
    pair_members = {'development': set(), 'test': set(), 'train': set()}
    for i, obj in enumerate(objects):
        for j, other in enumerate(objects):
            if i == j:
                continue
            distance = (j - i) % len(objects)
            partition = ('development' if distance == split['development_distance'] else
                         'test' if distance == split['test_distance'] else 'train')
            pair_members[partition].add((obj, other))
            for loc0, loc1 in itertools.combinations(locations, 2):
                for skill in spec['skills']:
                    pair_id = f'{skill}-{obj}-{other}-{loc0}-{loc1}'
                    for reverse in (0, 1):
                        loc, alt = (loc0, loc1) if reverse == 0 else (loc1, loc0)
                        fields = dict(object=obj, other=other, location=loc, alternative=alt)
                        context = templates[skill].format(**fields)
                        query = templates['query'].format(**fields)
                        answer = templates['answer'].format(**fields)
                        if partition == 'train':
                            train.add(context + query + answer + '\n')
                        if skill == 'two_facts':
                            probes[partition].append(dict(id=f'{pair_id}-{reverse}', pair=pair_id,
                                skill=skill, correct=reverse, context=context, query=query,
                                choice0=templates['answer'].format(location=loc0),
                                choice1=templates['answer'].format(location=loc1)))
    assert all(not pair_members[a] & pair_members[b]
               for a, b in itertools.combinations(pair_members, 2))
    for partition in ('development', 'test'):
        for row in probes[partition]:
            lesson = row['context'] + row['query'] + row[f"choice{row['correct']}"] + '\n'
            assert lesson not in train
            assert not any(row['context'] in item for item in train)
    documents = sorted(train)
    random.Random(819731).shuffle(documents)
    original = reading.read_bytes() if reading is not None else None
    if original is not None:
        if not original or original.endswith(b'\x1e'):
            raise ValueError('Prior reading corpus must contain text and end inside a document')
        normalized = original.replace(b'\r\n', b'\n').replace(b'\r', b'\n')
        for partition in ('development', 'test'):
            for row in probes[partition]:
                if row['context'].encode('ascii') in normalized:
                    raise ValueError('Prior reading corpus contains a held-out probe context')
    out.mkdir(parents=True, exist_ok=False)
    (out / 'train.dat').write_bytes(b'\x1e'.join(doc.encode('ascii') for doc in documents))
    shutil.copyfile(spec_path, out / 'source-spec.json')
    for partition, rows in probes.items():
        write_probes(out / f'{partition}.sgprobe', rows)
    if original is not None:
        (out / 'reading.dat').write_bytes(original)
        (out / 'reading-lessons.dat').write_bytes(original + b'\x1e' + (out / 'train.dat').read_bytes())
        for scale in (1, 64):
            (out / f'answer-{scale}.sg').write_bytes(
                f'SGCURRICULUM3\n6000 "reading.dat" 1 all 1\n18000 "reading-lessons.dat" 0.25 new {scale}\n'.encode('ascii'))
    files = {p.name: {'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
             for p in sorted(out.iterdir())}
    manifest = dict(version=spec['version'], source_spec_sha256=hashlib.sha256(spec_bytes).hexdigest(),
                    documents=len(documents), files=files,
                    object_pairs={k: sorted(v) for k, v in pair_members.items()},
                    probe_items={k: len(v) for k, v in probes.items()},
                    heldout_contexts_absent_from_training=True,
                    predefined_test_not_for_tuning=True,
                    reading_source=str(reading) if reading is not None else None,
                    provenance=spec['provenance'], limits=spec['limits'])
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in manifest.items() if k not in ('object_pairs', 'files')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', type=Path, default=Path('data/lessons-relations-v1.json'))
    parser.add_argument('--out', type=Path, default=Path('data/prepared/relations-v1'))
    parser.add_argument('--reading', type=Path, help='Explicitly selected prior training corpus for replay profiles')
    args = parser.parse_args()
    prepare(args.spec, args.out, args.reading)
