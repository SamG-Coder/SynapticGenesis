"""Selected prerequisite and binding lessons; data preparation only."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import random

from prepare_lessons import write_probes
from binding_lessons import render


def examples(spec):
    objects, locations = spec['objects'], spec['locations']
    if spec['version'] != 'binding-v2' or len(set(objects)) != 6 or len(set(locations)) != 4:
        raise ValueError('Invalid selected binding vocabulary')
    if not all(word.isascii() and word.isalpha() and len(word) == 3 for word in objects + locations):
        raise ValueError('Expected selected three-letter words')
    partitions = {name: {frozenset(p) for p in spec[name + '_pairs']} for name in ('development', 'test')}
    universe = {frozenset(p) for p in itertools.combinations(objects, 2)}
    if any(len(p) != 3 or not p <= universe for p in partitions.values()) or partitions['development'] & partitions['test']:
        raise ValueError('Invalid unordered object split')
    partitions['train'] = universe - partitions['development'] - partitions['test']
    return render(objects, locations, spec['facts'], spec['queries'], spec['answer'], partitions)


def prepare(spec_path, out, reading=None):
    raw = spec_path.read_bytes()
    spec = json.loads(raw)
    prerequisites, training, rows, partitions = examples(spec)
    original = reading.read_bytes() if reading is not None else None
    if original is not None:
        if not original or original.endswith(b'\x1e'):
            raise ValueError('Invalid prior reading corpus')
        normalized = original.replace(b'\r\n', b'\n').replace(b'\r', b'\n')
        if any(row['context'].encode('ascii') in normalized for name in ('development', 'test') for row in rows[name]):
            raise ValueError('Prior reading contains a held-out binding context')
    out.mkdir(parents=True, exist_ok=False)
    (out / 'source-spec.json').write_bytes(raw)
    for name, docs in [('prerequisites', prerequisites), ('binding', training)]:
        ordered = sorted(docs)
        random.Random(381970).shuffle(ordered)
        (out / f'{name}.dat').write_bytes(b'\x1e'.join(x.encode('ascii') for x in ordered))
    for name, records in rows.items():
        write_probes(out / f'{name}.sgprobe', records, 'SGPROBE2')
    if original is not None:
        (out / 'reading.dat').write_bytes(original)
        second = original + b'\x1e' + (out / 'prerequisites.dat').read_bytes()
        (out / 'through-prerequisites.dat').write_bytes(second)
        (out / 'through-binding.dat').write_bytes(second + b'\x1e' + (out / 'binding.dat').read_bytes())
        (out / 'curriculum.sg').write_bytes(b'SGCURRICULUM3\n6000 "reading.dat" 1 all 1\n10000 "through-prerequisites.dat" 0.25 new 64\n34000 "through-binding.dat" 0.25 new 64\n')
    manifest = dict(version=spec['version'], spec_sha256=hashlib.sha256(raw).hexdigest(),
                    files={p.name: dict(bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                           for p in sorted(out.iterdir())},
                    documents=dict(prerequisites=len(prerequisites), binding=len(training)),
                    probe_items={name:len(items) for name,items in rows.items()},
                    object_pairs={name:sorted(sorted(pair) for pair in pairs) for name,pairs in partitions.items()},
                    heldout_contexts_absent_from_training=True, joint_group_size=4,
                    reading_source=str(reading) if reading is not None else None,
                    limits=spec['limits'])
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in manifest.items() if k not in ('files','object_pairs')},indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--spec', type=Path, default=Path('data/lessons-binding-v2.json'))
    p.add_argument('--out', type=Path, default=Path('data/prepared/binding-v2'))
    p.add_argument('--reading', type=Path)
    args = p.parse_args()
    prepare(args.spec,args.out,args.reading)
