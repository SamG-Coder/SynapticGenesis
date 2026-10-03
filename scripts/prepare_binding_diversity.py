"""Prepare an explicitly selected vocabulary expansion with frozen old probes."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import random
import shutil

from binding_lessons import render
from prepare_binding import examples as original_examples
from prepare_lessons import write_probes


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def examples(spec_path):
    raw = spec_path.read_bytes()
    spec = json.loads(raw)
    base_path = spec_path.parent/spec['base_spec']
    base_raw = base_path.read_bytes()
    if spec['version'] != 'binding-diversity-v1' or digest(base_raw) != spec['base_spec_sha256']:
        raise ValueError('Diversity specification requires its exact selected base edition')
    base = json.loads(base_raw)
    _, _, original_rows, original_splits = original_examples(base)
    objects = base['objects'] + spec['added_objects']
    if (len(objects) != 24 or len(set(objects)) != 24 or set(objects) & set(base['locations']) or
            any(not w.isascii() or not w.isalpha() or not w.islower() or len(w) != 3 for w in objects)):
        raise ValueError('Expected 24 distinct selected three-letter objects, separate from locations')
    partitions = {name:original_splits[name] for name in ('development','test')}
    partitions['train'] = {frozenset(p) for p in itertools.combinations(objects,2)} - partitions['development'] - partitions['test']
    prereq, training, rows, _ = render(objects, base['locations'], base['facts'], base['queries'], base['answer'], partitions)
    for split in ('development','test'):
        if rows[split] != original_rows[split]:
            raise ValueError('Expansion changed a frozen held-out probe')
    groups = {}
    for row in rows['train']:
        groups.setdefault(row['pair'],[]).append(row)
    count = spec['monitor_groups']
    if count != 432:
        raise ValueError('Expected the declared 432-group expanded-training monitor')
    selected = random.Random(spec['monitor_seed']).sample(sorted(groups), count)
    monitor = [row for group in sorted(selected) for row in groups[group]]
    return spec, base, prereq, training, rows, monitor, partitions


def prepare(spec_path, original, out):
    spec_path, original, out = Path(spec_path), Path(original), Path(out)
    spec, base, prereq, training, rows, monitor, partitions = examples(spec_path)
    manifest = json.loads((original/'manifest.json').read_text())
    if manifest['spec_sha256'] != spec['base_spec_sha256']:
        raise ValueError('Prepared base does not match the selected base specification')
    for name, record in manifest['files'].items():
        if digest((original/name).read_bytes()) != record['sha256']:
            raise ValueError(f'Prepared base source changed: {name}')
    reading = (original/'reading.dat').read_bytes()
    normalized = reading.replace(b'\r\n',b'\n').replace(b'\r',b'\n')
    if any(r['context'].encode() in normalized for split in ('development','test') for r in rows[split]):
        raise ValueError('Reading contains a held-out binding context')
    out.mkdir(parents=True,exist_ok=False)
    (out/'source-spec.json').write_bytes(spec_path.read_bytes())
    (out/'base-spec.json').write_bytes((spec_path.parent/spec['base_spec']).read_bytes())
    (out/'reading.dat').write_bytes(reading)
    for name, docs in [('prerequisites',prereq),('binding',training)]:
        ordered = sorted(docs)
        random.Random(spec['shuffle_seed']).shuffle(ordered)
        (out/f'{name}.dat').write_bytes(b'\x1e'.join(d.encode('ascii') for d in ordered))
    for split in ('development','test','train'):
        shutil.copyfile(original/f'{split}.sgprobe',out/f'{split}.sgprobe')
    write_probes(out/'expanded-train-monitor.sgprobe',monitor,'SGPROBE2')
    second = reading + b'\x1e' + (out/'prerequisites.dat').read_bytes()
    (out/'through-prerequisites.dat').write_bytes(second)
    (out/'through-binding.dat').write_bytes(second+b'\x1e'+(out/'binding.dat').read_bytes())
    (out/'curriculum.sg').write_bytes(b'SGCURRICULUM3\n6000 "reading.dat" 1 all 1\n10000 "through-prerequisites.dat" 0.25 new 64\n130000 "through-binding.dat" 0.25 new 64\n')
    result = dict(version=spec['version'], spec_sha256=digest(spec_path.read_bytes()),
                  base_spec_sha256=spec['base_spec_sha256'], objects=base['objects']+spec['added_objects'],
                  documents=dict(prerequisites=len(prereq),binding=len(training)),
                  training_object_pairs=len(partitions['train']), training_groups=len(rows['train'])//4,
                  original_train_groups=432, expanded_monitor_groups=len(monitor)//4,
                  frozen_development_groups=144, frozen_test_groups=144,
                  heldout_pairs_excluded=True, old_probes_byte_identical=True,
                  files={p.name:dict(bytes=p.stat().st_size,sha256=digest(p.read_bytes())) for p in sorted(out.iterdir())},
                  limits=spec['limits'])
    (out/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--spec',type=Path,default=Path('data/lessons-binding-diversity-v1.json'))
    p.add_argument('--original',type=Path,default=Path('data/prepared/binding-v2'))
    p.add_argument('--out',type=Path,default=Path('data/prepared/binding-diversity-v1'))
    a=p.parse_args()
    r=prepare(a.spec,a.original,a.out)
    print(json.dumps({k:v for k,v in r.items() if k!='files'},indent=2))
