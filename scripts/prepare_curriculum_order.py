"""Prepare ordered and shuffled instances of exactly the same selected lessons.

Intentional document repetition expresses the observation schedule. The generic
unique-material extension preparer retains its stricter duplicate policy.
"""
import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
import shutil

from native_experiment import read, sha, verified_manifest, write
from prepare_binding_diversity import examples


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def multiset_identity(docs):
    counts = Counter(docs)
    record = '\n'.join(f'{digest(doc)} {count}' for doc, count in sorted(counts.items())).encode('ascii')
    return digest(record)


@dataclass
class LessonOrder:
    policy: dict
    selected_path: Path
    arms: dict
    phases: list
    selected_docs: set
    prerequisites: set
    heldout_contexts: set


def sequences(policy_path, smoke=False):
    policy = read(policy_path)
    selected_path = policy_path.parent/policy['selected_spec']
    if (policy['version'] != 'curriculum-order-v1' or
            sha(selected_path) != policy['selected_spec_sha256'] or policy['object_counts'] != [6, 12, 24] or
            policy['phase_observations'] != [56640, 11520, 51840]):
        raise ValueError('Order experiment requires its exact declared selection and schedule')
    selected, base, prerequisites, training, rows, _, _ = examples(selected_path)
    objects = base['objects']+selected['added_objects']
    groups = defaultdict(list)
    for row in rows['train']:
        groups[row['pair']].append((row['context']+row['query']+row[f"choice{row['correct']}"]+'\n').encode('ascii'))
    counts = [32, 16, 16] if smoke else policy['phase_observations']
    phases, records = [], []
    for index, (width, count) in enumerate(zip(policy['object_counts'], counts)):
        allowed = set(objects[:width])
        candidates = sorted(group for group in groups if set(group.split('-')[:2]) <= allowed)
        rng = random.Random(policy['phase_seed']+index)
        remaining, picked = count//4, []
        while remaining:
            cycle = list(candidates)
            rng.shuffle(cycle)
            take = min(remaining, len(cycle))
            picked.extend(cycle[:take])
            remaining -= take
        docs = [doc for group in picked for doc in groups[group]]
        rng.shuffle(docs)
        assert len(docs) == count and all(0 < len(doc)-1 <= 128 for doc in docs)
        phases.append(docs)
        records.append(dict(objects=objects[:width], observations=count,
                            available_groups=len(candidates), unique_documents=len(set(docs)),
                            observed_pairs=sum(len(doc)-1 for doc in docs)))
    ordered = [doc for phase in phases for doc in phase]
    shuffled = list(ordered)
    random.Random(policy['shuffle_seed']).shuffle(shuffled)
    controls, offset = [], 0
    for count in counts:
        controls.append(shuffled[offset:offset+count])
        offset += count
    assert Counter(ordered) == Counter(shuffled)
    selected_docs = {doc.encode('ascii') for doc in training}
    assert set(ordered) <= selected_docs
    if not smoke:
        assert set(ordered) == selected_docs and set(phases[-1]) == selected_docs
    return LessonOrder(policy, selected_path, dict(ramped=phases, shuffled=controls), records, selected_docs,
                       {doc.encode('ascii') for doc in prerequisites},
                       {row['context'].encode('ascii') for split in ('development', 'test') for row in rows[split]})


def prepare(policy_path, selected, out, smoke=False):
    policy_path, selected, out = Path(policy_path), Path(selected), Path(out)
    plan = sequences(policy_path, smoke)
    policy, selected_path, arms = plan.policy, plan.selected_path, plan.arms
    manifest = verified_manifest(selected)
    if manifest['spec_sha256'] != policy['selected_spec_sha256']:
        raise ValueError('Prepared lessons differ from the declared selection')
    if set((selected/'binding.dat').read_bytes().split(b'\x1e')) != plan.selected_docs:
        raise ValueError('Prepared lessons differ from the selected canonical documents')
    reading, prerequisite = (selected/'reading.dat').read_bytes(), (selected/'prerequisites.dat').read_bytes()
    if set(prerequisite.split(b'\x1e')) != plan.prerequisites:
        raise ValueError('Prerequisites differ from the selected canonical documents')
    normalized = reading.replace(b'\r\n', b'\n').replace(b'\r', b'\n')
    if any(context in normalized for context in plan.heldout_contexts):
        raise ValueError('Reading contains a held-out binding context')
    prefix = reading+b'\x1e'+prerequisite
    if prefix != (selected/'through-prerequisites.dat').read_bytes():
        raise ValueError('Common prerequisite prefix changed')
    reading_end, prerequisite_end = (16, 32) if smoke else (6000, 10000)
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(policy_path, out/'source-policy.json')
    shutil.copyfile(selected_path, out/'selected-spec.json')
    (out/'reading.dat').write_bytes(reading)
    (out/'through-prerequisites.dat').write_bytes(prefix)
    schedule = f'SGCURRICULUM3\n{reading_end} "reading.dat" 1 all 1\n{prerequisite_end} "through-prerequisites.dat" 0.25 new 64\n'
    (out/'common.sg').write_bytes(schedule.encode())
    records = {}
    for arm, additions in arms.items():
        text, cumulative, end, stages = schedule, prefix, prerequisite_end, []
        for index, docs in enumerate(additions, 1):
            added = b'\x1e'.join(docs)
            (out/f'{arm}-observations-{index}.dat').write_bytes(added)
            cumulative += b'\x1e'+added
            name = f'{arm}-through-{index}.dat'
            (out/name).write_bytes(cumulative)
            end += len(docs)
            text += f'{end} "{name}" 0.25 new 64\n'
            stages.append(dict(end_update=end, observations=len(docs), unique_documents=len(set(docs)),
                               observed_pairs=sum(len(doc)-1 for doc in docs),
                               cumulative_documents=len(cumulative.split(b'\x1e'))))
        (out/f'{arm}.sg').write_bytes(text.encode())
        flat = [doc for phase in additions for doc in phase]
        records[arm] = dict(stages=stages, multiset_sha256=multiset_identity(flat),
                            observed_pairs=sum(len(doc)-1 for doc in flat),
                            unique_documents=len(set(flat)), frequency_distribution=dict(sorted(Counter(Counter(flat).values()).items())))
    for name in ('train.sgprobe', 'development.sgprobe', 'test.sgprobe', 'expanded-train-monitor.sgprobe'):
        shutil.copyfile(selected/name, out/name)
    result = dict(version=policy['version'], purpose='smoke' if smoke else 'learning_comparison',
                  policy_sha256=sha(policy_path), selected_spec_sha256=sha(selected_path),
                  reading_end=reading_end, prerequisite_end=prerequisite_end,
                  ramped_phases=plan.phases, arms=records, matched_observation_multiset=True,
                  full_selected_binding_coverage=not smoke,
                  files={p.name: dict(bytes=p.stat().st_size, sha256=sha(p)) for p in sorted(out.iterdir())},
                  limits=policy['limits'])
    write(out/'manifest.json', result)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--policy', type=Path, default=Path('data/curriculum-order-v1.json'))
    p.add_argument('--selected', type=Path, default=Path('data/prepared/binding-diversity-v1'))
    p.add_argument('--out', type=Path, default=Path('data/prepared/curriculum-order-v1'))
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    report = prepare(a.policy, a.selected, a.out, a.smoke)
    print(json.dumps({k:v for k,v in report.items() if k not in ('files', 'ramped_phases')}, indent=2))
