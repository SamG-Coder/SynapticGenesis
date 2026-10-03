"""Validate actual prepared lesson order, matching exposure and source isolation."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import shutil
import sys

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
from extend_curriculum import read_schedule
from native_experiment import read, sha, write
from prepare_curriculum_order import prepare


def parse(doc):
    text = doc.decode('ascii')
    context, rest = text.split('\n', 1)
    query, answer = rest.split('\nAnswer: ')
    facts = re.findall(r'The (\w+) is in the (\w+)\.', context)
    if not facts:
        facts = [(obj, loc) for loc, obj in re.findall(r'In the (\w+) is the (\w+)\.', context)]
    question = re.fullmatch(r'(?:Where is the (\w+)\?|Find the (\w+)\.)', query)
    assert len(facts) == 2 and len(dict(facts)) == 2 and question and len(doc)-1 <= 128
    target = question[1] or question[2]
    assert answer == dict(facts)[target]+'.\n'
    group = (tuple(obj for obj, _ in facts), frozenset(loc for _, loc in facts),
             context.startswith('The '), query.startswith('Find '))
    state = (tuple(loc for _, loc in facts), target)
    return facts, group, state


def check(out):
    root = Path(__file__).parents[1]
    selected = root/'data/prepared/binding-diversity-v1'
    policy = root/'data/curriculum-order-v1.json'
    out.mkdir(parents=True, exist_ok=False)
    a, b = out/'prepared-a', out/'prepared-b'
    manifest = prepare(policy, selected, a)
    prepare(policy, selected, b)
    assert all(path.read_bytes() == (b/path.name).read_bytes() for path in a.iterdir())
    base = read(root/'data/lessons-binding-v2.json')
    expansion = read(root/'data/lessons-binding-diversity-v1.json')
    objects = base['objects']+expansion['added_objects']
    heldout = {frozenset(pair) for name in ('development_pairs', 'test_pairs') for pair in base[name]}
    source = set((selected/'binding.dat').read_bytes().split(b'\x1e'))
    parsed = {}
    for doc in source:
        facts, group, state = parse(doc)
        assert frozenset(obj for obj, _ in facts) not in heldout
        assert set(obj for obj, _ in facts) <= set(objects)
        assert len(set(loc for _, loc in facts)) == 2 and set(loc for _, loc in facts) <= set(base['locations'])
        parsed[doc] = facts, group, state
    actual = {}
    for arm in ('ramped', 'shuffled'):
        _, stages = read_schedule(a/f'{arm}.sg')
        assert [s['end_update'] for s in stages] == [6000, 10000, 66640, 78160, 130000]
        assert all(s['scope'] == 'new' and s['answer'] == '64' for s in stages[1:])
        all_docs, previous = [], stages[1]['content']
        for index, (stage, count, width) in enumerate(zip(stages[2:], [56640, 11520, 51840], [6, 12, 24]), 1):
            docs = (a/f'{arm}-observations-{index}.dat').read_bytes().split(b'\x1e')
            assert len(docs) == count and set(docs) <= source
            assert stage['content'] == previous+b'\x1e'+b'\x1e'.join(docs)
            previous = stage['content']
            if arm == 'ramped':
                grid = defaultdict(Counter)
                for doc in docs:
                    facts, group, state = parsed[doc]
                    assert set(obj for obj, _ in facts) <= set(objects[:width])
                    grid[group][state] += 1
                assert all(len(cells) == 4 and len(set(cells.values())) == 1 for cells in grid.values())
                assert len(set(docs)) == [1728, 11520, 51840][index-1]
            all_docs += docs
        actual[arm] = Counter(all_docs)
        assert sum(actual[arm].values()) == 120000 and set(actual[arm]) == source
    assert actual['ramped'] == actual['shuffled']
    assert Counter(actual['ramped'].values()) == {1: 40320, 2: 9792, 34: 384, 35: 1344}
    for name in ('train.sgprobe', 'development.sgprobe', 'test.sgprobe', 'expanded-train-monitor.sgprobe'):
        assert (a/name).read_bytes() == (selected/name).read_bytes()
    assert (a/'through-prerequisites.dat').read_bytes() == (selected/'through-prerequisites.dat').read_bytes()

    rejected = []
    def reject(name, action, target):
        try:
            action()
        except ValueError:
            assert not target.exists()
            rejected.append(name)
        else:
            raise AssertionError('Accepted invalid preparation: '+name)
    for filename in ('lessons-binding-diversity-v1.json', 'lessons-binding-v2.json'):
        shutil.copyfile(root/'data'/filename, out/filename)
    for name, change in [('changed-selection-hash', lambda p: p.update(selected_spec_sha256='0'*64)),
                         ('reversed-widths', lambda p: p.update(object_counts=[24, 12, 6])),
                         ('changed-budget', lambda p: p.update(phase_observations=[60000, 12000, 48000]))]:
        invalid = read(policy)
        change(invalid)
        path = out/f'{name}.json'
        write(path, invalid)
        target = out/f'{name}-output'
        reject(name, lambda: prepare(path, selected, target), target)
    modified = out/'modified-selected'
    shutil.copytree(selected, modified)
    heldout_doc = b'The key is in the box. The hat is in the bag.\nWhere is the key?\nAnswer: box.\n'
    changed_manifest = read(modified/'manifest.json')
    binding = modified/'binding.dat'
    binding.write_bytes(binding.read_bytes()+b'\x1e'+heldout_doc)
    changed_manifest['files']['binding.dat'] = dict(bytes=binding.stat().st_size, sha256=sha(binding))
    write(modified/'manifest.json', changed_manifest)
    reject('unselected-binding-document', lambda: prepare(policy, modified, out/'bad-binding'), out/'bad-binding')
    # Restore binding, then make all edited byte identities internally consistent
    # so the semantic held-out-context check, not a stale hash, must reject it.
    shutil.copyfile(selected/'binding.dat', binding)
    changed_manifest['files']['binding.dat'] = read(selected/'manifest.json')['files']['binding.dat']
    reading = modified/'reading.dat'
    reading.write_bytes(reading.read_bytes()+b'\x1e'+heldout_doc)
    combined = modified/'through-prerequisites.dat'
    combined.write_bytes(reading.read_bytes()+b'\x1e'+(modified/'prerequisites.dat').read_bytes())
    for path in (reading, combined):
        changed_manifest['files'][path.name] = dict(bytes=path.stat().st_size, sha256=sha(path))
    write(modified/'manifest.json', changed_manifest)
    reject('heldout-context-in-reading', lambda: prepare(policy, modified, out/'bad-reading'), out/'bad-reading')
    result = dict(passed=True, unique_labels_independently_checked=len(parsed),
                  instances_per_arm=120000, exact_document_multiset_matches=True,
                  all_selected_binding_documents_seen_in_both_arms=True,
                  observed_binding_pairs=8879952, full_reversal_groups_preserved_in_each_ramped_phase=True,
                  frozen_probes_and_common_prefix_identical=True, preparation_reproduces=True,
                  invalid_inputs_rejected=rejected, prepared_manifest=manifest,
                  no_model_evaluated_on_reserved_tests=True)
    write(out/'result.json', result)
    print(json.dumps({k:v for k,v in result.items() if k != 'prepared_manifest'}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    check(p.parse_args().out)
