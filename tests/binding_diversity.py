"""Independent labels, complete split grids and frozen probes for selected lessons."""
import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

from probes_cli import Reference
sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
from prepare_binding import examples as original_examples, prepare as prepare_original
from prepare_binding_diversity import examples, prepare


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def facts_in(text):
    facts = re.findall(r'The (\w+) is in the (\w+)\.', text)
    return facts or [(obj, loc) for loc, obj in re.findall(r'In the (\w+) is the (\w+)\.', text)]


def check(exe, out):
    exe = exe.resolve()
    out.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).parents[1]
    source = root/'data/lessons-binding-diversity-v1.json'
    original = root/'data/prepared/binding-v2'
    spec, base, prereq, training, rows, monitor, _ = examples(source)
    objects = set(base['objects'] + spec['added_objects'])
    locations = set(base['locations'])
    expected_splits = {name: {frozenset(pair) for pair in base[name+'_pairs']}
                       for name in ('development', 'test')}
    expected_splits['train'] = {frozenset(pair) for pair in itertools.combinations(objects, 2)} - (
        expected_splits['development'] | expected_splits['test'])
    assert len(training) == 51840 and len(prereq) == 384
    assert len(expected_splits['train']) == 270
    checked_groups = 0
    for split, items in rows.items():
        grouped = defaultdict(list)
        coverage, pairs = set(), Counter()
        for item in items:
            facts = facts_in(item['context'])
            query = re.fullmatch(r'(?:Where is the (\w+)\?|Find the (\w+)\.)\nAnswer: ', item['query'])
            assert len(facts) == 2 and query
            queried = query[1] or query[2]
            names = frozenset(obj for obj, _ in facts)
            assert names in expected_splits[split] and queried in names
            assert len(set(loc for _, loc in facts)) == 2
            assert set(loc for _, loc in facts) <= locations
            assert item[f"choice{item['correct']}"] == dict(facts)[queried] + '.'
            assert {item['choice0'], item['choice1']} == {loc+'.' for _, loc in facts}
            key = (tuple(facts), queried, item['context'].startswith('The '), item['query'].startswith('Find '))
            assert key not in coverage
            coverage.add(key)
            pairs[names] += 1
            grouped[item['pair']].append((item, facts, queried))
            if split == 'train':
                assert item['context'] + item['query'] + item[f"choice{item['correct']}"] + '\n' in training
        assert set(pairs) == expected_splits[split] and set(pairs.values()) == {192}
        for group in grouped.values():
            assert len(group) == 4
            assert len({row['context'] for row, _, _ in group}) == 2
            assert len({queried for _, _, queried in group}) == 2
            assert {sum(row['correct'] == answer for row, _, _ in group) for answer in (0, 1)} == {2}
            assert sum(facts[0][0] == queried for _, facts, queried in group) == 2
            assert sum(facts[-1][0] == queried for _, facts, queried in group) == 2
            checked_groups += 1
    assert checked_groups == 13248
    for doc in prereq:
        facts = facts_in(doc)
        assert len(facts) == 1 and facts[0][0] in objects and facts[0][1] in locations
        queried = re.search(r'(?:Where is the|Find the) (\w+)', doc)[1]
        assert queried == facts[0][0] and doc.endswith('\nAnswer: '+facts[0][1]+'.\n')
    train_by_id = {row['id']: row for row in rows['train']}
    monitored = Counter(row['pair'] for row in monitor)
    assert len(monitor) == 1728 and len(monitored) == 432 and set(monitored.values()) == {4}
    assert all(row == train_by_id[row['id']] for row in monitor)

    # Re-rendering the existing selected edition must not change any prior file.
    prepare_original(root/'data/lessons-binding-v2.json', out/'original', original/'reading.dat')
    old_manifest = json.loads((original/'manifest.json').read_text())
    assert all(sha(out/'original'/name) == record['sha256'] for name, record in old_manifest['files'].items())
    first = prepare(source, original, out/'prepared-a')
    prepare(source, original, out/'prepared-b')
    assert all(path.read_bytes() == (out/'prepared-b'/path.name).read_bytes() for path in (out/'prepared-a').iterdir())
    for split in ('train', 'development', 'test'):
        assert (out/'prepared-a'/f'{split}.sgprobe').read_bytes() == (original/f'{split}.sgprobe').read_bytes()

    failures = []
    def reject(name, action, target):
        try:
            action()
        except ValueError:
            assert not target.exists(), name
            failures.append(name)
        else:
            raise AssertionError('Accepted invalid input: '+name)

    shutil.copyfile(root/'data/lessons-binding-v2.json', out/'lessons-binding-v2.json')
    for name, change in [('duplicate-object', lambda s: s['added_objects'].__setitem__(0, 'key')),
                         ('location-as-object', lambda s: s['added_objects'].__setitem__(0, 'box')),
                         ('changed-base-hash', lambda s: s.update(base_spec_sha256='0'*64)),
                         ('changed-monitor-size', lambda s: s.update(monitor_groups=1))]:
        altered = json.loads(source.read_text())
        change(altered)
        path = out/f'{name}.json'
        path.write_text(json.dumps(altered))
        target = out/f'{name}-output'
        reject(name, lambda: prepare(path, original, target), target)
    # A held-out context hidden inside a template must still be caught after the
    # renderer extraction; context-set equality alone would miss this case.
    altered = dict(base)
    altered['queries'] = [rows['development'][0]['context'] + q for q in base['queries']]
    reject('heldout-context-in-template', lambda: original_examples(altered), out/'no-template-output')
    contaminated = out/'contaminated-base'
    shutil.copytree(original, contaminated)
    reading = contaminated/'reading.dat'
    reading.write_bytes(reading.read_bytes() + b'\x1e' + rows['development'][0]['context'].encode())
    manifest = json.loads((contaminated/'manifest.json').read_text())
    manifest['files']['reading.dat'] = dict(bytes=reading.stat().st_size, sha256=sha(reading))
    (contaminated/'manifest.json').write_text(json.dumps(manifest))
    reject('heldout-context-in-reading', lambda: prepare(source, contaminated, out/'bad-reading'), out/'bad-reading')

    # Only a disposable native test model sees the expanded monitor here.
    checkpoint = root/'build/selective-test-results/resume.ckpt'
    before = sha(checkpoint)
    result_path = out/'native-monitor.json'
    result = subprocess.run([str(exe), 'language-probes', '--checkpoint', str(checkpoint),
                             '--probes', str(out/'prepared-a/expanded-train-monitor.sgprobe'),
                             '--output', str(result_path)], capture_output=True)
    (out/'native-monitor.log').write_bytes(result.stdout+result.stderr)
    assert result.returncode == 0, result.stderr
    actual = json.loads(result_path.read_text())
    assert actual['groups'] == 432 and actual['group_size'] == 4
    assert actual['context_erased_joint_accuracy'] == 0 and sha(checkpoint) == before
    ref = Reference(checkpoint)
    max_error = 0
    for row, value in zip(monitor[:8], actual['results'][:8]):
        scores = [ref.score(row['context']+row['query'], row[f'choice{i}']) for i in range(2)]
        max_error = max(max_error, *(abs(a-b) for a, b in zip(scores, value['candidate_nll'])))
        assert ref.greedy(row['context']+row['query'], 4) == value['greedy']
    assert max_error < 3e-5
    report = dict(passed=True, executable_sha256=sha(exe), spec_sha256=sha(source),
                  independent_fact_query_items=sum(map(len, rows.values())), complete_groups=checked_groups,
                  training_documents=len(training), prerequisites=len(prereq),
                  expanded_monitor_items=len(monitor), expanded_monitor_groups=len(monitored),
                  monitor_full_grid_and_training_membership=True, original_prepared_files_byte_identical=True,
                  expanded_preparation_reproduces=True, original_probes_byte_identical=True,
                  invalid_inputs_rejected=failures, native_commands=1,
                  disposable_checkpoint_unchanged=True, cpu_reference_items=8, oracle_max_score_error=max_error,
                  reserved_test_labels_checked_but_no_model_evaluated_on_test=True,
                  prepared_manifest=first)
    (out/'result.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'prepared_manifest'}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--exe', type=Path, default=Path('build/synapticgenesis.exe'))
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    check(a.exe, a.out)
