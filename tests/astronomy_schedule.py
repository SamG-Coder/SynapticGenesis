"""Audit the real append-only edition and reject altered/held-out inputs."""
import argparse
from pathlib import Path
import shlex
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from astronomy_schedule import prepare, protect_training
from native_experiment import read, sha
from prose_founder import write


def audit(edition, out, report):
    out.mkdir(parents=True, exist_ok=False)
    plan = read(edition / 'protocol.json')
    assert plan['no_native_model_work'] and not plan['checkpoint_selected']
    assert all(sha(p) == h for p, h in plan['authenticated_inputs'].items())
    schedule = Path(plan['curriculum'])
    assert sha(schedule) == plan['curriculum_sha256']
    rows = [shlex.split(line) for line in schedule.read_text().splitlines()[1:]]
    assert len(rows) == 8
    original = Path(plan['parent_schedule'])
    assert sha(original) == plan['parent_schedule_sha256']
    previous = read(original.parent / 'protocol.json')
    for fields, stage in zip(rows[:4], previous['stages']):
        assert int(fields[0]) == stage['end_update']
        assert sha(schedule.parent / fields[1]) == stage['source_sha256']
        assert fields[2:] == ['1', 'new', '1']
    prepared = Path('data/prepared/astronomy-v1-reviewed')
    spec = read('data/sources-astronomy-v1.json')
    combined = Path(previous['stages'][-1]['cumulative_source']).read_bytes()
    expected_end, all_pairs, all_windows = previous['online_updates'], 0, 0
    selected_ids = []
    for index, fields in enumerate(rows[4:], 1):
        ids = [r['id'] for r in spec['sources'] if r['split'] == 'train' and r['stage'] == index]
        selected_ids.extend(ids)
        docs = [(prepared / f'{ident}.txt').read_bytes() for ident in ids]
        combined += b'\x1e' + b'\x1e'.join(docs)
        assert combined == (schedule.parent / fields[1]).read_bytes()
        observed, pairs = 0, 0
        for doc in docs:
            cursor = 0
            for start in range(0, len(doc) - 1, 128):
                assert start == cursor
                end = min(start + 128, len(doc) - 1)
                pairs += end - start
                observed += 1
                cursor = end
            assert cursor == len(doc) - 1
        expected_end += observed
        assert int(fields[0]) == expected_end and fields[2:] == ['1', 'new', '1']
        recorded = plan['stages'][index - 1]
        assert (recorded['observations'], recorded['source_pairs']) == (observed, pairs)
        all_pairs += pairs
        all_windows += observed
    assert len(selected_ids) == len(set(selected_ids)) == 174
    assert set(selected_ids) == {r['id'] for r in spec['sources'] if r['split'] == 'train'}
    assert (all_windows, all_pairs, expected_end) == (22227, 2832810, 238516)
    assert (plan['added_observations'], plan['added_source_pairs'], plan['end_update']) == (all_windows, all_pairs, expected_end)
    for name in ('LICENSE-source.txt', 'ATTRIBUTION.txt', 'original-preface.cnxml'):
        assert (schedule.parent / name).read_bytes() == (prepared / name).read_bytes()
    # Check both corpora's protected documents against the actual final payload.
    holdouts = [folder / f'{split}.dat' for folder in
                (Path('data/prepared/prose-scale-v1-pinned'), prepared) for split in ('validation', 'test')]
    protect_training([combined], [p.read_bytes() for p in holdouts])
    rejected = []
    for mutation in ('document', 'stage'):
        copy = out / f'changed-{mutation}'
        shutil.copytree(prepared, copy)
        changed = copy / (f'{selected_ids[0]}.txt' if mutation == 'document' else 'stage-1.dat')
        changed.write_bytes(changed.read_bytes() + b'altered')
        destination = out / f'rejected-{mutation}'
        try:
            prepare(original, copy, destination)
        except ValueError as error:
            assert ('document changed' if mutation == 'document' else 'stage changed') in str(error)
            assert not destination.exists()
            rejected.append(mutation)
        else:
            raise AssertionError('Changed prepared astronomy was accepted')
    long = b'A held-out paragraph explains how a spectrum identifies material in a star. ' * 3
    cases = [([b'held-out'], [b'held-out'], 'document'),
             ([b'New heading\n\n' + long + b'\n\nNew ending'], [long], 'paragraph'),
             ([b'New heading\n\n' + long.replace(b' ', b'   ')], [long], 'paragraph')]
    for training, protected, kind in cases:
        try:
            protect_training(training, protected)
        except ValueError as error:
            assert kind in str(error)
            rejected.append('held-out-' + kind)
        else:
            raise AssertionError('Protected input was accepted')
    result = dict(passed=True, protocol=plan, protocol_sha256=sha(edition / 'protocol.json'),
        script_sha256=sha(Path(__file__)), all_four_previous_stages_preserved=True,
        all_174_training_modules_once=True, complete_target_coverage=True,
        four_new_stages_verified=True, exact_heldout_document_or_paragraph_overlap=0,
        rejected_inputs=rejected, source_attribution_byte_identical=True,
        native_commands=0, native_admission_verified=False,
        model_learning_verified=False, reproduction_admitted=False)
    write(report, result)
    print('Eight stages, 174 new modules, all target bytes, source attribution and five rejections passed.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--edition', type=Path, default=Path('runs/prose-astronomy-curriculum'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    audit(args.edition, args.out, args.report)
