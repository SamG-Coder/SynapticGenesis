"""CPU preflight: original native arguments, initial-state guards and gating."""
import argparse
import copy
from pathlib import Path
import shutil
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from experiment_checkpoint import policy_checkpoint
from native_experiment import read
from prose_evaluation import assess
from prose_founder import file_hash, live_arguments, write
from prose_replay_capacity import ROOT, comparison_row, declare, execute, initial_identity
from prose_size_comparison import counters


def audit(out, report):
    out.mkdir(parents=True, exist_ok=False)
    commands = []
    for profile, directory in [('2m', Path('runs/prose-size-panel/founder-2m')),
                               ('105m', Path('runs/prose-105m-founder'))]:
        old = read(directory / 'commands/commands.json')[0][1:]
        updates = int(old[old.index('--updates') + 1])
        arguments = list(map(str, live_arguments(directory, Path('runs/prose-scale-curriculum/curriculum.sg'), updates, profile)))
        assert arguments == old
        changed = list(map(str, live_arguments(directory, Path('runs/prose-scale-curriculum/curriculum.sg'), updates, profile, 16384)))
        differences = [i for i, (a, b) in enumerate(zip(old, changed)) if a != b]
        assert len(old) == len(changed) and differences == [old.index('--replay-capacity') + 1]
        for stage in (1, 2, 3, 4):
            recorded = next(r for r in read('runs/prose-size-panel/partial.json')
                            if r['profile'] == profile and r['stage'] == stage)
            assert counters(directory / f'stage-{stage}.ckpt') == recorded['exposure_counters']
        commands.append(dict(profile=profile, original_command_journal_sha256=file_hash(directory / 'commands/commands.json'),
                             default_arguments_exact=True, only_candidate_argument_change='--replay-capacity'))
    rejected = []

    def rejects(label, function, contains):
        try:
            function()
        except ValueError as error:
            assert contains in str(error), str(error)
            rejected.append(label)
        else:
            raise AssertionError('Invalid case accepted: ' + label)

    for capacity in (0, 3, 65537):
        rejects(f'capacity {capacity}', lambda: live_arguments(None, None, 216289, '2m', capacity), 'capacity')
    checkpoint = Path('runs/prose-size-panel/founder-2m/initial.ckpt')
    identity = initial_identity(checkpoint, checkpoint, 1024, 1024)
    assert identity['only_capacity_and_payload_checksum_differ']
    rejects('wrong initial capacity', lambda: initial_identity(checkpoint, checkpoint, 1024, 16384), 'capacity differs')
    changed = out / 'changed-initial.ckpt'
    shutil.copyfile(checkpoint, changed)
    with changed.open('r+b') as stream:
        stream.seek(288)
        value = stream.read(1)
        stream.seek(288)
        stream.write(bytes([value[0] ^ 1]))
    rejects('changed initial weight byte', lambda: initial_identity(checkpoint, changed, 1024, 1024), 'Initial parameters')
    row = read('runs/prose-size-panel/partial.json')[0]
    same = comparison_row(row, row)
    assert same['validation_difference_from_1024'] == same['replay_pair_difference'] == 0
    modified = copy.deepcopy(row)
    modified['exposure_counters']['replay_pairs'] += 7
    assert comparison_row(modified, row)['replay_pair_difference'] == 7
    modified['exposure_counters']['replay_updates'] += 1
    rejects('changed replay update budget', lambda: comparison_row(modified, row), 'exposure differs')
    # More than 16384 extension words must be accepted by the diagnostic view.
    # This synthetic policy-only file is not submitted to a native model loader.
    meta, extra = policy_checkpoint(checkpoint)
    import struct
    header = list(meta)
    header[14], header[18], header[31] = 1, 0, 17 + 5 * 4 + 3 * 16384
    payload = [0] * header[31]
    payload[1], payload[3] = 3, 16384
    fixture = out / 'large-policy-reader-fixture.ckpt'
    fixture.write_bytes(struct.pack('<32Q', *header) + bytes(44) + struct.pack(f'<{len(payload)}Q', *payload))
    actual, words = policy_checkpoint(fixture)
    assert actual[31] == 49189 and words[3] == 16384
    study = out / 'study'
    declare(study)
    replayed_assessments = []
    for profile, directory in [('2m', Path('runs/prose-size-panel/founder-2m')),
                               ('105m', Path('runs/prose-105m-founder'))]:
        archive = Path(f'runs/prose-size-panel/{profile}-stage-4')
        archived_commands = read(archive / 'commands.json')
        call_index = 0

        def replay(*args):
            nonlocal call_index
            actual = list(map(str, args))
            expected = archived_commands[call_index][1:]
            assert actual[:-1] == expected[:-1] and actual[-2] == '--output'
            shutil.copyfile(expected[-1], actual[-1])
            call_index += 1

        destination = out / f'assess-{profile}'
        with patch('prose_evaluation.NativeCommands', return_value=replay):
            assess(study / 'assessment/protocol.json', directory, 4, destination)
        current, previous = read(destination / 'result.json'), read(archive / 'result.json')
        assert call_index == 10
        assert all(current[k] == previous[k] for k in ('books', 'samples', 'validation_mean_nats_per_byte'))
        replayed_assessments.append(dict(profile=profile, archived_native_commands=10,
                                        full_assessment_authentication=True, book_and_sample_fields_exact=True))
    with patch('prose_replay_capacity.predecessors', side_effect=ValueError('incomplete test predecessor')):
        with patch('prose_replay_capacity.train_founder') as native:
            rejects('learning before predecessors finish', lambda: execute(study, 0, None), 'incomplete test predecessor')
            native.assert_not_called()
    failure = read(study / 'failure.json')
    assert failure['completed_assessments'] == 0 and failure['partial_artifacts_preserved']
    result = dict(passed=True, original_commands=commands, eight_counter_records_preserved=True,
        assessment_replay=replayed_assessments,
        initial_guard_reference_sha256=identity['original_checkpoint_sha256'],
        changed_weight_rejected=True, large_policy_diagnostic_words=49189,
        synthetic_policy_fixture_native_validation=False, observed_replay_pair_difference_reported=True,
        rejected=rejected, native_commands=0, candidate_learning_verified=False,
        code_sha256={p.relative_to(ROOT).as_posix(): file_hash(p) for p in [Path(__file__),
            ROOT / 'scripts/prose_founder.py', ROOT / 'scripts/prose_evaluation.py',
            ROOT / 'scripts/prose_size_comparison.py', ROOT / 'scripts/prose_replay_capacity.py']},
        limits='CPU preflight only. Larger-capacity CUDA learning, initial-array agreement with the original '
               'founders and quality changes remain required native-run checks. A synthetic oversized policy '
               'tests only the bounded reader, not native checkpoint admission.')
    write(report, result)
    print('PASS: original commands, eight counter records, larger-policy reader and seven negative cases; no CUDA work.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    audit(args.out, args.report)
