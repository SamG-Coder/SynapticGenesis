"""Authenticate completed larger-replay founders and publish exact study bytes."""
import argparse
import hashlib
from pathlib import Path
import struct

from checkpoint_assessment import verify_saved
from experiment_checkpoint import policy_checkpoint
from native_experiment import read
from prose_founder import file_hash, live_arguments, write
from prose_replay_capacity import initial_identity
from prose_retention_inputs import authenticate
from prose_size_comparison import counters, require_complete


def verify_comparisons(rows, baseline):
    """Recompute all paired deltas from authenticated book records."""
    matched = ('online_updates', 'global_updates', 'observed_pairs', 'generated_bytes', 'replay_updates')
    comparisons = []
    for row in rows:
        old = next(r for r in baseline['rows'] if (r['profile'], r['stage']) == (row['profile'], row['stage']))
        assert row['parameters'] == old['parameters'] and row['spiking_neurons'] == old['spiking_neurons']
        assert all(row['exposure_counters'][key] == old['exposure_counters'][key] for key in matched)
        assert [(b['book'], b['role']) for b in row['books']] == [(b['book'], b['role']) for b in old['books']]
        expected = dict(profile=row['profile'], stage=row['stage'], source_observations=row['online_updates'],
            validation_mean_nats_per_byte=row['validation_mean_nats_per_byte'],
            validation_difference_from_1024=row['validation_mean_nats_per_byte'] - old['validation_mean_nats_per_byte'],
            replay_pairs=row['exposure_counters']['replay_pairs'],
            replay_pair_difference=row['exposure_counters']['replay_pairs'] - old['exposure_counters']['replay_pairs'],
            books=[dict(book=current['book'], role=current['role'], loss_nats_per_byte=current['loss_nats_per_byte'],
                difference_from_1024=current['loss_nats_per_byte'] - prior['loss_nats_per_byte'])
                for current, prior in zip(row['books'], old['books'])])
        assert expected == row['comparison'], 'Recorded replay comparison differs from raw values'
        comparisons.append(expected)
    return comparisons


def verify(root):
    root = Path(root)
    plan_path, result_path = root / 'protocol.json', root / 'result.json'
    plan, result = read(plan_path), read(result_path)
    assert result['complete'] and not (root / 'failure.json').exists()
    assert result['protocol'] == plan
    assert all(result[k] for k in ('same_initial_model_arrays', 'same_candidate_replay_history_across_sizes',
                                  'matched_source_update_and_speech_exposure'))
    assert not result['reserved_tests_scored'] and not result['reproduction_admitted']
    authenticate(plan['authenticated_inputs'])
    predecessors = read(root / 'predecessors-complete.json')
    authenticate(predecessors)
    spec = plan['specification']
    assert spec['profiles'] == ['2m', '105m'] and spec['seed'] == 1337
    assert (spec['baseline_capacity'], spec['candidate_capacity'], spec['source_observations']) == (1024, 16384, 216289)
    assessment_path = Path(plan['assessment_plan'])
    assessment = read(assessment_path)
    authenticate(assessment['authenticated_inputs'])
    evaluation = assessment['specification']
    baseline = read(spec['baseline_comparison'])
    assert baseline['complete'] and not baseline['reserved_tests_scored']
    assert [(r['profile'], r['stage']) for r in result['rows']] == [(p, s) for p in spec['profiles'] for s in range(1, 5)]
    assert [r['profile'] for r in result['initial']] == [r['profile'] for r in result['completed']] == spec['profiles']
    hashes = {result_path.as_posix(): file_hash(result_path), plan_path.as_posix(): file_hash(plan_path),
              assessment_path.as_posix(): file_hash(assessment_path)}
    cases, initial, sessions, policies = [], [], [], {}
    for profile, recorded_initial, recorded_completion in zip(spec['profiles'], result['initial'], result['completed']):
        directory = root / f'founder-{profile}'
        original = Path(recorded_initial['original_checkpoint'])
        original_case = next(r for r in baseline['rows'] if r['profile'] == profile and r['stage'] == 1)
        assert original == Path(original_case['checkpoint']).parent / 'initial.ckpt'
        identity = initial_identity(original, directory / 'initial.ckpt', 1024, 16384)
        assert dict(profile=profile, **identity) == recorded_initial
        initial.append(recorded_initial)
        for name in ('original_checkpoint', 'candidate_checkpoint'):
            hashes[identity[name]] = identity[name + '_sha256']
        founder_path = directory / 'protocol.json'
        founder = read(founder_path)
        assert founder['random_initialization'] and not founder['imported_weights'] and founder['seed'] == spec['seed']
        assert founder['source_schedule'] == assessment['exposure'] and founder['replay_capacity'] == 16384
        assert founder['updates'] == spec['source_observations'] and founder['profile'] == profile
        authenticate(founder['authenticated_inputs'])
        completion = read(directory / 'result.json')
        assert recorded_completion['result'] == completion and completion['complete'] and completion['full_corpus_completed']
        assert completion['protocol'] == founder and completion['session'] == read(directory / 'session.json')
        final = require_complete(directory, assessment)
        assert final['parameters'] == completion['parameters']
        assert Path(completion['checkpoint']) == directory / 'stage-4.ckpt'
        assert completion['checkpoint_sha256'] == file_hash(directory / 'stage-4.ckpt')
        expected = list(map(str, live_arguments(directory, Path(evaluation['schedule']), spec['source_observations'], profile, 16384)))
        commands_path = directory / 'commands/commands.json'
        commands = read(commands_path)
        assert len(commands) == 1 and commands[0][1:] == expected
        assert Path(commands[0][0]).resolve() == Path('build/synapticgenesis.exe').resolve()
        for path in (founder_path, directory / 'result.json', directory / 'session.json',
                     commands_path, directory / 'commands/command-001.log'):
            hashes[path.as_posix()] = file_hash(path)
        sessions.append(dict(profile=profile, exact_declared_learning_arguments=True,
                             session=completion['session'], final_counters=final))
    for row in result['rows']:
        directory = root / f'{row["profile"]}-stage-{row["stage"]}'
        local_path = directory / 'result.json'
        local = read(local_path)
        assert all(row[key] == value for key, value in local.items())
        assert row['plan_sha256'] == file_hash(assessment_path)
        assert row['founder_protocol_sha256'] == file_hash(root / f'founder-{row["profile"]}' / 'protocol.json')
        checkpoint = root / f'founder-{row["profile"]}' / f'stage-{row["stage"]}.ckpt'
        assert Path(row['checkpoint']) == checkpoint and file_hash(checkpoint) == row['checkpoint_sha256']
        meta, extra = policy_checkpoint(checkpoint)
        assert (meta[1], meta[5], meta[6]) == (6, 1, 128)
        assert list(meta[2:5]) == evaluation['profiles'][row['profile']] and meta[10] == spec['seed']
        assert meta[24] == row['online_updates'] == evaluation['stage_endpoints'][row['stage'] - 1]
        assert row['parameters'] == meta[14] and row['spiking_neurons'] == meta[3] * meta[4]
        assert extra[2] == 4 and extra[3] == 16384 and counters(checkpoint) == row['exposure_counters']
        assert row['exposure_counters']['curriculum_stage'] == row['stage']
        policy = hashlib.sha256(struct.pack(f'<{len(extra)}Q', *extra)).hexdigest()
        assert policy == row['replay_policy_sha256']
        policies.setdefault(row['stage'], set()).add(policy)
        commands, raw = verify_saved(directory, row, evaluation, meta[7])
        assert len(commands) == 10
        hashes.update(raw)
        hashes[checkpoint.as_posix()] = row['checkpoint_sha256']
        hashes[local_path.as_posix()] = file_hash(local_path)
        prior = next(r for r in baseline['rows'] if (r['profile'], r['stage']) == (row['profile'], row['stage']))
        assert file_hash(prior['checkpoint']) == prior['checkpoint_sha256']
        assert counters(Path(prior['checkpoint'])) == prior['exposure_counters']
        old_directory = Path(spec['baseline_comparison']).parent / f'{row["profile"]}-stage-{row["stage"]}'
        _, old_raw = verify_saved(old_directory, prior, evaluation, prior['exposure_counters']['global_updates'])
        hashes.update(old_raw)
        hashes[prior['checkpoint']] = prior['checkpoint_sha256']
        cases.append(dict(profile=row['profile'], stage=row['stage'], checkpoint_sha256=row['checkpoint_sha256'],
            result_sha256=hashes[local_path.as_posix()], command_journal_sha256=hashes[(directory / 'commands.json').as_posix()],
            native_assessment_commands=10, replay_policy_sha256=policy, exact_saved_assessment_verified=True))
    differences = verify_comparisons(result['rows'], baseline)
    assert all(len(values) == 1 for values in policies.values()) and len(policies) == 4
    assert result['native_learning_commands'] == plan['native_learning_commands_planned'] == 2
    assert result['native_assessment_commands'] == plan['native_assessment_commands_planned'] == 80
    # Recheck evidence after all reads; active studies are in separate directories.
    authenticate(hashes)
    authenticate(plan['authenticated_inputs'])
    authenticate(predecessors)
    return dict(passed=True, result_sha256=file_hash(result_path), protocol_sha256=file_hash(plan_path),
        verified_candidate_native_commands=82, candidate_learning_commands=2, candidate_book_evaluations=48,
        candidate_generations=32, verified_baseline_assessment_commands=80, cases=cases,
        initial_identity=initial, learning_sessions=sessions, paired_differences=differences,
        raw_artifact_sha256=hashes, identical_candidate_replay_histories_across_sizes=True,
        paired_source_update_and_speech_counts_match=True, replay_target_count_differences_preserved=True,
        new_native_commands=0, new_model_updates=0, reserved_tests_scored=False, reproduction_admitted=False,
        implementation_sha256={p.as_posix(): file_hash(p) for p in (Path(__file__),
            Path('scripts/checkpoint_assessment.py'), Path('scripts/prose_replay_capacity.py'),
            Path('scripts/prose_size_comparison.py'))},
        limits='Verification of one exploratory seed and its original controls. This adds no independent numerical '
               'oracle, new training seed, speed benchmark or evidence of useful general question answering.')


def publish(root, report, audit_path):
    if report.exists() or audit_path.exists():
        raise ValueError('Use fresh publication paths')
    evidence = verify(root)
    report.write_bytes((root / 'result.json').read_bytes())
    assert file_hash(report) == evidence['result_sha256']
    write(audit_path, dict(**evidence, published_sha256=file_hash(report), exact_result_copy=True))
    print('Verified 82 candidate native commands and 80 baseline assessments; published exact replay-capacity results.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('runs/prose-replay-capacity'))
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    args = parser.parse_args()
    publish(args.root, args.report, args.audit)
