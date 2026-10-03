"""Audit the completed width study and publish compact, fully paired results."""
import argparse
import json
from pathlib import Path
import statistics

from checkpoint_assessment import verify_saved
from early_width_observations import audit, match_guard_speech
from native_experiment import read
from prose_early_width import arguments, initial_match, match_policy, validate
from prose_founder import file_hash, write
from prose_retention_inputs import authenticate, require, state


def summaries(rows, spec):
    expected = [(seed, case) for seed in spec['seeds'] for case in spec['cases']]
    require([(r['seed'], r['case']) for r in rows] == expected, 'Missing or reordered width cases')
    groups = []
    for case in spec['cases']:
        selected = [r for r in rows if r['case'] == case]
        values = [r['quality']['validation_mean_nats_per_byte'] for r in selected]
        firing, zero = [], []
        for row in selected:
            require(row['observations'][-1]['source_update'] == spec['source_observations'],
                    'Missing final observed source chunk')
            layers = row['observations'][-1]['pre_update_forward_layers']
            require(len(layers) == case['layers'], 'Missing final activity layer')
            firing.append(statistics.mean(r['spike_event_fraction'] for r in layers))
            zero.append(statistics.mean(r['zero_local_spike_surrogate_fraction'] for r in layers))
        groups.append(dict(case=case, parameters=selected[0]['native_result']['parameters'],
            seeds=spec['seeds'], validation_means=values, validation_mean_across_seeds=statistics.mean(values),
            validation_sample_stddev=statistics.stdev(values),
            final_observed_chunk_firing_by_seed=firing, final_observed_chunk_zero_local_surrogate_by_seed=zero,
            source_clipping_fractions=[r['native_result']['source_updates_clipped'] / spec['source_observations']
                                       for r in selected]))
    paired = []
    for control, candidate in [('c512-control', 'c512-core-half'), ('c1024-control', 'c1024-core-quarter')]:
        old, new = [next(g for g in groups if g['case']['name'] == name) for name in (control, candidate)]
        deltas = [b - a for a, b in zip(old['validation_means'], new['validation_means'])]
        paired.append(dict(control=control, candidate=candidate, seeds=spec['seeds'],
                           validation_differences=deltas, mean_difference=statistics.mean(deltas),
                           improved_seeds=sum(d < 0 for d in deltas), worsened_seeds=sum(d > 0 for d in deltas)))
    return dict(cases=groups, rate_pairs=paired)


def verify(root):
    root = Path(root)
    plan, result = read(root / 'protocol.json'), read(root / 'result.json')
    require(result['complete'] and result['protocol'] == plan and not (root / 'failure.json').exists(),
            'Width study is incomplete')
    spec = plan['specification']
    validate(spec)
    require((result['native_guard_commands'], result['native_learning_commands'],
             result['native_assessment_commands']) == (2, 15, 150), 'Native command counts differ')
    require(result['random_initialization'] and not result['imported_weights'] and
            not result['reserved_tests_scored'] and not result['reproduction_admitted'], 'Study scope differs')
    authenticate(plan['authenticated_inputs'])
    predecessors = read(root / 'predecessors-complete.json')
    authenticate(predecessors)
    evaluation = read(plan['assessment_plan'])
    source = evaluation['exposure']['stages'][0]['cumulative_source']
    schedule = Path(evaluation['specification']['schedule'])
    original = read(spec['baseline_comparison'])
    hashes = {}

    def remember(path):
        hashes[Path(path).as_posix()] = file_hash(path)
        return hashes[Path(path).as_posix()]

    for p in (root / 'protocol.json', root / 'result.json', root / 'predecessors-complete.json',
              Path(plan['assessment_plan'])):
        remember(p)
    tiny = dict(channels=8, hidden=32, layers=2, core_scale=1.)
    guard = audit(root / 'guard-observed', tiny, 1337, spec['native_guard_updates'],
                  spec['native_guard_points'], source)
    stored_guard = read(root / 'native-guard.json')
    require(all(stored_guard[k] == v for k, v in guard.items()), 'Native guard audit changed')
    require(stored_guard == result['native_guard'], 'Stored and published guard differ')
    require(match_guard_speech(root / 'guard-control', root / 'guard-observed', spec['native_guard_updates']) ==
            stored_guard['speech'], 'Native guard speech differs')
    hashes.update(guard['inputs_sha256'])
    for name in ('initial.ckpt', 'latest.ckpt'):
        require(remember(root / 'guard-control' / name) == remember(root / 'guard-observed' / name),
                'Guard checkpoint bytes differ')
    for name, points, exe, destination in [
            ('guard-control-commands', None, Path('build/synapticgenesis.exe'), 'guard-control'),
            ('guard-observed-commands', spec['native_guard_points'], Path(plan['diagnostic_executable']), 'guard-observed')]:
        recorded = read(root / name / 'commands.json')
        expected = list(map(str, arguments(tiny, 1337, root / destination, schedule,
                                           spec['native_guard_updates'], points)))
        require(len(recorded) == 1 and recorded[0][1:] == expected and
                Path(recorded[0][0]).resolve() == exe.resolve(), 'Guard command differs')
        remember(root / name / 'commands.json'); remember(root / name / 'command-001.log')
    commands = read(root / 'learning-commands/commands.json')
    require(len(commands) == len(result['rows']) == 15, 'Missing learner commands')
    summary = summaries(result['rows'], spec)
    compact, per_seed, legacy = [], {}, []
    for index, (row, command) in enumerate(zip(result['rows'], commands), 1):
        case, seed = row['case'], row['seed']
        directory = root / f'{seed}-{case["name"]}'
        require(Path(row['directory']) == directory, 'Learner directory differs')
        expected = list(map(str, arguments(case, seed, directory, schedule, spec['source_observations'],
                                          spec['observation_points'])))
        require(command[1:] == expected and Path(command[0]).resolve() == Path(plan['diagnostic_executable']).resolve(),
                'Learner command differs')
        measured = audit(directory, case, seed, spec['source_observations'], spec['observation_points'], source)
        require(measured == read(directory / 'observation-audit.json') and
                remember(directory / 'observation-audit.json') == row['observation_audit_sha256'], 'Coordinate audit differs')
        hashes.update(measured['inputs_sha256'])
        checkpoint = directory / 'latest.ckpt'
        require(remember(checkpoint) == row['checkpoint_sha256'] and state(checkpoint) == row['state'],
                'Learned checkpoint/state differs')
        remember(directory / 'initial.ckpt')
        require(row['native_result'] == read(directory / 'result.json'), 'Native learner result differs')
        observations = [json.loads(line) for line in (directory / 'observations.jsonl').read_text().splitlines()]
        require(observations == row['observations'], 'Observed rows differ')
        same_seed = per_seed.setdefault(seed, [])
        if same_seed:
            match_policy(same_seed[0]['state'], row['state'])
        if case['core_scale'] != 1:
            control = next(r for r in same_seed if r['case']['channels'] == case['channels'] and
                           r['case']['core_scale'] == 1)
            initial_match(Path(control['directory']) / 'initial.ckpt', directory / 'initial.ckpt')
        same_seed.append(row)
        quality = dict(checkpoint=str(checkpoint), checkpoint_unchanged=True, reserved_tests_scored=False, **row['quality'])
        verified_commands, quality_hashes = verify_saved(directory / 'assessment', quality,
            evaluation['specification'], row['state']['counters']['global_updates'])
        require(len(verified_commands) == 10, 'Assessment command count differs')
        hashes.update(quality_hashes)
        if seed == 1337 and case['baseline_profile']:
            old = next(r for r in original['rows'] if r['profile'] == case['baseline_profile'] and r['stage'] == 1)
            require(remember(old['checkpoint']) == row['checkpoint_sha256'] and
                    remember(Path(old['checkpoint']).parent / 'initial.ckpt') == file_hash(directory / 'initial.ckpt'),
                    'Preserved control checkpoints differ')
            require(all(row['quality'][k] == old[k] for k in row['quality']), 'Preserved control quality differs')
            legacy.append(dict(profile=case['baseline_profile'], initial_and_final_byte_identical=True,
                               checkpoint_sha256=row['checkpoint_sha256']))
        remember(root / 'learning-commands' / f'command-{index:03d}.log')
        compact.append({**row, 'observations': [{k: v for k, v in r.items() if k != 'roles'} for r in observations]})
        print('Verified', seed, case['name'], 'raw coordinates, checkpoint and all ten assessments.', flush=True)
    require(legacy == result['legacy_checks'], 'Legacy control record differs')
    remember(root / 'learning-commands/commands.json'); remember(root / 'native-guard.json')
    authenticate(plan['authenticated_inputs'])
    publication = dict(complete=True, source_result_sha256=file_hash(root / 'result.json'), protocol=plan,
        rows=compact, summary=summary, native_guard=stored_guard, legacy_checks=legacy,
        native_commands_verified=167, reserved_tests_scored=False, reproduction_admitted=False,
        sampled_role_statistics_reconstructed_from_raw_coordinates=True,
        compact_scope='All quality results and layer-cache observations are included. Per-tensor sampled-role '
                      'statistics remain in authenticated local observation files and can be regenerated with this auditor.',
        limits=spec['limits'])
    evidence = dict(passed=True, native_commands_verified=167, cases=15, assessment_commands=150,
        raw_coordinate_files_verified=15 * 19 + 10, evidence_sha256=hashes,
        source_result_sha256=file_hash(root / 'result.json'),
        publisher_sha256=file_hash(__file__), independent_neural_forward_executed=False,
        reserved_tests_scored=False, reproduction_admitted=False)
    return publication, evidence


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists() and not args.audit.exists(), 'Use fresh publication paths')
    publication, evidence = verify(args.experiment)
    write(args.output, publication)
    write(args.audit, evidence)
