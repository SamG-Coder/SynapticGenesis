"""Fixed source admission and policy checks for a controlled objective comparison."""
from pathlib import Path
import shutil

from corpus.selection import require_training_spec
from development_schedule import prose_history
from membrane_study_state import f32, require
from native_experiment import read
from prose_founder import file_hash, live_arguments

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / 'data/membrane-learning-v1.json'


def validate(spec):
    require(spec['version'] == 'membrane-learning-v1' and spec['profile'] == '105m' and
            spec['shape'] == [1024, 4096, 8] and spec['parameters'] == 104851472,
            'Unexpected objective-comparison model')
    require(spec['seeds'] == [1337, 2027, 4099] and spec['endpoints'] == [8176, 26110],
            'Unexpected seeds or exposure endpoints')
    require([(a['name'], a['activity_cost'], a['membrane_cost']) for a in spec['arms']] == [
        ('control', 0, 0), ('activity-0p1', .1, 0), ('membrane-0p001', 0, .001),
        ('membrane-0p01', 0, .01), ('membrane-0p1', 0, .1)], 'Objective arms changed')
    require((spec['membrane_band'], spec['learning_rate'], spec['replay_capacity'],
             spec['replay_every'], spec['log_every'], spec['save_every']) == (1.5, .0003, 16384, 4, 1, 8192),
            'Shared learning policy changed')
    require(spec['stage_two_training_monitors'] == [572, 420], 'Training monitor choice changed')


def arguments(spec, arm, seed, out, schedule):
    validate(spec)
    require(arm in spec['arms'] and seed in spec['seeds'], 'Undeclared objective case')
    args = list(live_arguments(out, schedule, spec['endpoints'][-1], spec['profile'],
                               spec['replay_capacity'], save_every=spec['save_every']))
    for key, value in [('seed', seed), ('log-every', spec['log_every'])]:
        args[args.index('--' + key) + 1] = value
    if arm['activity_cost']:
        args += ['--activity-cost', arm['activity_cost']]
    if arm['membrane_cost']:
        args += ['--membrane-cost', arm['membrane_cost'], '--membrane-band', spec['membrane_band']]
    return list(map(str, args))


def admit():
    spec = read(SPEC)
    validate(spec)
    evaluation = read(spec['evaluation_spec'])
    schedule = Path(evaluation['schedule'])
    _, exposure, inputs = prose_history(schedule)
    source = require_training_spec(evaluation['source_spec'])
    roles = {r['id']: (r['split'], r['stage']) for r in source['sources']}
    require(all(roles[n][0] == 'validation' for n in evaluation['validation_books']) and
            all(roles[n] == ('train', 1) for n in evaluation['training_retention_books']) and
            all(roles[n] == ('train', 2) for n in spec['stage_two_training_monitors']),
            'Assessment roles differ from the admitted sources')
    selected = evaluation['validation_books'] + evaluation['training_retention_books'] + spec['stage_two_training_monitors']
    require(len(selected) == len(set(selected)) == 8, 'Assessment books overlap')
    require(evaluation['evaluation']['target_bytes_per_book'] == 65536 and
            (evaluation['evaluation']['batch'], evaluation['evaluation']['context'], evaluation['evaluation']['batches']) ==
            (16, 128, 32) and len(evaluation['generation']['prompts']) == 4, 'Assessment geometry changed')
    require([s['end_update'] for s in exposure['stages'][:2]] == spec['endpoints'], 'Source stages changed')
    expected, pairs = {}, 0
    for stage in exposure['stages'][:2]:
        pairs += stage['passes'] * stage['source_pairs_per_pass']
        end = stage['end_update']
        expected[str(end)] = dict(online_updates=end, global_updates=end + end // 4,
            observed_pairs=pairs, generated_bytes=end // 500 * 96, replay_updates=end // 4,
            curriculum_stage=stage['stage'])
    baseline = read(spec['legacy_control_result'])
    require(baseline['complete'] and baseline['same_initial_model_arrays'] and
            not baseline['reserved_tests_scored'], 'Legacy control evidence incomplete')
    legacy = {str(r['stage']): dict(checkpoint=r['checkpoint'], sha256=r['checkpoint_sha256'])
              for r in baseline['rows'] if r['profile'] == '105m' and r['stage'] in (1, 2)}
    require(set(legacy) == {'1', '2'}, 'Legacy control endpoints missing')
    initial = [r for r in baseline['initial'] if r['profile'] == '105m']
    require(len(initial) == 1, 'Legacy initial state missing')
    legacy['initial'] = dict(checkpoint=initial[0]['candidate_checkpoint'],
                             sha256=initial[0]['candidate_checkpoint_sha256'])
    for row in legacy.values():
        require(file_hash(row['checkpoint']) == row['sha256'], 'Legacy control checkpoint changed')
        inputs.append(Path(row['checkpoint']))
    evidence = read(spec['gradient_evidence'])
    require(evidence['complete'] and evidence['chunks_analyzed'] == 576 and evidence['new_model_updates'] == 0,
            'Gradient scale evidence differs')
    inputs += [SPEC, Path(spec['evaluation_spec']), Path(spec['legacy_control_result']),
               Path(spec['gradient_evidence']), Path('data/training-selection.json')]
    return dict(specification=spec, evaluation=evaluation, exposure=exposure, expected_exposure=expected,
                legacy_control=legacy, authenticated_inputs={str(p.resolve()): file_hash(p) for p in inputs})


def verify_state(state, spec, arm, seed, expected):
    require(state['shape'] == spec['shape'] and state['parameters'] == spec['parameters'] and
            state['seed'] == seed and state['meta'][5:7] == [1, 128] and state['meta'][16] == 1,
            'Learned model identity differs')
    require(all(state['counters'][k] == v for k, v in expected.items()), 'Learning exposure differs')
    require((state['graph'], state['replay_every'], state['replay_capacity']) == (1, 4, 16384) and
            state['speech_policy'] == [500, 96, 40, 1066179217761871269], 'Replay or speech policy differs')
    hp = state['hyperparameters']
    require(hp[0] == hp[7] == f32(.0003) and hp[1] == f32(.01) and hp[2] == hp[6] == 1 and
            hp[3] == f32(3.4028234663852886e38) and hp[5] == f32(.8) and
            hp[4] == f32(arm['activity_cost']) and state['membrane_cost'] == f32(arm['membrane_cost']) and
            state['membrane_band'] == 1.5, 'Objective or optimizer policy differs')


def storage(out, remaining_models=15):
    # Initial, latest and two stage files per founder, plus two transactional saves.
    maximum = 352 + 12 * 104851472 + 4 * 73728 + 8 * (17 + 5 * 2 + 3 * 16384)
    required = (remaining_models * 4 + 2) * maximum + 10 * 1024**3
    free = shutil.disk_usage(Path(out).resolve().anchor).free
    require(free >= required, 'Insufficient disk for preserved objective-comparison checkpoints')
    return dict(maximum_checkpoint_bytes=maximum, remaining_models=remaining_models,
                required_free_bytes=required, available_bytes=free)
