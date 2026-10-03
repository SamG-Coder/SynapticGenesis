"""Read-only completion verification for the preserved full 411M comparison."""
from contextlib import contextmanager
from pathlib import Path
import os

from checkpoint_assessment import verify_saved
from large_founder_inputs import facts, validate
from native_experiment import read
from prose_founder import file_hash, live_arguments
from prose_large_founder import same_exposure
from prose_retention_inputs import authenticate, require


@contextmanager
def directory(path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def completed_header(plan, result):
    require(plan['status'] == 'declared_before_full_411m_learning' and
            plan['specification']['version'] == 'prose-411m-v1', 'Unexpected predecessor protocol')
    require(result['complete'] is True and result['protocol'] == plan, 'Predecessor is incomplete or changed')
    for key in ('baseline_assessments_exact', 'matched_source_replay_and_speech_exposure',
                'matched_replay_payloads', 'random_initialization'):
        require(result[key] is True, f'Predecessor did not verify {key}')
    for key in ('imported_weights', 'reserved_tests_scored', 'reproduction_admitted'):
        require(result[key] is False, f'Unexpected predecessor state: {key}')
    require(result['native_learning_commands'] == 1 and result['native_assessment_commands'] == 80,
            'Predecessor command coverage differs')
    for key, profile in (('controls', '105m'), ('rows', '411m')):
        require([(r['stage'], r['profile']) for r in result[key]] == [(i, profile) for i in range(1, 5)],
                'Predecessor is missing assessment endpoints')


def verify_assessment(path, item, evaluation, runtime, expected_checkpoint):
    """Call from the original study's working directory to preserve CLI spelling."""
    path = Path(path)
    require(read(path / 'result.json') == item, 'Saved assessment row differs')
    checkpoint = Path(item['checkpoint']).resolve()
    require(checkpoint == Path(expected_checkpoint).resolve(), 'Assessment checkpoint identity differs')
    require(file_hash(checkpoint) == item['checkpoint_sha256'], 'Assessed checkpoint changed')
    require(facts(checkpoint) == item['state'], 'Assessed checkpoint state differs')
    calls, hashes = verify_saved(path, item, evaluation, item['state']['counters']['global_updates'],
                                 executable=Path(runtime))
    require(len(calls) == 10, 'Endpoint assessment command coverage differs')
    hashes[str(checkpoint)] = item['checkpoint_sha256']
    hashes[str(path / 'result.json')] = file_hash(path / 'result.json')
    return len(calls), hashes


def verify(workspace, root, protocol_sha256):
    """Check saved outputs, commands, exposure and hashes without CUDA work.

    This function alone is not a process-liveness gate. Callers waiting to use
    the GPU must also wait on the complete study driver's process handle.
    """
    workspace, root = Path(workspace).resolve(), Path(root).resolve()
    require(root.is_relative_to(workspace / 'runs'), 'Predecessor is outside its workspace runs directory')
    with directory(workspace):
        require(file_hash(root / 'protocol.json') == protocol_sha256, 'Predecessor protocol changed')
        require(not (root / 'failure.json').exists() and (root / 'result.json').is_file(),
                'Predecessor has not completed successfully')
        plan, result = read(root / 'protocol.json'), read(root / 'result.json')
        completed_header(plan, result)
        authenticate(plan['authenticated_inputs'])
        spec, evaluation = plan['specification'], plan['evaluation']
        validate(spec)
        baseline = [r for r in read(spec['baseline_result'])['rows'] if r['profile'] == '105m']
        require([r['stage'] for r in baseline] == [1, 2, 3, 4], 'Original baseline endpoints differ')
        artifact_hashes, assessment_commands = {}, 0
        for control, row, original in zip(result['controls'], result['rows'], baseline):
            same_exposure(row['state'], control['state'])
            counters = row['state']['counters']
            require([counters[k] for k in ('channels', 'hidden', 'layers')] == spec['shape'] and
                    counters['parameters'] == spec['parameters'], 'Assessed candidate has the wrong shape')
            require(all(control[k] == original[k] for k in ('books', 'samples', 'validation_mean_nats_per_byte')),
                    'Rechecked baseline differs from its original measurements')
            expected_comparison = dict(
                validation_difference_from_105m=row['validation_mean_nats_per_byte'] -
                    control['validation_mean_nats_per_byte'],
                books=[dict(book=a['book'], role=a['role'],
                            difference_from_105m=a['loss_nats_per_byte'] - b['loss_nats_per_byte'])
                       for a, b in zip(row['books'], control['books'])])
            require(row['comparison'] == expected_comparison, 'Recorded 411M comparison differs')
            for label, item in (('baseline-105m', control), ('411m', row)):
                # Match the original driver's relative CLI arguments exactly.
                path = root.relative_to(workspace) / f"{label}-stage-{item['stage']}"
                expected = (Path(original['checkpoint']).resolve() if label == 'baseline-105m' else
                            root / 'founder-411m' / f"stage-{item['stage']}.ckpt")
                count, hashes = verify_assessment(path, item, evaluation, spec['runtime'], expected)
                assessment_commands += count
                artifact_hashes.update(hashes)
        founder = root / 'founder-411m'
        require(not (founder / 'failure.json').exists(), 'Founder has a failure record')
        completion = read(founder / 'result.json')
        require(result['founder_completion'] == completion and completion['complete'] is True and
                completion['full_corpus_completed'] is True and completion['parameters'] == 411028496 and
                completion['spiking_neurons'] == 65536, 'Founder completion differs')
        require(completion['protocol'] == read(founder / 'protocol.json') and
                completion['session'] == read(founder / 'session.json'), 'Founder protocol or session differs')
        require(completion['session']['online_updates'] == spec['source_observations'] == 216289,
                'Founder did not finish its scheduled source updates')
        journal = read(founder / 'commands/commands.json')
        relative_founder = founder.relative_to(workspace)
        expected = [str(Path(spec['runtime']).resolve()), *map(str, live_arguments(relative_founder,
                    Path(evaluation['schedule']), spec['source_observations'], spec['profile'],
                    spec['replay_capacity'], save_every=spec['save_every']))]
        require(journal == [expected], 'Founder learning command differs from the declared schedule')
        last = result['rows'][-1]
        require(file_hash(founder / 'latest.ckpt') == last['checkpoint_sha256'] == completion['checkpoint_sha256'],
                'Founder final checkpoints differ')
        require(assessment_commands == 80, 'Not all predecessor assessments were verified')
        for path in (root / 'protocol.json', root / 'result.json', founder / 'result.json',
                     founder / 'protocol.json', founder / 'session.json', founder / 'commands/commands.json',
                     founder / 'commands/command-001.log'):
            artifact_hashes[str(path)] = file_hash(path)
        return dict(passed=True, cuda_executed=False, verified_assessment_commands=assessment_commands,
                    verified_learning_commands=1, source_updates=216289,
                    result_sha256=file_hash(root / 'result.json'), artifacts_sha256=artifact_hashes,
                    quality_improvement_required_for_this_completion_gate=False)
