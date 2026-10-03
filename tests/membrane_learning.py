"""CPU evidence and fault checks; synthetic gate results are never CUDA evidence."""
import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import membrane_learning as runner
import verify_membrane_acceptance as gate_reader
from membrane_learning_inputs import SPEC, admit, arguments, storage, validate, verify_state
from membrane_learning_metrics import audit_metrics
from membrane_study_state import f32, initial_fingerprint, matched_exposure, read_state
from native_experiment import read
from prose_founder import file_hash, write


def rejected(label, action, expected, records):
    try:
        action()
    except (ValueError, RuntimeError) as error:
        assert expected in str(error), (label, str(error))
        records.append(dict(case=label, reason=str(error)))
    else:
        raise AssertionError('Unexpected acceptance: ' + label)


def checkpoints(out, admission, records):
    fixtures = Path('runs/membrane-regularization-worktree/runs/membrane-checkpoint-host-final')
    legacy, wrapped = [fixtures / f'6-5-0-{name}.ckpt' for name in ('legacy', 'membrane')]
    originals = {str(p): file_hash(p) for p in (legacy, wrapped)}
    left, right = read_state(legacy), read_state(wrapped)
    assert left['prefix_bytes'] == 288 and right['prefix_bytes'] == 352
    assert right['membrane_cost'] == f32(.01) and right['membrane_band'] == 1.5
    assert initial_fingerprint(legacy, left) == initial_fingerprint(wrapped, right)
    matched_exposure(left, right)
    raw = wrapped.read_bytes()
    replay_at = right['prefix_bytes'] + 12 * right['parameters'] + 4 * right['meta'][18]

    def changed(offset, layout, value):
        data = bytearray(raw)
        struct.pack_into(layout, data, offset, value)
        return data

    malformed = [
        ('short-header', raw[:20], 'Truncated checkpoint header'),
        ('short-wrapper', raw[:300], 'Truncated checkpoint header'),
        ('bad-magic', changed(0, '<Q', 0), 'Unexpected associative'),
        ('bad-layout', changed(14 * 8, '<Q', 7), 'Model layout'),
        ('nonfinite-policy', changed(256, '<f', math.nan), 'Nonfinite'),
        ('wrapper-base', changed(288 + 16, '<Q', 6), 'policy wrapper'),
        ('wrapper-high-bits', changed(288 + 24, '<Q', 1 << 40), 'high bits'),
        ('wrapper-zero-cost', changed(288 + 24, '<Q', 0), 'policy values'),
        ('wrapper-band', changed(288 + 32, '<Q', 0x40000000), 'policy values'),
        ('wrapper-reserved', changed(288 + 40, '<Q', 1), 'policy wrapper'),
        ('extent', raw + b'\0', 'checkpoint extent'),
        ('unbounded-replay', changed(31 * 8, '<Q', 1 << 40), 'Unbounded replay'),
        ('replay-si', changed(replay_at + 9 * 8, '<Q', 1), 'without consolidation'),
        ('replay-records', changed(replay_at + 19 * 8, '<Q', 2), 'Replay records differ'),
    ]
    metadata = changed(replay_at + 15 * 8, '<Q', 1)
    struct.pack_into('<Q', metadata, replay_at + 16 * 8, 2)
    malformed.append(('group-metadata-truncated', metadata, 'Truncated replay group'))
    for name, data, error in malformed:
        destination = out / f'{name}.ckpt'
        destination.write_bytes(data)
        rejected(name, lambda p=destination: read_state(p), error, records)
    altered = out / 'changed-neural-array.ckpt'
    data = bytearray(raw)
    data[352] ^= 1
    altered.write_bytes(data)
    assert initial_fingerprint(altered, read_state(altered)) != initial_fingerprint(wrapped, right)
    altered_activity = out / 'activity-policy.ckpt'
    data = bytearray(legacy.read_bytes())
    struct.pack_into('<f', data, 256 + 4 * 4, .1)
    altered_activity.write_bytes(data)
    assert initial_fingerprint(altered_activity, read_state(altered_activity)) == initial_fingerprint(legacy, left)
    for key, value in [('replay_payload_sha256', 'changed'), ('cursor_and_rng', [1]),
                       ('counters', {}), ('speech_policy', [1])]:
        other = deepcopy(right)
        other[key] = value
        rejected('matched-' + key, lambda other=other: matched_exposure(left, other), key, records)
    for offset in (0, 2, 5, 6):
        other = deepcopy(right)
        other['hyperparameters'][offset] += .1
        rejected('optimizer-' + str(offset), lambda other=other: matched_exposure(left, other),
                 'optimizer setting', records)
    real_states = []
    spec = admission['specification']
    for label, row in admission['legacy_control'].items():
        state = read_state(row['checkpoint'])
        if label == 'initial':
            expected = dict(online_updates=0, global_updates=0, observed_pairs=0, generated_bytes=0,
                            replay_updates=0, replay_pairs=0, curriculum_stage=1)
        else:
            expected = admission['expected_exposure'][str(spec['endpoints'][int(label) - 1])]
        verify_state(state, spec, spec['arms'][0], 1337, expected)
        real_states.append(dict(stage=label, checkpoint_sha256=row['sha256'], counters=state['counters']))
        mutated = deepcopy(state)
        mutated['hyperparameters'][5] = .9
        rejected('temperature-' + label,
                 lambda: verify_state(mutated, spec, spec['arms'][0], 1337, expected), 'optimizer policy', records)
    assert all(file_hash(p) == digest for p, digest in originals.items())
    return dict(native_host_fixture_sha256=originals, wrapped_and_ordinary_initial_neural_state_exact=True,
                altered_arrays_detected=True, real_105m_states=real_states,
                native_checksum_validation_claimed=False)


def metric_fixture(out, records):
    expected = {'2': dict(curriculum_stage=1, observed_pairs=135),
                '4': dict(curriculum_stage=2, observed_pairs=276)}
    rows = [dict(online_update=i, curriculum_stage=1 if i <= 2 else 2,
        global_update=i + i // 4, replay_updates=i // 4, consolidation_events=0,
        observed_pairs=pairs, loss=loss, spike_rate=rate, gradient_norm=norm,
        membrane_cost=.01, membrane_band=1.5)
        for i, (pairs, loss, rate, norm) in enumerate([
            (128, 2., .5, .5), (135, 3., .75, 1.), (263, 4., 1., 1.5), (276, 5., .25, 2.)], 1)]
    rows.insert(2, dict(event='curriculum_transition', online_update=2, stage=2))
    path = out / 'metrics.jsonl'
    start = dict(event='session_start', online_update=0, curriculum_stage=1, resume=False,
                 si_strength=0, replay_every=4, learning_rate=.0003)

    def save(data):
        path.write_text(''.join(json.dumps(row) + '\n' for row in [start, *data]), encoding='utf-8')

    save(rows)
    result = audit_metrics(path, expected, membrane_cost=.01)
    assert result['source_updates'] == 4 and result['every_source_update_present']
    assert [r['source_target_bytes'] for r in result['stages']] == [135, 141]
    assert [r['reported_above_clip'] for r in result['stages']] == [0, 2]
    assert [r['reported_equal_to_clip'] for r in result['stages']] == [1, 0]
    assert result['stages'][0]['mean_language_loss'] == (128 * 2 + 7 * 3) / 135
    assert result['stages'][1]['gradient_norm_p95'] == 1.975
    assert not result['replay_gradient_norms_measured']
    variations = [('missing-row', rows[:3] + rows[4:], 'Missing or duplicated'),
                  ('missing-transition', rows[:2] + rows[3:], 'Incomplete metric'),
                  ('duplicate-row', [rows[0], *rows], 'Missing or duplicated'),
                  ('misplaced-transition', [rows[2], *rows[:2], *rows[3:]], 'misplaced'),
                  ('incomplete-stream', rows[:-1], 'Incomplete metric'),
                  ('duplicate-session', [start, *rows], 'initial learning session')]
    for field, value, error in [('loss', math.nan, 'Invalid source'), ('spike_rate', 1.1, 'Invalid source'),
        ('gradient_norm', -.1, 'Invalid source'), ('observed_pairs', 129, 'exposure counters'),
        ('replay_updates', 1, 'exposure counters'), ('curriculum_stage', 2, 'exposure counters'),
        ('membrane_cost', 0, 'membrane policy')]:
        data = deepcopy(rows)
        data[0][field] = value
        variations.append(('metric-' + field, data, error))
    for name, data, error in variations:
        save(data)
        rejected(name, lambda: audit_metrics(path, expected, membrane_cost=.01), error, records)
    save(rows)
    actual = Path('runs/prose-replay-capacity/founder-105m/metrics.jsonl')
    rejected('sparse-historical-log', lambda: audit_metrics(actual, expected), 'Missing or duplicated', records)
    return dict(fixture=result, rounded_threshold_separate=True, byte_weighted_means=True,
                historical_sparse_log_rejected=True)


def specifications(out, admission, records):
    spec = admission['specification']
    schedule = Path(admission['evaluation']['schedule'])
    planned = runner.cases(spec, out, Path('candidate.exe').resolve(), schedule)
    assert len(planned) == 15 and len({r['directory'] for r in planned}) == 15
    assert [(r['seed'], r['arm']['name']) for r in planned] == [
        (seed, arm['name']) for seed in (1337, 2027, 4099) for arm in spec['arms']]
    for case in planned:
        args = case['argv']
        assert args[1] == 'live' and args[args.index('--updates') + 1] == '26110'
        assert args[args.index('--log-every') + 1] == '1'
        assert args[args.index('--seed') + 1] == str(case['seed'])
        assert args[args.index('--replay-capacity') + 1] == '16384'
        assert args[args.index('--curriculum') + 1] == str(schedule)
        assert ('--membrane-cost' in args) == bool(case['arm']['membrane_cost'])
        assert ('--activity-cost' in args) == bool(case['arm']['activity_cost'])
    for key, value in [('parameters', 411028496), ('seeds', [1337]), ('endpoints', [8176, 216289]),
                       ('log_every', 512), ('membrane_band', 2.), ('stage_two_training_monitors', [107, 420])]:
        other = deepcopy(spec)
        other[key] = value
        rejected('spec-' + key, lambda: validate(other), 'changed' if key in (
            'log_every', 'membrane_band', 'stage_two_training_monitors') else 'Unexpected', records)
    rejected('undeclared-seed', lambda: arguments(spec, spec['arms'][0], 42, out, schedule),
             'Undeclared', records)
    with patch('membrane_learning_inputs.shutil.disk_usage', return_value=SimpleNamespace(free=1)):
        rejected('disk-reservation', lambda: storage(out), 'Insufficient disk', records)
    return dict(learning_cases=len(planned), endpoint_assessments=30, commands_per_assessment=12,
                expected_source_exposure=admission['expected_exposure'], input_files=len(admission['authenticated_inputs']))


def legacy_comparator(admission, records):
    baseline = read(admission['specification']['legacy_control_result'])
    assessments = deepcopy([r for r in baseline['rows'] if r['profile'] == '105m' and r['stage'] in (1, 2)])
    for row in assessments:
        row['books'] += [dict(book=book, role='stage_two_training') for book in (572, 420)]
    case = dict(seed=1337, arm=admission['specification']['arms'][0])
    directory = Path(admission['legacy_control']['initial']['checkpoint']).parent
    result = runner.legacy_check(admission, case, directory, assessments)
    assert result['passed']
    variants = []
    other = deepcopy(assessments)
    other[0]['books'][0]['loss_nats_per_byte'] += .01
    variants.append(('legacy-book', other))
    other = deepcopy(assessments)
    other[1]['samples'][0]['output_hex'] += '00'
    variants.append(('legacy-generation', other))
    other = deepcopy(assessments)
    other[1]['validation_mean_nats_per_byte'] += .01
    variants.append(('legacy-mean', other))
    # The positive check above hashes the actual three checkpoints. Reuse those
    # known identities for small assessment mutations, avoiding repeated GiB reads.
    known = {str(Path(row['checkpoint'])): row['sha256'] for row in admission['legacy_control'].values()}
    with patch.object(runner, 'file_hash', side_effect=lambda p: known[str(p)]):
        for name, altered in variants:
            rejected(name, lambda: runner.legacy_check(admission, case, directory, altered),
                     'assessment differs', records)
    return result


def acceptance_aggregation(out, records):
    """Exercise gate plumbing with explicitly mocked native result readers."""
    fixture = out / 'synthetic-acceptance'
    fixture.mkdir()
    runtime = fixture / 'runtime.fixture'
    runtime.write_bytes(b'not executable; synthetic acceptance gate fixture')
    previous = dict(fixture=True, completed_predecessor=True)
    write(fixture / 'predecessor-verified.json', previous)
    wanted = [dict(label=f'fixture-{i}', argv=['not-executable', str(i)]) for i in range(20)]
    completed = []
    for i, item in enumerate(wanted, 1):
        log = fixture / f'command-{i:03d}.log'
        log.write_bytes(f'CPU synthetic fixture {i}\n'.encode())
        completed.append(dict(label=item['label'], returncode=0, log_sha256=file_hash(log)))
    plan = dict(version='membrane-acceptance-v1', status='declared_before_gpu_execution', root=str(fixture),
        authenticated_inputs={str(runtime): file_hash(runtime)}, workspace=str(Path.cwd()),
        predecessor='synthetic', predecessor_protocol_sha256='synthetic', commands=wanted,
        executables=dict(python='synthetic', ctest='synthetic', old='synthetic', new=str(runtime), probe='synthetic'))
    write(fixture / 'protocol.json', plan)
    protocol = file_hash(fixture / 'protocol.json')
    execution = dict(phase='complete', protocol_sha256=protocol, native_work_started=True, completed=20)
    write(fixture / 'execution.json', execution)
    write(fixture / 'commands.json', wanted)
    write(fixture / 'completed-commands.json', completed)
    result = dict(complete=True, passed=True, protocol_sha256=protocol, commands=completed,
                  fixture_aggregate='expected', predecessor_verification_sha256=file_hash(fixture / 'predecessor-verified.json'))
    write(fixture / 'result.json', result)
    accepted = SimpleNamespace(ROOT=fixture, authenticate=runner.authenticate, command_plan=lambda *a: wanted,
        collect_results=lambda *a: dict(passed=True, fixture_aggregate='expected'), verify_predecessor=lambda *a: previous)
    old_path = sys.path[:]
    try:
        with patch.dict(sys.modules, {'membrane_acceptance': accepted}):
            print('Synthetic gate fixture only; native result readers are mocked.', flush=True)
            passed = gate_reader.verify(fixture, protocol, fixture / 'verified.json')
            assert passed['command_count'] == 20 and passed['native_commands_launched'] == 0
            mutations = [
                ('gate-incomplete', 'result.json', dict(result, complete=False), 'Incomplete candidate'),
                ('gate-phase', 'execution.json', dict(execution, phase='running'), 'Incomplete candidate'),
                ('gate-coverage', 'completed-commands.json', completed[:-1], 'coverage differs'),
                ('gate-journal', 'commands.json', wanted[:-1], 'journal differs'),
                ('gate-aggregate', 'result.json', dict(result, fixture_aggregate='changed'), 'artifacts differ'),
                ('gate-predecessor', 'predecessor-verified.json', dict(previous, changed=True), 'predecessor verification'),
            ]
            for name, file, value, error in mutations:
                path = fixture / file
                original = path.read_bytes()
                write(path, value)
                rejected(name, lambda: gate_reader.verify(fixture, protocol, fixture / 'rejected.json'), error, records)
                assert not (fixture / 'rejected.json').exists()
                path.write_bytes(original)
            log = fixture / 'command-007.log'
            original = log.read_bytes()
            log.write_bytes(b'changed')
            rejected('gate-log', lambda: gate_reader.verify(fixture, protocol, fixture / 'rejected.json'),
                     'command result differs', records)
            log.write_bytes(original)
            write(fixture / 'failure.json', dict(error='fixture'))
            rejected('gate-failure-file', lambda: gate_reader.verify(fixture, protocol, fixture / 'rejected.json'),
                     'not completed successfully', records)
    finally:
        sys.path[:] = old_path
    return dict(synthetic_success_and_artifact_faults_checked=True, native_result_readers_mocked=True,
                proves_actual_cuda_acceptance=False)


def dispatch_gates(out, admission, records):
    identity = dict(pid=123, executable='python.exe', creation_filetime=456)
    reached = []
    for scenario, error in [('identity', 'identity changed'), ('exit', 'predecessor failed'),
                             ('verification', 'verification failed'), ('runtime', 'runtime differs'),
                             ('ready', 'TEST native boundary')]:
        directory = out / f'dispatch-{scenario}'
        directory.mkdir()
        runtime = directory / 'candidate.fixture'
        runtime.write_bytes(b'not executable')
        plan = dict(version='membrane-learning-plan-v1', status='declared_before_learning',
            specification=admission['specification'], evaluation=admission['evaluation'], source_checkout=str(ROOT),
            workspace=str(Path.cwd().resolve()), runtime=str(runtime), authenticated_inputs={str(runtime): file_hash(runtime)},
            learning_commands_planned=15, assessment_commands_planned=360, wait_for_acceptance_driver=identity,
            acceptance='synthetic', acceptance_protocol_sha256='synthetic', python=sys.executable)
        plan['cases'] = runner.cases(plan['specification'], directory, runtime, Path(plan['evaluation']['schedule']))
        write(directory / 'protocol.json', plan)
        process = Mock()
        process.identity = dict(identity, creation_filetime=999) if scenario == 'identity' else identity
        process.wait.return_value = 0
        if scenario == 'exit':
            process.wait.side_effect = RuntimeError('TEST predecessor failed')

        def verify_subprocess(argv, **kwargs):
            assert process.wait.call_count == 1 and process.close.call_count == 1
            assert read(directory / 'execution.json')['native_work_started'] is False
            if scenario != 'verification':
                write(directory / 'acceptance-verified.json', dict(passed=True, candidate_runtime=str(runtime),
                    candidate_runtime_sha256='wrong' if scenario == 'runtime' else file_hash(runtime)))
            return SimpleNamespace(returncode=1 if scenario == 'verification' else 0, stdout=b'CPU fixture', stderr=b'')

        def native_boundary(*args):
            assert scenario == 'ready'
            assert (directory / 'acceptance-verified.json').exists()
            reached.append(scenario)
            raise RuntimeError('TEST native boundary reached; no command executed')

        with patch.object(runner, 'ProcessGate', return_value=process), \
             patch.object(runner.subprocess, 'run', side_effect=verify_subprocess) as verifier, \
             patch.object(runner, 'NativeCommands', side_effect=native_boundary) as native:
            rejected('dispatch-' + scenario, lambda: runner.execute(directory, file_hash(directory / 'protocol.json')),
                     error, records if scenario != 'ready' else [])
            assert native.call_count == int(scenario == 'ready')
            assert verifier.call_count == int(scenario not in ('identity', 'exit'))
        assert read(directory / 'failure.json')['completed_models'] == 0
        if scenario != 'identity':
            assert read(directory / 'execution.json')['native_work_started'] is False
        assert process.close.call_count == 1
    assert reached == ['ready']
    return dict(process_identity_and_exit_enforced=True, verifier_runs_before_native_boundary=True,
                failed_gates_dispatch_no_native_commands=True, process_and_verifier_mocked=True)


def check(out, pending_acceptance=None):
    out = out.resolve()
    assert not out.exists(), 'Use a fresh CPU audit directory'
    out.mkdir(parents=True)
    records = []
    admission = admit()
    state = checkpoints(out, admission, records)
    metrics = metric_fixture(out, records)
    protocol = specifications(out, admission, records)
    archived = legacy_comparator(admission, records)
    synthetic = acceptance_aggregation(out, records)
    dispatch = dispatch_gates(out, admission, records)
    if pending_acceptance:
        actual_gate = pending_acceptance.resolve()
        assert read(actual_gate / 'execution.json')['native_work_started'] is False
        pending = subprocess.run([sys.executable, '-X', 'utf8', str(ROOT / 'scripts/verify_membrane_acceptance.py'),
            '--acceptance', str(actual_gate), '--protocol-sha256', file_hash(actual_gate / 'protocol.json'),
            '--output', str(out / 'must-not-exist.json')], capture_output=True)
        (out / 'pending-acceptance.log').write_bytes(pending.stdout + pending.stderr)
        assert pending.returncode != 0 and b'has not completed successfully' in pending.stderr
        assert not (out / 'must-not-exist.json').exists()
    files = [Path(__file__), SPEC, *(ROOT / 'scripts' / name for name in (
        'membrane_learning.py', 'membrane_learning_inputs.py', 'membrane_learning_metrics.py',
        'membrane_study_state.py', 'verify_membrane_acceptance.py'))]
    result = dict(passed=True, cpu_only=True, native_commands=0, new_model_updates=0,
        checkpoint_evidence=state, metrics=metrics, protocol=protocol, archived_control_comparator=archived,
        synthetic_acceptance=synthetic,
        dispatch_gates=dispatch, actual_pending_acceptance_rejected=bool(pending_acceptance), rejected_cases=records,
        source_sha256={str(p): file_hash(p) for p in files},
        limits='No GPU checks or actual positive acceptance were performed. Synthetic acceptance and process '
               'fixtures test orchestration only. The real queued acceptance and all learning remain pending.')
    write(out / 'result.json', result)
    print(f'CPU audit passed: {len(records)} rejection paths, real legacy/wrapped states, full-source metrics and queued gate refusal.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--pending-acceptance', type=Path)
    args = parser.parse_args()
    check(args.out, args.pending_acceptance)
