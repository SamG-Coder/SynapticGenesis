"""Run the declared objective comparison after completed native acceptance."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys

from checkpoint_assessment import measure, verify_saved
from membrane_learning_inputs import ROOT, SPEC, admit, arguments, storage, validate, verify_state
from membrane_learning_metrics import audit_metrics
from membrane_study_state import initial_fingerprint, matched_exposure, read_state, require
from native_experiment import NativeCommands, read
from process_gate import ProcessGate
from prose_founder import file_hash, write


def authenticate(pins):
    for path, expected in pins.items():
        require(file_hash(path) == expected, 'Declared learning input changed: ' + path)


def cases(spec, out, runtime, schedule):
    return [dict(seed=seed, arm=arm, directory=str(out / f'{seed}-{arm["name"]}'),
        argv=[str(runtime), *arguments(spec, arm, seed, out / f'{seed}-{arm["name"]}', schedule)])
        for seed in spec['seeds'] for arm in spec['arms']]


def declare(out, workspace, acceptance, wait_pid, wait_executable):
    out, workspace, acceptance = map(lambda p: Path(p).resolve(), (out, workspace, acceptance))
    require(Path.cwd().resolve() == workspace and not out.exists(), 'Use the workspace cwd and a fresh study directory')
    accepted = read(acceptance / 'protocol.json')
    require(accepted['version'] == 'membrane-acceptance-v1' and Path(accepted['workspace']) == workspace and
            not accepted['language_trial'], 'Unexpected candidate acceptance')
    authenticate(accepted['authenticated_inputs'])
    plan = admit()
    spec = plan['specification']
    runtime = Path(accepted['executables']['new'])
    require(file_hash(runtime) == accepted['authenticated_inputs'][str(runtime)], 'Candidate runtime changed')
    gate = ProcessGate(wait_pid, wait_executable) if wait_pid else None
    try:
        require(gate is not None and Path(gate.identity['executable']).name.lower() == 'python.exe',
                'Declare while the complete acceptance Python driver is available')
        budget = storage(out)
        inputs = [acceptance / 'protocol.json', Path(sys.executable), SPEC,
                  ROOT / 'data/training-selection.json', ROOT / 'tests/membrane_learning.py',
                  *sorted((ROOT / 'scripts').rglob('*.py'))]
        pins = dict(plan['authenticated_inputs'])
        pins.update(accepted['authenticated_inputs'])
        pins.update({str(p.resolve()): file_hash(p) for p in inputs})
        planned = cases(spec, out, runtime, Path(plan['evaluation']['schedule']))
        plan.update(version='membrane-learning-plan-v1', status='declared_before_learning',
            created_utc=datetime.now(timezone.utc).isoformat(), workspace=str(workspace),
            source_checkout=str(ROOT), source_commit=subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            acceptance=str(acceptance), acceptance_protocol_sha256=file_hash(acceptance / 'protocol.json'),
            wait_for_acceptance_driver=gate.identity, runtime=str(runtime), python=sys.executable,
            cases=planned, storage=budget, authenticated_inputs=pins,
            learning_commands_planned=15, assessment_commands_planned=360,
            random_initialization=True, imported_weights=False, generated_text_targets=False,
            reserved_tests_scored=False, reproduction_admitted=False)
        out.mkdir(parents=True)
        write(out / 'protocol.json', plan)
        print('Declared 15 fresh 105M founders and 360 assessments; no CUDA execution.', flush=True)
        return plan
    finally:
        if gate:
            gate.close()


def legacy_check(plan, case, directory, assessments):
    if case['seed'] != 1337 or case['arm']['name'] != 'control':
        return None
    for stage in ('initial', '1', '2'):
        path = directory / ('initial.ckpt' if stage == 'initial' else f'stage-{stage}.ckpt')
        require(file_hash(path) == plan['legacy_control'][stage]['sha256'],
                'Seed-1337 disabled control differs from the archived checkpoint: ' + stage)
    baseline = read(plan['specification']['legacy_control_result'])
    for current in assessments:
        previous = next(r for r in baseline['rows'] if r['profile'] == '105m' and r['stage'] == current['stage'])
        original_books = [r for r in current['books'] if r['role'] != 'stage_two_training']
        require(original_books == previous['books'] and current['samples'] == previous['samples'] and
                current['validation_mean_nats_per_byte'] == previous['validation_mean_nats_per_byte'],
                'Seed-1337 disabled control assessment differs from the preserved baseline')
    return dict(passed=True, three_complete_checkpoints_exact=True,
                original_six_book_scores_and_four_samples_exact_at_both_endpoints=True)


def comparison(rows):
    result = []
    for row in rows:
        control = next(r for r in rows if r['seed'] == row['seed'] and r['arm']['name'] == 'control')
        first, final = row['assessments']
        before = {r['book']: r for r in first['books']}
        baseline = {r['book']: r for r in control['assessments'][-1]['books']}
        result.append(dict(seed=row['seed'], arm=row['arm']['name'],
            final_validation_mean=final['validation_mean_nats_per_byte'],
            validation_difference_from_same_seed_control=final['validation_mean_nats_per_byte'] -
                control['assessments'][-1]['validation_mean_nats_per_byte'],
            books=[dict(book=r['book'], role=r['role'], final_loss=r['loss_nats_per_byte'],
                change_from_stage_one=r['loss_nats_per_byte'] - before[r['book']]['loss_nats_per_byte'],
                final_difference_from_control=r['loss_nats_per_byte'] - baseline[r['book']]['loss_nats_per_byte'])
                for r in final['books']]))
    return result


def execute(out, protocol_sha256):
    out = Path(out).resolve()
    require(file_hash(out / 'protocol.json') == protocol_sha256, 'Learning declaration changed')
    require(not (out / 'execution.json').exists(), 'Study already started; preserve its evidence')
    plan = read(out / 'protocol.json')
    spec = plan['specification']
    validate(spec)
    require(plan['version'] == 'membrane-learning-plan-v1' and plan['status'] == 'declared_before_learning' and
            Path(plan['source_checkout']) == ROOT and Path(plan['workspace']) == Path.cwd().resolve(),
            'Unexpected study checkout or workspace')
    require(plan['cases'] == cases(spec, out, Path(plan['runtime']), Path(plan['evaluation']['schedule'])) and
            plan['learning_commands_planned'] == 15 and plan['assessment_commands_planned'] == 360,
            'Study command plan changed')
    authenticate(plan['authenticated_inputs'])
    gate, rows = None, []
    try:
        waiting = plan['wait_for_acceptance_driver']
        gate = ProcessGate(waiting['pid'], waiting['executable'])
        require(gate.identity == waiting, 'Acceptance driver process identity changed')
        write(out / 'execution.json', dict(protocol_sha256=protocol_sha256, phase='waiting-for-acceptance',
              native_work_started=False, acceptance_driver=waiting))
        print('Waiting for complete candidate acceptance:', waiting['pid'], flush=True)
        code = gate.wait()
        write(out / 'acceptance-exit.json', dict(**gate.identity, exit_code=code))
        gate.close(); gate = None
        authenticate(plan['authenticated_inputs'])
        verifier = [plan['python'], '-X', 'utf8', str(ROOT / 'scripts/verify_membrane_acceptance.py'),
                    '--acceptance', plan['acceptance'], '--protocol-sha256', plan['acceptance_protocol_sha256'],
                    '--output', str(out / 'acceptance-verified.json')]
        write(out / 'acceptance-verifier-command.json', verifier)
        write(out / 'execution.json', dict(protocol_sha256=protocol_sha256, phase='verify-acceptance',
              native_work_started=False, acceptance_driver=waiting))
        verified = subprocess.run(verifier, cwd=plan['workspace'], capture_output=True)
        (out / 'acceptance-verifier.log').write_bytes(verified.stdout + verified.stderr)
        require(verified.returncode == 0, 'Candidate acceptance verification failed; no learning started')
        accepted = read(out / 'acceptance-verified.json')
        require(accepted['passed'] and accepted['candidate_runtime'] == plan['runtime'] and
                accepted['candidate_runtime_sha256'] == file_hash(plan['runtime']), 'Verified runtime differs')
        command_dir = out / 'learning-commands'
        command_dir.mkdir()
        native = NativeCommands(plan['runtime'], command_dir)
        seed_initial, seed_endpoints, assessment_commands = {}, {}, []
        additional = [('stage_two_training', spec['stage_two_training_monitors'])]
        for index, case in enumerate(plan['cases']):
            authenticate(plan['authenticated_inputs'])
            budget = storage(out, len(plan['cases']) - index)
            directory = Path(case['directory'])
            require(not directory.exists(), 'Founder output already exists')
            directory.mkdir()
            write(directory / 'protocol.json', dict(study_protocol_sha256=protocol_sha256, case=case,
                  storage_before_learning=budget, random_initialization=True, imported_weights=False))
            write(out / 'execution.json', dict(protocol_sha256=protocol_sha256, phase='learning',
                  native_work_started=True, case_index=index, case=directory.name, completed_models=len(rows)))
            print('Learning', directory.name, 'through both source stages.', flush=True)
            native(*case['argv'][1:])
            require(native.commands[-1] == case['argv'], 'Actual learning command differs')
            initial = read_state(directory / 'initial.ckpt')
            initial_expected = dict(online_updates=0, global_updates=0, observed_pairs=0, generated_bytes=0,
                                    replay_updates=0, curriculum_stage=1)
            verify_state(initial, spec, case['arm'], case['seed'], initial_expected)
            fingerprint = initial_fingerprint(directory / 'initial.ckpt', initial)
            require(seed_initial.setdefault(case['seed'], fingerprint) == fingerprint,
                    'Objective arms did not start with identical neural state')
            session = read(directory / 'session.json')
            metrics = audit_metrics(directory / 'metrics.jsonl', plan['expected_exposure'],
                                    clip=1, membrane_cost=case['arm']['membrane_cost'])
            write(directory / 'metrics-audit.json', metrics)
            assessments = []
            for stage, end in enumerate(spec['endpoints'], 1):
                checkpoint = directory / f'stage-{stage}.ckpt'
                state = read_state(checkpoint)
                verify_state(state, spec, case['arm'], case['seed'], plan['expected_exposure'][str(end)])
                key = (case['seed'], stage)
                matched_exposure(seed_endpoints.setdefault(key, state), state)
                identity = file_hash(checkpoint)
                destination = directory / f'assessment-{stage}'
                destination.mkdir()
                assessment = NativeCommands(plan['runtime'], destination)
                write(out / 'execution.json', dict(protocol_sha256=protocol_sha256, phase='assessment',
                      native_work_started=True, case_index=index, case=directory.name, stage=stage,
                      completed_models=len(rows)))
                measured = measure(assessment, checkpoint, plan['evaluation'], destination, additional)
                require(file_hash(checkpoint) == identity, 'Assessment modified the checkpoint')
                row = dict(complete=True, checkpoint=str(checkpoint), checkpoint_sha256=identity,
                    seed=case['seed'], arm=case['arm']['name'], stage=stage, online_updates=end,
                    state=state, **measured, checkpoint_unchanged=True, reserved_tests_scored=False)
                commands, hashes = verify_saved(destination, row, plan['evaluation'],
                    state['counters']['global_updates'], additional, executable=Path(plan['runtime']))
                require(len(commands) == 12, 'Incomplete endpoint assessment')
                assessment_commands.extend(commands)
                row['artifact_sha256'] = hashes
                write(destination / 'result.json', row)
                assessments.append(row)
            final = assessments[-1]
            require(file_hash(directory / 'latest.ckpt') == final['checkpoint_sha256'], 'Final saved states differ')
            require(all(session[k] == v for k, v in final['state']['counters'].items()) and
                    session['session_online_updates'] == spec['endpoints'][-1] and
                    session['consolidation'] == 'none' and session['shared_weights'] and
                    session['persistent_membranes'] and not session['trains_on_generated_text'],
                    'Native session differs from the completed founder')
            exact_legacy = legacy_check(plan, case, directory, assessments)
            record = dict(seed=case['seed'], arm=case['arm'], directory=str(directory),
                initial_fingerprint=fingerprint, session=session, metrics=metrics, assessments=assessments,
                legacy_control=exact_legacy, reserved_tests_scored=False, reproduction_admitted=False)
            rows.append(record)
            write(directory / 'result.json', dict(complete=True, **record))
            write(out / 'completed-models.json', rows)
            print(directory.name, 'final validation:', final['validation_mean_nats_per_byte'], flush=True)
        authenticate(plan['authenticated_inputs'])
        require(len(rows) == len(native.commands) == 15 and len(assessment_commands) == 360 and
                rows[0]['legacy_control']['passed'], 'Incomplete study')
        result = dict(complete=True, protocol_sha256=protocol_sha256, protocol=plan, rows=rows,
            paired_comparisons=comparison(rows), native_learning_commands=15, native_assessment_commands=360,
            exact_archived_control=True, same_initial_neural_state_within_seed=True,
            matched_source_replay_and_speech_exposure=True, acceptance_verification_sha256=file_hash(out / 'acceptance-verified.json'),
            reserved_tests_scored=False, reproduction_admitted=False, limits=spec['limits'])
        write(out / 'result.json', result)
        write(out / 'execution.json', dict(protocol_sha256=protocol_sha256, phase='complete',
              native_work_started=True, completed_models=15))
        print('Objective comparison complete; every setting and seed retained, no model promotion.', flush=True)
    except BaseException as error:
        write(out / 'failure.json', dict(error=str(error), completed_models=len(rows), artifacts_preserved=True))
        raise
    finally:
        if gate:
            gate.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest='command', required=True)
    create = commands.add_parser('declare')
    create.add_argument('--out', type=Path, required=True)
    create.add_argument('--workspace', type=Path, required=True)
    create.add_argument('--acceptance', type=Path, required=True)
    create.add_argument('--wait-pid', type=int, required=True)
    create.add_argument('--wait-executable', type=Path, required=True)
    run = commands.add_parser('run')
    run.add_argument('--out', type=Path, required=True)
    run.add_argument('--protocol-sha256', required=True)
    args = parser.parse_args()
    if args.command == 'declare':
        declare(args.out, args.workspace, args.acceptance, args.wait_pid, args.wait_executable)
    else:
        execute(args.out, args.protocol_sha256)
