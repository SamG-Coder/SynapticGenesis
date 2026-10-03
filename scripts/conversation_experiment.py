"""Ask, record an explicit correction, resume native learning, then ask anew."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys

from conversation_material import ROOT, SPEC, prepare
from conversation_measure import assess, contextual_control
from membrane_study_state import f32, read_state, require
from native_experiment import NativeCommands, read
from process_gate import ProcessGate
from prose_founder import file_hash, write
from quantitative_assessment_inputs import authenticate


def previous_complete(spec):
    directory = Path(spec['predecessor'])
    require(file_hash(directory / 'protocol.json') == spec['predecessor_protocol_sha256'] and
            not (directory / 'failure.json').exists(), 'Assessment predecessor failed or changed')
    result, execution = read(directory / 'result.json'), read(directory / 'execution.json')
    require(result['complete'] and result['protocol_sha256'] == spec['predecessor_protocol_sha256'] and
            result['native_commands'] == 18 and result['learning_commands'] == 0 and
            execution['phase'] == 'complete' and execution['completed_models'] == 2 and
            len(result['rows']) == 2 and all(row['complete'] for row in result['rows']),
            'Quantitative baseline has not completed')
    require(result['rows'][0]['checkpoint_sha256'] == spec['base_sha256'], 'Predecessor used a different prose base')
    return dict(complete=True, result_sha256=file_hash(directory / 'result.json'))


def learning_arguments(checkpoint, original, extension, out, endpoint, first):
    args = ['live', '--resume', str(checkpoint), '--curriculum', str(original),
            '--out', str(out), '--updates', str(endpoint), '--log-every', '1', '--save-every', '100000000']
    if first:
        args.extend(['--extend-curriculum', str(extension)])
    return args


def verify_learning(state, initial, material, passes, session):
    spec, base = material['specification'], initial['counters']
    updates = 6 * passes
    replays = (base['online_updates'] + updates) // initial['replay_every'] - base['online_updates'] // initial['replay_every']
    expected = dict(online_updates=base['online_updates'] + updates,
        global_updates=base['global_updates'] + updates + replays,
        observed_pairs=base['observed_pairs'] + passes * material['target_pairs_per_pass'],
        generated_bytes=base['generated_bytes'] + ((base['online_updates'] + updates) // 500 - base['online_updates'] // 500) * 96,
        replay_updates=base['replay_updates'] + replays, curriculum_stage=5)
    require(all(state['counters'][key] == value for key, value in expected.items()),
            'Correction exposure or inherited counters differ')
    require(state['parameters'] == initial['parameters'] and state['seed'] == initial['seed'] and
            state['graph'] == initial['graph'] and state['replay_every'] == initial['replay_every'] and
            state['replay_capacity'] == initial['replay_capacity'] and state['speech_policy'] == initial['speech_policy'] and
            state['hyperparameters'][1:] == initial['hyperparameters'][1:] and
            state['hyperparameters'][0] == f32(initial['hyperparameters'][7] * spec['rate_scale']),
            'Correction changed model shape or an undeclared learning policy')
    require(len(state['replay_groups']) == 5 and state['replay_groups'][-1][1] == updates and
            [g[1] for g in state['replay_groups'][:4]] == [g[1] for g in initial['replay_groups']],
            'Earlier exposure history or correction replay group differs')
    require(session['shared_weights'] and session['persistent_membranes'] and not session['trains_on_generated_text'] and
            session['answer_emphasized_documents'] == 6 and session['online_first_document'] == 55 and
            session['online_document_count'] == 6 and session['global_updates'] == expected['global_updates'],
            'Native correction session differs')
    return expected


def declare(out, wait_pid, wait_executable):
    out = Path(out).resolve()
    require(not out.exists(), 'Use a fresh conversation output directory')
    gate = ProcessGate(wait_pid, wait_executable)
    try:
        require(gate.kernel.WaitForSingleObject(gate.handle, 0) == 258, 'Declare while the assessment driver is live')
        material = prepare(out / 'material')
        spec = material['specification']
        pins = dict(material['authenticated_inputs'])
        pins.update({p.as_posix(): file_hash(p) for p in (
            out / 'material/manifest.json', ROOT / 'scripts/conversation_experiment.py',
            ROOT / 'scripts/conversation_measure.py', ROOT / 'tests/conversation_correction.py',
            Path(sys.executable).resolve(), Path(spec['predecessor']) / 'protocol.json')})
        base, schedule = material['base']['checkpoint'], out / 'material/curriculum/curriculum.sg'
        rounds = []
        for index, passes in enumerate(spec['passes']):
            directory = out / f'practice-{passes}'
            old = Path(spec['original_curriculum']) if index == 0 else schedule
            rounds.append(dict(passes=passes, directory=str(directory),
                arguments=learning_arguments(base, old, schedule, directory,
                    spec['base_online_updates'] + 6 * passes, index == 0)))
            base = directory / 'latest.ckpt'
        plan = dict(version='conversation-plan-v1', status='declared_before_conversation',
            created_utc=datetime.now(timezone.utc).isoformat(), source_checkout=str(ROOT),
            source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            workspace=str(Path.cwd().resolve()), material=material, rounds=rounds,
            runtime=read(ROOT / 'data/quantitative-assessment-v1.json')['runtime'],
            wait_for_driver=gate.identity, authenticated_inputs=pins,
            learning_commands_planned=3, assessment_commands_planned=94, reproduction_admitted=False)
        authenticate(pins)
        write(out / 'protocol.json', plan)
        print('Declared six corrections, cold-context checks after 1/16/256 passes, and four retention books.', flush=True)
        return plan
    finally:
        gate.close()


def execute(out, protocol_sha256):
    out = Path(out).resolve()
    require(file_hash(out / 'protocol.json') == protocol_sha256 and not (out / 'execution.json').exists(),
            'Conversation declaration changed or already started')
    plan = read(out / 'protocol.json')
    require(plan['version'] == 'conversation-plan-v1' and plan['source_checkout'] == str(ROOT) and
            plan['workspace'] == str(Path.cwd().resolve()) and plan['material']['specification'] == read(SPEC),
            'Conversation checkout, workspace or policy differs')
    authenticate(plan['authenticated_inputs'])
    gate, results, started = None, [], False
    try:
        identity = plan['wait_for_driver']
        gate = ProcessGate(identity['pid'], identity['executable'])
        require(gate.identity == identity, 'Predecessor process identity changed')
        write(out / 'execution.json', dict(phase='waiting-for-baseline', native_work_started=False,
            protocol_sha256=protocol_sha256, predecessor=identity))
        print('Waiting on the held assessment-driver handle:', identity['pid'], flush=True)
        gate.wait()
        write(out / 'predecessor-exit.json', dict(**gate.identity, exit_code=0))
        gate.close(); gate = None
        material = plan['material']; spec = material['specification']
        predecessor = previous_complete(spec)
        write(out / 'predecessor-completion.json', predecessor)
        authenticate(plan['authenticated_inputs'])
        write(out / 'execution.json', dict(phase='before-correction', native_work_started=True,
            protocol_sha256=protocol_sha256))
        started = True
        baseline = assess(plan['runtime'], spec['base'], material, out / 'before')
        contextual = contextual_control(plan['runtime'], spec['base'], material, out / 'context-only')
        dialogue = [dict(question=lesson['question'], model_before=answer,
            feedback_kind='confirm' if answer['first_line_matches'] else 'correct',
            teacher_answer=lesson['answer'], training_target_source='reviewed correction edition')
            for lesson, answer in zip(material['source']['lessons'], baseline['answers'][:6])]
        write(out / 'dialogue.json', dialogue)
        checkpoint = spec['base']
        schedule = out / 'material/curriculum/curriculum.sg'
        for index, declared in enumerate(plan['rounds']):
            passes, directory = declared['passes'], Path(declared['directory'])
            require(directory == out / f'practice-{passes}' and passes == spec['passes'][index],
                    'Declared practice endpoint differs')
            expected = learning_arguments(checkpoint, Path(spec['original_curriculum']) if index == 0 else schedule,
                schedule, directory, spec['base_online_updates'] + 6 * passes, index == 0)
            require(declared['arguments'] == expected, 'Declared learning command differs')
            directory.mkdir()
            native = NativeCommands(plan['runtime'], directory)
            write(out / 'execution.json', dict(phase='correction-learning', passes=passes,
                native_work_started=True, protocol_sha256=protocol_sha256, completed_rounds=len(results)))
            native(*expected)
            checkpoint = directory / 'latest.ckpt'
            state, session = read_state(checkpoint), read(directory / 'session.json')
            verify_learning(state, material['base']['state'], material, passes, session)
            measured = assess(plan['runtime'], checkpoint, material, out / f'after-{passes}')
            changes = [dict(book=now['book'], role=now['role'],
                loss_change_nats_per_byte=now['loss_nats_per_byte'] - old['loss_nats_per_byte'])
                for old, now in zip(baseline['books'], measured['books'])]
            record = dict(complete=True, passes=passes, session=session, assessment=measured,
                retention_changes=changes, checkpoint_sha256=file_hash(checkpoint))
            write(directory / 'result.json', record)
            results.append(record)
            write(out / 'completed-rounds.json', results)
            print('Direct correction passes:', passes, 'first-line matches:', measured['first_line_scores'], flush=True)
        authenticate(plan['authenticated_inputs'])
        require(file_hash(spec['base']) == spec['base_sha256'], 'Ancestor checkpoint changed')
        result = dict(complete=True, protocol_sha256=protocol_sha256, protocol=plan,
            baseline=baseline, contextual_control=contextual, dialogue=dialogue, rounds=results,
            native_learning_commands=3, native_assessment_commands=94,
            ancestor_unchanged=True, reserved_tests_scored=False, reproduction_admitted=False, limits=spec['limits'])
        write(out / 'result.json', result)
        write(out / 'execution.json', dict(phase='complete', native_work_started=True,
            protocol_sha256=protocol_sha256, completed_rounds=3))
        print('Direct correction experiment complete; all raw answers retained.', flush=True)
        return result
    except BaseException as error:
        write(out / 'failure.json', dict(error=str(error), completed_rounds=len(results),
            native_work_started=started, artifacts_preserved=True))
        raise
    finally:
        if gate:
            gate.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest='command', required=True)
    creation = commands.add_parser('declare')
    creation.add_argument('--out', required=True, type=Path)
    creation.add_argument('--wait-pid', required=True, type=int)
    creation.add_argument('--wait-executable', required=True, type=Path)
    execution = commands.add_parser('run')
    execution.add_argument('--out', required=True, type=Path)
    execution.add_argument('--protocol-sha256', required=True)
    args = parser.parse_args()
    if args.command == 'declare':
        declare(args.out, args.wait_pid, args.wait_executable)
    else:
        execute(args.out, args.protocol_sha256)
