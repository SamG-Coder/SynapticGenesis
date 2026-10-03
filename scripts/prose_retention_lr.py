"""Controlled late-stage learning-rate continuation through the native live loop."""
import argparse
from pathlib import Path
import struct
import subprocess
import time

from checkpoint_assessment import measure
from native_experiment import NativeCommands, read
from process_gate import ProcessGate
from prose_founder import file_hash, write
from prose_retention_inputs import (COUNTERS, ROOT, SPEC, admit, authenticate, matched,
                                   predecessors, require, state)


def declare(out):
    plan = admit(read(SPEC))
    code = [ROOT / 'scripts' / name for name in ('prose_retention_lr.py', 'prose_retention_inputs.py',
            'checkpoint_assessment.py', 'native_experiment.py', 'audit_binding.py', 'process_gate.py',
            'prose_founder.py', 'prose_size_comparison.py', 'prose_evaluation.py', 'corpus/selection.py')]
    plan['authenticated_inputs'].update({p.as_posix(): file_hash(p) for p in code})
    plan.update(status='declared_before_learning_rate_comparison',
        source_commit=subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
        source_worktree=ROOT.as_posix(), native_commands_planned=64,
        learning_commands_planned=4, assessment_commands_planned=60,
        predecessors=['runs/prose-size-panel/comparison.json', 'runs/prose-spike-panel/result.json'],
        generated_text_targets=False, reserved_tests_scored=False, reproduction_admitted=False,
        policy='Wait for successful process exit and complete size/neuron reports, authenticate all inputs, '
               'then run parent assessment, exact resumed control and quarter-rate continuation sequentially. '
               'Each arm starts from the same immutable parent with all optimizer, recurrence and replay history. '
               'Two native sessions per arm provide matched halfway and final assessment checkpoints. '
               'Stop on incomplete exposure, mismatched replay, altered inputs or a non-identical control.')
    out.mkdir(parents=True, exist_ok=False)
    write(out / 'protocol.json', plan)
    print('Declared two rates, two endpoints per arm, eight book monitors and four raw prompts.', flush=True)


def learning_args(spec, checkpoint, directory, end, rate):
    # Native resume owns all inherited controls. Explicitly override only LR.
    return ('live', '--resume', checkpoint, '--curriculum', spec['schedule'], '--out', directory,
            '--updates', end, '--lr', rate, '--prompt', spec['prompt'],
            '--log-every', spec['log_every'], '--save-every', spec['save_every'])


def check_endpoint(plan, checkpoint, end, rate, session=None):
    record = state(checkpoint)
    require(all(record['counters'][k] == v for k, v in plan['expected_exposure'][str(end)].items()),
            'Native continuation did not reach the declared exposure')
    exact_rate = struct.unpack('<f', struct.pack('<f', rate))[0]
    require(record['hyperparameters'][0] == record['hyperparameters'][7] == exact_rate,
            'Native checkpoint rate differs from the declared arm')
    parent = plan['parent_state']
    for key in ('parameters', 'channels', 'hidden', 'layers'):
        require(record['counters'][key] == parent['counters'][key], 'Continuation model shape changed')
    for key in ('seed', 'batch', 'chunk', 'fast', 'speech_policy', 'graph', 'replay_every', 'replay_capacity'):
        require(record[key] == parent[key], f'Continuation changed inherited policy: {key}')
    require(record['hyperparameters'][1:7] == parent['hyperparameters'][1:7], 'Inherited optimizer policy changed')
    if session is not None:
        require(all(session[k] == record['counters'][k] for k in COUNTERS), 'Session and checkpoint counters differ')
        require(session['shared_weights'] and session['persistent_membranes'] and
                not session['trains_on_generated_text'] and session['gradient_reductions'] == 'ordered_v1',
                'Shared live-learning behavior differs')
    return record


def assessment(native, plan, checkpoint, destination, label):
    identity = file_hash(checkpoint)
    destination.mkdir(parents=True, exist_ok=False)
    result = measure(native, checkpoint, plan['evaluation'], destination,
                     [('new_stage_training', plan['additional_books'])])
    require(file_hash(checkpoint) == identity, 'Assessment changed the checkpoint')
    result.update(label=label, checkpoint=checkpoint.as_posix(), checkpoint_sha256=identity,
                  state=state(checkpoint), reserved_tests_scored=False, checkpoint_unchanged=True)
    write(destination / 'result.json', result)
    print(label, 'validation mean', result['validation_mean_nats_per_byte'], flush=True)
    return result


def differences(rows):
    parent = rows[0]
    base = {r['book']: r for r in parent['books']}
    result = []
    for row in rows[1:]:
        end = row['state']['counters']['online_updates']
        control = next(r for r in rows if r.get('arm') == 'resumed-control' and
                       r['state']['counters']['online_updates'] == end)
        reference = {r['book']: r for r in control['books']}
        result.append(dict(arm=row['arm'], online_updates=end,
            validation_mean_nats_per_byte=row['validation_mean_nats_per_byte'],
            validation_change_from_parent=row['validation_mean_nats_per_byte'] - parent['validation_mean_nats_per_byte'],
            validation_difference_from_control=row['validation_mean_nats_per_byte'] - control['validation_mean_nats_per_byte'],
            books=[dict(book=b['book'], role=b['role'], loss_nats_per_byte=b['loss_nats_per_byte'],
                        change_from_parent=b['loss_nats_per_byte'] - base[b['book']]['loss_nats_per_byte'],
                        difference_from_control=b['loss_nats_per_byte'] - reference[b['book']]['loss_nats_per_byte'])
                   for b in row['books']]))
    return result


def execute(out, wait_pid, wait_executable):
    plan_path = out / 'protocol.json'
    plan = read(plan_path)
    require(plan['status'] == 'declared_before_learning_rate_comparison', 'Undeclared learning-rate study')
    require(not (out / 'execution.json').exists(), 'This study has already started; use preserved evidence')
    authenticate(plan['authenticated_inputs'])
    spec = plan['specification']
    gate = ProcessGate(wait_pid, wait_executable) if wait_pid else None
    native, rows, sessions = None, [], []
    try:
        write(out / 'execution.json', dict(protocol_sha256=file_hash(plan_path),
              native_predecessor=gate.identity if gate else None, native_work_started=False))
        if gate:
            print('Waiting behind the matched neuron panel:', gate.identity, flush=True)
            code = gate.wait()
            write(out / 'predecessor-exit.json', dict(**gate.identity, exit_code=code))
        authenticate(plan['authenticated_inputs'])
        prior = predecessors(*(Path(p) for p in plan['predecessors']))
        # Bind the completed output to the predecessor protocols declared here.
        require(read(plan['predecessors'][0])['protocol'] == read('runs/prose-size-panel/protocol.json') and
                read(plan['predecessors'][1])['protocol'] == read('runs/prose-spike-panel/protocol.json'),
                'Predecessor completion does not belong to the declared protocol')
        write(out / 'predecessors-complete.json', prior)
        commands = out / 'commands'
        commands.mkdir()
        native = NativeCommands('build/synapticgenesis.exe', commands)
        parent = assessment(native, plan, Path(spec['checkpoint']), out / 'assessments/parent', 'parent')
        original_parent = plan['parent_assessment']
        require([b for b in parent['books'] if b['role'] != 'new_stage_training'] == original_parent['books'] and
                parent['samples'] == original_parent['samples'] and
                parent['validation_mean_nats_per_byte'] == original_parent['validation_mean_nats_per_byte'],
                'Fresh parent assessment differs from the published native assessment')
        rows.append(parent)
        for arm in spec['arms']:
            authenticate(plan['authenticated_inputs'])
            checkpoint = Path(spec['checkpoint'])
            previous = spec['starting_observations']
            for end in spec['endpoints']:
                directory = out / arm['name'] / f'through-{end}'
                require(not directory.exists(), 'Continuation output already exists')
                print('Learning', arm['name'], 'from', previous, 'through', end, flush=True)
                started = time.perf_counter()
                native(*learning_args(spec, checkpoint, directory, end, arm['learning_rate']))
                process_seconds = time.perf_counter() - started
                final = end == spec['endpoints'][-1]
                checkpoint = directory / ('stage-4.ckpt' if final else 'latest.ckpt')
                session = read(directory / 'session.json')
                record = check_endpoint(plan, checkpoint, end, arm['learning_rate'], session)
                require(session['session_online_updates'] == end - previous, 'Session source update count differs')
                require(session['session_observed_pairs'] == plan['expected_exposure'][str(end)]['observed_pairs'] -
                        plan['expected_exposure'][str(previous)]['observed_pairs'], 'Session byte count differs')
                if final:
                    require(file_hash(directory / 'latest.ckpt') == file_hash(checkpoint), 'Final checkpoint copies differ')
                if arm['name'] == 'resumed-control' and final:
                    require(file_hash(checkpoint) == plan['original_completion']['checkpoint_sha256'],
                            'Resumed control is not byte-identical to the original final checkpoint; candidate not started')
                    write(out / 'control-identity.json', dict(passed=True,
                          original_sha256=plan['original_completion']['checkpoint_sha256'],
                          resumed_sha256=file_hash(checkpoint), full_checkpoint_byte_identical=True))
                if arm['name'] != 'resumed-control':
                    control = next(r for r in rows if r.get('arm') == 'resumed-control' and
                                   r['state']['counters']['online_updates'] == end)
                    matched(record, control['state'])
                row = assessment(native, plan, checkpoint, out / 'assessments' / f'{arm["name"]}-{end}',
                                 f'{arm["name"]}-{end}')
                row['arm'], row['learning_rate'] = arm['name'], arm['learning_rate']
                rows.append(row)
                sessions.append(dict(arm=arm['name'], endpoint=end, native_session=session,
                                     complete_native_process_seconds=process_seconds))
                write(out / 'partial.json', dict(assessments=rows, sessions=sessions))
                authenticate(plan['authenticated_inputs'])
                previous = end
        authenticate(prior)
        require(len(native.commands) == plan['native_commands_planned'], 'Native command count differs')
        require(sum(c[1] == 'live' for c in native.commands) == plan['learning_commands_planned'], 'Learning command count differs')
        write(out / 'result.json', dict(complete=True, protocol=plan, assessments=rows, sessions=sessions,
              differences=differences(rows), native_commands=len(native.commands),
              parent_assessment_exact=True, control_checkpoint_byte_identical=True,
              matched_source_replay_and_speech_counters=True, matched_replay_payloads=True,
              reserved_tests_scored=False, reproduction_admitted=False,
              timing_limit='Native session time includes source learning, replay, speech and periodic/final saves. '
                           'Complete process time additionally includes loading and startup. Assessment time is separate. '
                           'Rates run sequentially on one seed; this is not a controlled speed benchmark.',
              limits=spec['limits']))
        print('Learning-rate continuation comparison complete; no model automatically promoted.', flush=True)
    except Exception as error:
        write(out / 'failure.json', dict(error=str(error), completed_assessments=len(rows),
              attempted_native_commands=len(native.commands) if native else 0,
              partial_artifacts_preserved=True, candidate_promoted=False))
        raise
    finally:
        if gate:
            gate.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subs = parser.add_subparsers(dest='command', required=True)
    declaration = subs.add_parser('declare')
    declaration.add_argument('--out', type=Path, required=True)
    execution = subs.add_parser('run')
    execution.add_argument('--out', type=Path, required=True)
    execution.add_argument('--wait-pid', type=int, default=0)
    execution.add_argument('--wait-executable', type=Path)
    args = parser.parse_args()
    if args.command == 'declare':
        declare(args.out)
    else:
        if args.wait_pid and args.wait_executable is None:
            parser.error('--wait-executable is required with --wait-pid')
        execute(args.out, args.wait_pid, args.wait_executable)
