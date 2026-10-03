"""Fresh larger-reservoir controls using the existing native founder and assessments."""
import argparse
import hashlib
from pathlib import Path
import subprocess

from experiment_checkpoint import policy_checkpoint
from native_experiment import read
from process_gate import ProcessGate
from prose_evaluation import assess, declare as declare_assessment
from prose_founder import file_hash, run as train_founder, write
from prose_retention_inputs import authenticate, predecessors, require
from prose_size_comparison import counters, require_complete


ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / 'data/prose-replay-capacity-v1.json'
MATCHED = ('online_updates', 'global_updates', 'observed_pairs', 'generated_bytes', 'replay_updates')


def array_hash(path, count):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        stream.seek(256)  # Hyperparameters, weights, moments and live recurrence.
        left = 32 + 12 * count[14] + 4 * count[18]
        while left:
            block = stream.read(min(left, 8 * 1024 * 1024))
            require(bool(block), 'Truncated initial arrays')
            digest.update(block)
            left -= len(block)
    return digest.hexdigest()


def initial_identity(original, candidate, before_capacity, after_capacity):
    old, old_extra = policy_checkpoint(original)
    new, new_extra = policy_checkpoint(candidate)
    require(old[7] == old[24] == new[7] == new[24] == 0, 'Expected untouched random initial checkpoints')
    require(all(a == b for i, (a, b) in enumerate(zip(old, new)) if i != 15), 'Initial model metadata differ')
    require(old_extra[3] == before_capacity and new_extra[3] == after_capacity, 'Initial capacity differs')
    require(len(old_extra) == len(new_extra) and
            all(a == b for i, (a, b) in enumerate(zip(old_extra, new_extra)) if i != 3),
            'Other initial replay policy differs')
    original_arrays, candidate_arrays = array_hash(original, old), array_hash(candidate, new)
    require(original_arrays == candidate_arrays, 'Initial parameters, optimizer, recurrence or hyperparameters differ')
    return dict(original_checkpoint=original.as_posix(), candidate_checkpoint=candidate.as_posix(),
                original_checkpoint_sha256=file_hash(original), candidate_checkpoint_sha256=file_hash(candidate),
                identical_initial_arrays_sha256=original_arrays,
                only_capacity_and_payload_checksum_differ=True)


def declare(out):
    spec = read(SPEC)
    require(spec['version'] == 'prose-replay-capacity-v1' and spec['profiles'] == ['2m', '105m'] and
            spec['seed'] == 1337 and spec['candidate_capacity'] == 16384 and spec['baseline_capacity'] == 1024 and
            spec['source_observations'] == 216289 and spec['learning_rate'] == .0003 and spec['replay_every'] == 4,
            'Unexpected replay-capacity v1 policy')
    out.mkdir(parents=True, exist_ok=False)
    assessment = out / 'assessment'
    declare_assessment(assessment)
    evaluation = read(assessment / 'protocol.json')
    files = [SPEC, assessment / 'protocol.json', ROOT / 'data/training-selection.json',
             Path('runs/prose-size-panel/protocol.json'), Path('runs/prose-spike-panel/protocol.json'),
             Path('runs/prose-retention-lr/protocol.json'), Path('reports/prose-replay-coverage.json'),
             *(ROOT / 'scripts' / n for n in ('prose_replay_capacity.py', 'prose_founder.py', 'prose_evaluation.py',
                 'checkpoint_assessment.py', 'prose_size_comparison.py', 'prose_retention_inputs.py',
                 'experiment_checkpoint.py', 'native_experiment.py', 'process_gate.py', 'audit_binding.py', 'corpus/selection.py'))]
    pins = dict(evaluation['authenticated_inputs'])
    pins.update({p.as_posix(): file_hash(p) for p in files})
    plan = dict(status='declared_before_replay_capacity_comparison', specification=spec,
        source_commit=subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
        source_worktree=ROOT.as_posix(), assessment_plan=(assessment / 'protocol.json').as_posix(),
        authenticated_inputs=pins, native_learning_commands_planned=2, native_assessment_commands_planned=80,
        no_additional_replay_scoring=True, generated_text_targets=False,
        reserved_tests_scored=False, reproduction_admitted=False)
    write(out / 'protocol.json', plan)
    print('Declared fresh 2M and 105M founders with 16384 replay slots; four assessments per size.', flush=True)


def comparison_row(candidate, baseline):
    current, previous = candidate['exposure_counters'], baseline['exposure_counters']
    require(all(current[k] == previous[k] for k in MATCHED), 'Source, update or speech exposure differs')
    require(candidate['profile'] == baseline['profile'] and candidate['stage'] == baseline['stage'] and
            candidate['parameters'] == baseline['parameters'], 'Capacity controls have different shapes/endpoints')
    old = {r['book']: r for r in baseline['books']}
    return dict(profile=candidate['profile'], stage=candidate['stage'], source_observations=current['online_updates'],
        validation_mean_nats_per_byte=candidate['validation_mean_nats_per_byte'],
        validation_difference_from_1024=candidate['validation_mean_nats_per_byte'] - baseline['validation_mean_nats_per_byte'],
        replay_pairs=current['replay_pairs'], replay_pair_difference=current['replay_pairs'] - previous['replay_pairs'],
        books=[dict(book=r['book'], role=r['role'], loss_nats_per_byte=r['loss_nats_per_byte'],
                    difference_from_1024=r['loss_nats_per_byte'] - old[r['book']]['loss_nats_per_byte'])
               for r in candidate['books']])


def execute(out, wait_pid, wait_executable):
    plan = read(out / 'protocol.json')
    require(plan['status'] == 'declared_before_replay_capacity_comparison' and
            not (out / 'execution.json').exists(), 'Study undeclared or already started')
    authenticate(plan['authenticated_inputs'])
    spec = plan['specification']
    gate = ProcessGate(wait_pid, wait_executable) if wait_pid else None
    rows, initial, completed = [], [], []
    try:
        write(out / 'execution.json', dict(protocol_sha256=file_hash(out / 'protocol.json'),
              native_predecessor=gate.identity if gate else None, native_work_started=False))
        if gate:
            print('Waiting behind the learning-rate comparison:', gate.identity, flush=True)
            code = gate.wait()
            write(out / 'predecessor-exit.json', dict(**gate.identity, exit_code=code))
        authenticate(plan['authenticated_inputs'])
        prior = predecessors(Path(spec['baseline_comparison']), Path(spec['neuron_panel']))
        rate = read(spec['learning_rate_comparison'])
        require(rate['complete'] and rate['control_checkpoint_byte_identical'] and
                rate['matched_source_replay_and_speech_counters'] and rate['matched_replay_payloads'] and
                rate['native_commands'] == 64 and not rate['reserved_tests_scored'], 'Learning-rate predecessor is incomplete')
        require(rate['protocol'] == read('runs/prose-retention-lr/protocol.json'), 'Learning-rate predecessor protocol differs')
        prior[spec['learning_rate_comparison']] = file_hash(spec['learning_rate_comparison'])
        baseline = read(spec['baseline_comparison'])
        require(baseline['protocol'] == read('runs/prose-size-panel/protocol.json') and
                read(spec['neuron_panel'])['protocol'] == read('runs/prose-spike-panel/protocol.json'),
                'Predecessor protocol differs')
        write(out / 'predecessors-complete.json', prior)
        assessment_plan = Path(plan['assessment_plan'])
        evaluation = read(assessment_plan)
        baseline_pins = {}
        for profile in spec['profiles']:
            authenticate(plan['authenticated_inputs'])
            directory = out / f'founder-{profile}'
            print('Starting fresh', profile, 'with', spec['candidate_capacity'], 'replay slots.', flush=True)
            train_founder(directory, spec['source_observations'], profile, spec['candidate_capacity'])
            require_complete(directory, evaluation)
            original = Path(baseline['protocol']['original_founder']) if profile == '105m' else Path(spec['baseline_comparison']).parent / f'founder-{profile}'
            identity = initial_identity(original / 'initial.ckpt', directory / 'initial.ckpt',
                                        spec['baseline_capacity'], spec['candidate_capacity'])
            baseline_pins[identity['original_checkpoint']] = identity['original_checkpoint_sha256']
            initial.append(dict(profile=profile, **identity))
            for stage in (1, 2, 3, 4):
                destination = out / f'{profile}-stage-{stage}'
                assess(assessment_plan, directory, stage, destination)
                row = read(destination / 'result.json')
                checkpoint = directory / f'stage-{stage}.ckpt'
                meta, extra = policy_checkpoint(checkpoint)
                require(extra[3] == spec['candidate_capacity'], 'Candidate replay capacity changed')
                row['exposure_counters'] = counters(checkpoint)
                row['replay_policy_sha256'] = hashlib.sha256(bytes_of_policy(extra)).hexdigest()
                control = next(r for r in baseline['rows'] if r['profile'] == profile and r['stage'] == stage)
                require(file_hash(control['checkpoint']) == control['checkpoint_sha256'], 'Baseline checkpoint changed')
                baseline_pins[control['checkpoint']] = control['checkpoint_sha256']
                row['comparison'] = comparison_row(row, control)
                rows.append(row)
                write(out / 'partial.json', dict(rows=rows, initial=initial, completed=completed))
            completed.append(dict(profile=profile, result=read(directory / 'result.json')))
            authenticate(plan['authenticated_inputs'])
        for stage in (1, 2, 3, 4):
            matched = [r for r in rows if r['stage'] == stage]
            require(matched[0]['replay_policy_sha256'] == matched[1]['replay_policy_sha256'],
                    'Candidate sizes received different replay histories')
        native_calls = sum(len(read(out / f'founder-{p}/commands/commands.json')) for p in spec['profiles'])
        assessment_calls = sum(len(read(out / f'{r["profile"]}-stage-{r["stage"]}/commands.json')) for r in rows)
        require(native_calls == plan['native_learning_commands_planned'] and
                assessment_calls == plan['native_assessment_commands_planned'], 'Native command counts differ')
        authenticate(prior)
        authenticate(baseline_pins)
        write(out / 'result.json', dict(complete=True, protocol=plan, rows=rows, initial=initial, completed=completed,
            native_learning_commands=native_calls, native_assessment_commands=assessment_calls,
            same_initial_model_arrays=True, same_candidate_replay_history_across_sizes=True,
            matched_source_update_and_speech_exposure=True, reserved_tests_scored=False,
            reproduction_admitted=False, timing_limit=spec['timing'], limits=spec['limits']))
        print('Replay-capacity comparison complete; no model automatically promoted.', flush=True)
    except Exception as error:
        write(out / 'failure.json', dict(error=str(error), completed_assessments=len(rows),
                                        partial_artifacts_preserved=True, candidate_promoted=False))
        raise
    finally:
        if gate:
            gate.close()


def bytes_of_policy(extra):
    import struct
    return struct.pack(f'<{len(extra)}Q', *extra)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    preparation = sub.add_parser('declare')
    preparation.add_argument('--out', type=Path, required=True)
    execution = sub.add_parser('run')
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
