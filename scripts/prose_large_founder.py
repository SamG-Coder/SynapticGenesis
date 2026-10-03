"""Preflight, train and assess a saved 411M founder through the shared runtime."""
import argparse
import math
from pathlib import Path
import subprocess

from checkpoint_assessment import measure, verify_saved
from large_founder_inputs import ROOT, admit, facts, first_stage_pairs, storage_budget
from native_experiment import NativeCommands, read
from prose_founder import file_hash, run as train_founder, write
from prose_retention_inputs import authenticate, require


def native_preflight(out):
    plan = admit()
    spec, evaluation = plan['specification'], plan['evaluation']
    resource = spec['native_preflight']
    plan.update(status='declared_before_411m_checkpoint_preflight', storage=storage_budget(out),
                fresh_random_diagnostic_only=True, native_commands_planned=4,
                source_commit=subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip())
    out.mkdir(parents=True, exist_ok=False)
    write(out / 'protocol.json', plan)
    try:
        print('411M preflight: fresh learning and full checkpoint save.', flush=True)
        fresh = out / 'fresh'
        train_founder(fresh, resource['fresh_source_observations'], spec['profile'], spec['replay_capacity'],
                      executable=Path(spec['runtime']), save_every=spec['save_every'])
        before = facts(fresh / 'latest.ckpt')
        journal = out / 'commands'
        journal.mkdir()
        native = NativeCommands(spec['runtime'], journal)
        resumed = out / 'resumed'
        print('411M preflight: reload optimizer/state and perform another update.', flush=True)
        native('live', '--resume', fresh / 'latest.ckpt', '--curriculum', evaluation['schedule'],
               '--out', resumed, '--updates', resource['resumed_source_observations'],
               '--prompt', 'The bird ', '--log-every', 512, '--save-every', spec['save_every'])
        checkpoint = resumed / 'latest.ckpt'
        identity = file_hash(checkpoint)
        after = facts(checkpoint)
        for row, end in [(before, 128), (after, 129)]:
            require(row['counters']['online_updates'] == end and
                    row['counters']['global_updates'] == end + end // 4 and
                    row['counters']['observed_pairs'] == first_stage_pairs(
                        plan['exposure']['stages'][0]['cumulative_source'], end) and
                    row['counters']['replay_updates'] == end // 4 and
                    row['counters']['generated_bytes'] == 0, 'Preflight exposure differs')
            require((row['counters']['channels'], row['counters']['hidden'], row['counters']['layers']) ==
                    tuple(spec['shape']) and row['counters']['parameters'] == spec['parameters'], 'Preflight shape differs')
            require((row['seed'], row['batch'], row['chunk'], row['fast'], row['graph'],
                     row['replay_every'], row['replay_capacity'], row['hyperparameters'][6]) ==
                    (1337, 1, 128, 1, 1, 4, 16384, 1.), 'Preflight policy differs')
        require(before['hyperparameters'] == after['hyperparameters'], 'Resume changed hyperparameters')
        print('411M preflight: original 16 x 128 strict-FP32 evaluation shape.', flush=True)
        book = Path(evaluation['prepared']) / f"{evaluation['validation_books'][0]}.txt"
        score = out / 'evaluation.json'
        native('evaluate', '--checkpoint', checkpoint, '--data', book, '--batch', resource['evaluation_batch'],
               '--context', resource['evaluation_context'], '--batches', resource['evaluation_batches'], '--output', score)
        observed = read(score)
        require(observed['evaluated_bytes'] == 4096 and observed['step'] == after['counters']['global_updates'] and
                math.isfinite(observed['loss_nats_per_byte']), 'Preflight evaluation differs')
        print('411M preflight: graph generation from the saved checkpoint.', flush=True)
        sample = out / 'sample.txt'
        native('sample', '--checkpoint', checkpoint, '--prompt', 'The bird ', '--tokens', resource['sample_bytes'],
               '--seed', 42, '--temperature', .8, '--top-k', 40, '--graph', '--output', sample)
        raw = sample.read_bytes()
        require(raw.startswith(b'The bird ') and len(raw) == 73, 'Preflight generation framing differs')
        require(file_hash(checkpoint) == identity, 'Evaluation changed the checkpoint')
        authenticate(plan['authenticated_inputs'])
        artifacts = [fresh / 'initial.ckpt', fresh / 'latest.ckpt', fresh / 'protocol.json', fresh / 'result.json',
                     fresh / 'session.json', fresh / 'commands/commands.json', fresh / 'commands/command-001.log',
                     resumed / 'latest.ckpt', resumed / 'session.json', score, sample,
                     *sorted(journal.glob('*'))]
        result = dict(passed=True, protocol_sha256=file_hash(out / 'protocol.json'), native_commands=4,
                      full_size_save_load_and_resume_executed=True, full_evaluation_batch_executed=True,
                      graph_generation_executed=True, before=before, after=after, evaluation=observed,
                      sample_hex=raw.hex(), checkpoint_unchanged=True,
                      artifacts_sha256={p.as_posix(): file_hash(p) for p in artifacts},
                      language_quality_established=False, reproduction_admitted=False,
                      limit='Actual 411M save/load, one resumed update and evaluation geometry; not a '
                            'full-size uninterrupted-versus-resumed trajectory comparison or quality result.')
        write(out / 'result.json', result)
        print('411M checkpoint, resumed learning, full evaluation batch and graph sampling passed.', flush=True)
    except Exception as error:
        write(out / 'failure.json', dict(error=str(error), artifacts_preserved=True))
        raise


def declare(out, preflight):
    plan = admit()
    passed = read(preflight / 'result.json')
    require(passed['passed'] and passed['native_commands'] == 4 and
            not (preflight / 'failure.json').exists(), '411M checkpoint preflight has not passed')
    require(file_hash(preflight / 'protocol.json') == passed['protocol_sha256'], 'Preflight protocol changed')
    preflight_plan = read(preflight / 'protocol.json')
    require(preflight_plan['authenticated_inputs'] == plan['authenticated_inputs'], 'Preflight source/runtime differs')
    authenticate(passed['artifacts_sha256'])
    spec = plan['specification']
    baseline = read(spec['baseline_result'])
    rows = [r for r in baseline['rows'] if r['profile'] == '105m']
    require([r['stage'] for r in rows] == [1, 2, 3, 4], 'Missing baseline stages')
    for row in rows:
        require(file_hash(row['checkpoint']) == row['checkpoint_sha256'], 'Baseline checkpoint changed')
        plan['authenticated_inputs'][row['checkpoint']] = row['checkpoint_sha256']
    plan['authenticated_inputs'].update({(preflight / n).as_posix(): file_hash(preflight / n)
                                         for n in ('protocol.json', 'result.json')})
    plan.update(status='declared_before_full_411m_learning', preflight=preflight.as_posix(),
                storage=storage_budget(out), source_worktree=ROOT.as_posix(),
                source_commit=subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
                baseline_assessment_commands_planned=40, learning_commands_planned=1,
                candidate_assessment_commands_planned=40, random_initialization=True,
                imported_weights=False, reserved_tests_scored=False, reproduction_admitted=False)
    out.mkdir(parents=True, exist_ok=False)
    write(out / 'protocol.json', plan)
    print('Declared full 411M learning with 40 baseline and 40 candidate assessments.', flush=True)


def assess_stage(native, checkpoint, evaluation, directory, stage, profile):
    identity = file_hash(checkpoint)
    observed = facts(checkpoint)
    quality = measure(native, checkpoint, evaluation, directory)
    require(file_hash(checkpoint) == identity, 'Assessment changed saved learning state')
    row = dict(profile=profile, stage=stage, checkpoint=checkpoint.as_posix(), checkpoint_sha256=identity,
               state=observed, checkpoint_unchanged=True, reserved_tests_scored=False, **quality)
    verify_saved(directory, row, evaluation, observed['counters']['global_updates'], executable=native.exe)
    return row


def same_exposure(candidate, baseline):
    for key in ('online_updates', 'global_updates', 'observed_pairs', 'generated_bytes', 'replay_updates',
                'replay_pairs', 'curriculum_stage'):
        require(candidate['counters'][key] == baseline['counters'][key], f'Matched exposure differs: {key}')
    for key in ('seed', 'batch', 'chunk', 'fast', 'hyperparameters', 'graph', 'replay_every',
                'replay_capacity', 'cursor_and_rng', 'speech_policy', 'replay_payload_sha256'):
        require(candidate[key] == baseline[key], f'Matched policy differs: {key}')


def execute(out):
    plan = read(out / 'protocol.json')
    require(plan['status'] == 'declared_before_full_411m_learning' and
            not (out / 'execution.json').exists(), '411M study undeclared or already started')
    authenticate(plan['authenticated_inputs'])
    spec, evaluation = plan['specification'], plan['evaluation']
    baseline = [r for r in read(spec['baseline_result'])['rows'] if r['profile'] == '105m']
    write(out / 'execution.json', dict(protocol_sha256=file_hash(out / 'protocol.json'),
                                      storage=storage_budget(out), automatic_parent_promotion=False))
    controls, rows = [], []
    try:
        for old in baseline:
            stage = old['stage']
            directory = out / f'baseline-105m-stage-{stage}'
            directory.mkdir()
            native = NativeCommands(spec['runtime'], directory)
            print('Rechecking preserved 105M baseline at stage', stage, flush=True)
            row = assess_stage(native, Path(old['checkpoint']), evaluation, directory, stage, '105m')
            require(all(row[k] == old[k] for k in ('books', 'samples', 'validation_mean_nats_per_byte')),
                    'Validated runtime did not reproduce the large baseline assessment exactly')
            write(directory / 'result.json', row)
            controls.append(row)
            write(out / 'controls.json', controls)
        authenticate(plan['authenticated_inputs'])
        founder = out / 'founder-411m'
        print('Starting the fresh 411M founder for the complete selected prose curriculum.', flush=True)
        train_founder(founder, spec['source_observations'], spec['profile'], spec['replay_capacity'],
                      executable=Path(spec['runtime']), save_every=spec['save_every'])
        for control in controls:
            stage = control['stage']
            checkpoint = founder / f'stage-{stage}.ckpt'
            observed = facts(checkpoint)
            same_exposure(observed, control['state'])
            require([observed['counters'][k] for k in ('channels', 'hidden', 'layers')] == spec['shape'] and
                    observed['counters']['parameters'] == spec['parameters'], '411M endpoint dimensions differ')
            directory = out / f'411m-stage-{stage}'
            directory.mkdir()
            native = NativeCommands(spec['runtime'], directory)
            print('Assessing the saved 411M founder at stage', stage, flush=True)
            row = assess_stage(native, checkpoint, evaluation, directory, stage, '411m')
            row['comparison'] = dict(validation_difference_from_105m=row['validation_mean_nats_per_byte'] -
                                    control['validation_mean_nats_per_byte'],
                                    books=[dict(book=a['book'], role=a['role'],
                                        difference_from_105m=a['loss_nats_per_byte'] - b['loss_nats_per_byte'])
                                        for a, b in zip(row['books'], control['books'])])
            write(directory / 'result.json', row)
            rows.append(row)
            write(out / 'partial.json', rows)
        authenticate(plan['authenticated_inputs'])
        require(file_hash(founder / 'latest.ckpt') == file_hash(founder / 'stage-4.ckpt'), 'Final checkpoint differs')
        write(out / 'result.json', dict(complete=True, protocol=plan, controls=controls, rows=rows,
              founder_completion=read(founder / 'result.json'), baseline_assessments_exact=True,
              matched_source_replay_and_speech_exposure=True, matched_replay_payloads=True,
              native_learning_commands=1, native_assessment_commands=80,
              random_initialization=True, imported_weights=False, reserved_tests_scored=False,
              reproduction_admitted=False, limits=spec['limits']))
        print('Full 411M comparison complete; no parent or teacher promotion.', flush=True)
    except Exception as error:
        write(out / 'failure.json', dict(error=str(error), completed_candidate_assessments=len(rows),
                                        completed_baseline_assessments=len(controls), artifacts_preserved=True))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('preflight', 'declare', 'run'):
        child = sub.add_parser(name)
        child.add_argument('--out', type=Path, required=True)
        if name == 'declare':
            child.add_argument('--preflight', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'preflight':
        native_preflight(args.out)
    elif args.command == 'declare':
        declare(args.out, args.preflight)
    else:
        execute(args.out)
