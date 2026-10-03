"""Serialize early-width learning controls behind the existing native studies."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess

from checkpoint_assessment import measure
from early_width_observations import (SPEECH_EVERY, SPEECH_GENERATED_BYTES, SPEECH_PROMPT,
                                      audit, match_guard_speech)
from native_experiment import NativeCommands, read
from process_gate import ProcessGate
from prose_evaluation import declare as declare_assessments
from prose_founder import file_hash, live_arguments, write
from prose_retention_inputs import COUNTERS, authenticate, predecessors, require, state

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / 'data/prose-early-width-v1.json'
PROBE = ROOT / 'build/early-learning-probe/synaptic-early-learning-probe.exe'


def validate(spec):
    require(spec['version'] == 'prose-early-width-v1' and spec['source_observations'] == 8176,
            'Unexpected early-width exposure')
    require(spec['observation_points'] == [1, 32, 128, 512, 2048, 8176] and
            spec['seeds'] == [1337, 2027, 4099], 'Early-width observations or seeds differ')
    expected = [('c512-control', 512, 2048, 1., '27m'), ('c1024-control', 1024, 4096, 1., '105m'),
                ('c256-control', 256, 1024, 1., None), ('c512-core-half', 512, 2048, .5, None),
                ('c1024-core-quarter', 1024, 4096, .25, None)]
    require([(c['name'], c['channels'], c['hidden'], c['core_scale'], c['baseline_profile'])
             for c in spec['cases']] == expected and all(c['layers'] == 8 for c in spec['cases']),
            'Width family or rate controls differ')
    require(spec['native_guard_updates'] == 511 and spec['native_guard_points'] == [1, 128, 511],
            'Native compatibility workload differs')


def arguments(case, seed, out, schedule, end, points=None):
    """Reuse the founder's declared live policy; change only explicit study knobs."""
    args = list(live_arguments(out, schedule, end, case.get('baseline_profile') or '27m'))
    require(args[args.index('--prompt') + 1].encode('utf-8') == SPEECH_PROMPT and
            args[args.index('--tokens') + 1] == SPEECH_GENERATED_BYTES and
            args[args.index('--speak-every') + 1] == SPEECH_EVERY, 'Early-width speech policy differs')
    for key in ('channels', 'hidden', 'layers'):
        args[args.index('--' + key) + 1] = case[key]
    args[args.index('--seed') + 1] = seed
    args += ['--core-scale', case['core_scale']]
    if points is None:
        return args
    args[0] = 'run'
    for key in ('--cell', '--log-every', '--save-every'):
        at = args.index(key)
        del args[at:at + 2]
    return args + ['--points', ','.join(map(str, points))]


def declare(out):
    spec = read(SPEC)
    validate(spec)
    out.mkdir(parents=True, exist_ok=False)
    assessment_dir = out / 'assessment-plan'
    declare_assessments(assessment_dir)
    evaluation = read(assessment_dir / 'protocol.json')
    require(evaluation['exposure']['stages'][0]['end_update'] == spec['source_observations'], 'First source stage differs')
    predecessor = read(spec['predecessor_protocol'])
    require(predecessor['specification']['candidate_capacity'] == 16384, 'Queued predecessor differs')
    files = [SPEC, PROBE, ROOT / 'CMakeLists.txt', ROOT / 'build.ps1',
             *(ROOT / 'src').glob('*.cu'), *(ROOT / 'src').glob('*.cuh'),
             ROOT / 'experiments/early_learning_probe.cu', ROOT / 'experiments/learning_scale_observer.cuh',
             ROOT / 'tests/prose_early_width.py', ROOT / 'data/training-selection.json',
             ROOT / 'tests/early_width_speech.py',
             *(ROOT / 'scripts' / n for n in ('prose_early_width.py', 'early_width_observations.py',
                 'prose_projection_snapshot.py', 'prose_retention_inputs.py', 'prose_size_comparison.py',
                 'experiment_checkpoint.py', 'checkpoint_assessment.py', 'prose_founder.py', 'native_experiment.py',
                 'process_gate.py', 'corpus/selection.py')),
             Path(spec['predecessor_protocol']), Path(spec['baseline_comparison']), Path(spec['neuron_panel'])]
    pins = dict(evaluation['authenticated_inputs'])
    pins.update({p.as_posix(): file_hash(p) for p in files})
    write(out / 'protocol.json', dict(status='declared_before_early_width_learning', specification=spec,
        source_checkout=ROOT.as_posix(), source_commit=subprocess.check_output(
            ['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
        diagnostic_executable=PROBE.as_posix(), assessment_plan=(assessment_dir / 'protocol.json').as_posix(),
        authenticated_inputs=pins, native_guard_commands_planned=2,
        native_learning_commands_planned=15, native_assessment_commands_planned=150,
        random_initialization=True, imported_weights=False, reproduction_admitted=False,
        scope='Early developmental comparison only; unchanged native neuron/optimizer equations.'))
    print('Declared 15 fresh founders, three seeds, five width/rate cases and 167 native commands.', flush=True)


def complete_predecessors(spec):
    pins = predecessors(Path(spec['baseline_comparison']), Path(spec['neuron_panel']))
    rate, capacity = read(spec['learning_rate_result']), read(spec['predecessor_result'])
    require(rate['complete'] and rate['control_checkpoint_byte_identical'] and rate['native_commands'] == 64 and
            not rate['reserved_tests_scored'], 'Learning-rate predecessor is incomplete')
    require(capacity['complete'] and capacity['native_learning_commands'] == 2 and
            capacity['native_assessment_commands'] == 80 and capacity['same_initial_model_arrays'] and
            capacity['same_candidate_replay_history_across_sizes'] and not capacity['reserved_tests_scored'],
            'Replay-capacity predecessor is incomplete')
    require(capacity['protocol'] == read(spec['predecessor_protocol']), 'Replay predecessor protocol differs')
    for name in (spec['learning_rate_result'], spec['predecessor_result']):
        pins[name] = file_hash(name)
    return pins


def match_policy(left, right):
    require(all(left['counters'][k] == right['counters'][k] for k in COUNTERS), 'Learning exposures differ')
    for key in ('replay_payload_sha256', 'cursor_and_rng', 'seed', 'batch', 'chunk', 'fast',
                'speech_policy', 'graph', 'replay_every', 'replay_capacity'):
        require(left[key] == right[key], f'Learning policy differs: {key}')
    require(all(left['hyperparameters'][i] == right['hyperparameters'][i] for i in (0, 1, 2, 3, 4, 5, 7)),
            'A policy other than the core rate changed')


def initial_match(control, candidate):
    def content(path):
        with path.open('rb') as stream:
            meta = list(struct.unpack('<32Q', stream.read(256)))
            hp = list(struct.unpack('<8f', stream.read(32)))
            payload = hashlib.file_digest(stream, 'sha256').hexdigest()
        meta[15], hp[6] = 0, 0
        require(meta[7] == meta[22] == meta[24] == meta[30] == 0, 'Initial checkpoint has learning history')
        return meta, hp, payload
    require(content(control) == content(candidate), 'Core-rate pair did not start with identical model arrays/state')


def execute(out, wait_pid=0, wait_executable=None):
    plan = read(out / 'protocol.json')
    require(plan['status'] == 'declared_before_early_width_learning', 'Undeclared early-width experiment')
    spec = plan['specification']
    validate(spec)
    authenticate(plan['authenticated_inputs'])
    gate = ProcessGate(wait_pid, wait_executable) if wait_pid else None
    rows = []
    try:
        if gate:
            write(out / 'waiting-process.json', gate.identity)
            print('Waiting behind the replay-capacity study:', gate.identity, flush=True)
            code = gate.wait()
            write(out / 'predecessor-exit.json', dict(**gate.identity, exit_code=code))
        authenticate(plan['authenticated_inputs'])
        prior = complete_predecessors(spec)
        write(out / 'predecessors-complete.json', prior)
        evaluation = read(plan['assessment_plan'])
        source_stage = evaluation['exposure']['stages'][0]['cumulative_source']
        schedule = Path(evaluation['specification']['schedule'])
        old = read(spec['baseline_comparison'])
        for name in ('guard-control-commands', 'guard-observed-commands', 'learning-commands'):
            (out / name).mkdir()
        control = NativeCommands('build/synapticgenesis.exe', out / 'guard-control-commands')
        guard_probe = NativeCommands(plan['diagnostic_executable'], out / 'guard-observed-commands')
        native = NativeCommands(plan['diagnostic_executable'], out / 'learning-commands')
        tiny = dict(channels=8, hidden=32, layers=2, core_scale=1.)
        guard_end = spec['native_guard_updates']
        control(*arguments(tiny, 1337, out / 'guard-control', schedule, guard_end))
        guard_probe(*arguments(tiny, 1337, out / 'guard-observed', schedule, guard_end, spec['native_guard_points']))
        guard = audit(out / 'guard-observed', tiny, 1337, guard_end, spec['native_guard_points'], source_stage)
        for filename in ('initial.ckpt', 'latest.ckpt'):
            require(file_hash(out / 'guard-control' / filename) == file_hash(out / 'guard-observed' / filename),
                    'Diagnostic observer changed the native guard checkpoint')
        speech_guard = match_guard_speech(out / 'guard-control', out / 'guard-observed', guard_end)
        guard.update(initial_and_final_checkpoints_byte_identical=True, includes_scheduled_speech=True,
                     includes_replay=True, native_commands=2, speech=speech_guard)
        write(out / 'native-guard.json', guard)
        assessment_commands = 0
        legacy_checks = []
        for seed in spec['seeds']:
            same_seed = []
            for case in spec['cases']:
                authenticate(plan['authenticated_inputs'])
                directory = out / f'{seed}-{case["name"]}'
                print('Learning', directory.name, 'from random initialization.', flush=True)
                native(*arguments(case, seed, directory, schedule, spec['source_observations'], spec['observation_points']))
                observation_audit = audit(directory, case, seed, spec['source_observations'], spec['observation_points'], source_stage)
                write(directory / 'observation-audit.json', observation_audit)
                checkpoint = directory / 'latest.ckpt'
                identity, observed = file_hash(checkpoint), state(checkpoint)
                require(observed['hyperparameters'][6] == case['core_scale'], 'Saved core learning rate differs')
                if same_seed:
                    match_policy(same_seed[0]['state'], observed)
                if case['core_scale'] != 1:
                    match = next(r for r in same_seed if r['case']['channels'] == case['channels'] and r['case']['core_scale'] == 1)
                    initial_match(Path(match['directory']) / 'initial.ckpt', directory / 'initial.ckpt')
                if seed == 1337 and case['baseline_profile']:
                    original = next(r for r in old['rows'] if r['profile'] == case['baseline_profile'] and r['stage'] == 1)
                    original_initial = Path(original['checkpoint']).parent / 'initial.ckpt'
                    require(file_hash(original['checkpoint']) == original['checkpoint_sha256'] == identity and
                            file_hash(original_initial) == file_hash(directory / 'initial.ckpt'),
                            'Instrumented width control differs from the original learned checkpoint')
                    legacy_checks.append(dict(profile=case['baseline_profile'], initial_and_final_byte_identical=True,
                                              checkpoint_sha256=identity))
                assessment = directory / 'assessment'
                assessment.mkdir()
                scorer = NativeCommands('build/synapticgenesis.exe', assessment)
                quality = measure(scorer, checkpoint, evaluation['specification'], assessment)
                assessment_commands += len(scorer.commands)
                require(file_hash(checkpoint) == identity, 'Assessment changed its checkpoint')
                if seed == 1337 and case['baseline_profile']:
                    require(all(quality[k] == original[k] for k in quality), 'Original width-control quality differs')
                observations = [json.loads(line) for line in (directory / 'observations.jsonl').read_text().splitlines()]
                row = dict(case=case, seed=seed, directory=directory.as_posix(), checkpoint_sha256=identity,
                           state=observed, native_result=read(directory / 'result.json'), quality=quality,
                           observations=observations, observation_audit_sha256=file_hash(directory / 'observation-audit.json'))
                rows.append(row)
                same_seed.append(row)
                write(out / 'partial.json', dict(rows=rows, legacy_checks=legacy_checks))
                print(directory.name, 'validation mean', quality['validation_mean_nats_per_byte'], flush=True)
        require(len(rows) == len(native.commands) == 15 and assessment_commands == 150 and len(legacy_checks) == 2,
                'Early-width command or compatibility counts differ')
        authenticate(plan['authenticated_inputs'])
        authenticate(prior)
        write(out / 'result.json', dict(complete=True, protocol=plan, rows=rows, legacy_checks=legacy_checks,
            native_guard=guard, native_guard_commands=2, native_learning_commands=15,
            native_assessment_commands=assessment_commands, same_source_replay_and_speech_policy=True,
            paired_initial_model_arrays_identical=True, random_initialization=True, imported_weights=False,
            reserved_tests_scored=False, reproduction_admitted=False, limits=spec['limits']))
        print('Early-width comparison complete; no default model or learning policy promoted.', flush=True)
    except Exception as error:
        write(out / 'failure.json', dict(error=str(error), completed_cases=len(rows), partial_artifacts_preserved=True))
        raise
    finally:
        if gate:
            gate.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    declaration = sub.add_parser('declare')
    declaration.add_argument('--out', type=Path, required=True)
    run = sub.add_parser('run')
    run.add_argument('--out', type=Path, required=True)
    run.add_argument('--wait-pid', type=int, default=0)
    run.add_argument('--wait-executable', type=Path)
    args = parser.parse_args()
    if args.command == 'declare':
        declare(args.out)
    else:
        if args.wait_pid and args.wait_executable is None:
            parser.error('--wait-executable is required with --wait-pid')
        execute(args.out, args.wait_pid, args.wait_executable)
