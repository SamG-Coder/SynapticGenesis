"""Serial native baseline assessment after the live objective study exits."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys

from membrane_study_state import read_state, require
from native_experiment import NativeCommands, read
from process_gate import ProcessGate
from prose_founder import file_hash, write
from quantitative_assessment_inputs import ROOT, SPEC, admit, arguments, authenticate, completed_predecessor, sample_rows
from quantitative_report import audit


def declare(out, workspace, wait_pid, wait_executable):
    out, workspace = Path(out).resolve(), Path(workspace).resolve()
    require(Path.cwd().resolve() == workspace and not out.exists(), 'Use the workspace cwd and a fresh assessment directory')
    admission = admit()
    spec = admission['specification']
    gate = ProcessGate(wait_pid, wait_executable)
    try:
        require(Path(gate.identity['executable']).name.lower() == 'python.exe', 'Wait on the complete Python study driver')
        require(gate.kernel.WaitForSingleObject(gate.handle, 0) == 258, 'Declare while the study driver is still live')
        cases = [dict(case, directory=str(out / case['profile'])) for case in admission['cases']]
        for case in cases:
            case['commands'] = arguments(spec['runtime'], case, Path(spec['suite']), admission['questions'],
                                         spec['generation'], Path(case['directory']))
        pins = dict(admission['authenticated_inputs'])
        pins[str(Path(sys.executable).resolve())] = file_hash(sys.executable)
        plan = dict(version='quantitative-assessment-plan-v1', status='declared_before_native_scoring',
            created_utc=datetime.now(timezone.utc).isoformat(), workspace=str(workspace), source_checkout=str(ROOT),
            source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            specification=spec, cases=cases, questions=admission['questions'],
            authenticated_inputs=pins, wait_for_study_driver=gate.identity,
            native_commands_planned=18, learning_commands_planned=0, test_set_scored=False,
            suite_host_audit=admission['suite_preparation_audit'], reproduction_admitted=False)
        out.mkdir(parents=True)
        write(out / 'protocol.json', plan)
        print('Declared two complete prose checkpoints and 18 read-only native commands; no CUDA work.', flush=True)
        return plan
    finally:
        gate.close()


def execute(out, protocol_sha256):
    out = Path(out).resolve()
    require(file_hash(out / 'protocol.json') == protocol_sha256 and not (out / 'execution.json').exists(),
            'Assessment declaration changed or already started')
    plan = read(out / 'protocol.json')
    require(plan['version'] == 'quantitative-assessment-plan-v1' and
            plan['status'] == 'declared_before_native_scoring' and Path(plan['source_checkout']) == ROOT and
            Path(plan['workspace']) == Path.cwd().resolve(), 'Unexpected assessment checkout or workspace')
    spec = plan['specification']
    require(spec == read(SPEC) and plan['native_commands_planned'] == 18 and
            plan['learning_commands_planned'] == 0, 'Declared assessment policy changed')
    require([{k: case[k] for k in declared} for case, declared in zip(plan['cases'], spec['cases'])] == spec['cases']
            and len(plan['cases']) == 2, 'Declared baseline selection changed')
    require(plan['questions'] == read(Path(spec['suite']) / 'questions.json'), 'Declared questions changed')
    for case in plan['cases']:
        require(case['commands'] == arguments(spec['runtime'], case, Path(spec['suite']), plan['questions'],
                spec['generation'], Path(case['directory'])), 'Declared native arguments changed')
        require(Path(case['directory']) == out / case['profile'], 'Assessment output directory differs')
    authenticate(plan['authenticated_inputs'])
    gate, results, native_started = None, [], False
    try:
        waiting = plan['wait_for_study_driver']
        gate = ProcessGate(waiting['pid'], waiting['executable'])
        require(gate.identity == waiting, 'Study process identity changed')
        write(out / 'execution.json', dict(protocol_sha256=protocol_sha256, phase='waiting-for-study',
            native_work_started=False, completed_models=0, study_driver=waiting))
        print('Waiting on the held study-driver handle:', waiting['pid'], flush=True)
        code = gate.wait()
        write(out / 'predecessor-exit.json', dict(**gate.identity, exit_code=code))
        gate.close(); gate = None
        predecessor = completed_predecessor(spec['predecessor'], spec['predecessor_protocol_sha256'])
        write(out / 'predecessor-completion.json', predecessor)
        authenticate(plan['authenticated_inputs'])
        suite_bytes = (Path(spec['suite']) / 'development.sgprobe').read_bytes()
        for case in plan['cases']:
            authenticate(plan['authenticated_inputs'])
            require(read_state(case['checkpoint']) == case['state'], 'Baseline checkpoint metadata changed')
            directory = Path(case['directory'])
            directory.mkdir()
            native = NativeCommands(spec['runtime'], directory)
            write(out / 'execution.json', dict(protocol_sha256=protocol_sha256, phase='assessment',
                native_work_started=True, profile=case['profile'], completed_models=len(results)))
            native_started = True
            for command in case['commands']:
                require(Path(command[0]).resolve() == Path(spec['runtime']).resolve(), 'Native executable changed')
                native(*command[1:])
            require(native.commands == case['commands'], 'Native command journal differs')
            raw = read(directory / 'probes.json')
            checked = audit(raw, plan['questions'], suite_bytes, case['state']['meta'][15])
            generated = []
            for question in sample_rows(plan['questions']):
                sample = directory / f"sample-{question['skill']}.txt"
                value, prefix = sample.read_bytes(), (question['context'] + question['query']).encode('ascii')
                require(value.startswith(prefix) and len(value) == len(prefix) + spec['generation']['bytes'],
                        'Native raw sample prompt or length differs')
                generated.append(dict(id=question['id'], prompt=prefix.decode('ascii'),
                    output_hex=value.hex(), output_utf8=value.decode('utf-8', errors='replace'),
                    sha256=file_hash(sample), generated_bytes=spec['generation']['bytes']))
            require(file_hash(case['checkpoint']) == case['checkpoint_sha256'], 'Read-only assessment changed the checkpoint')
            artifacts = {p.name: file_hash(p) for p in sorted(directory.iterdir()) if p.is_file()}
            record = dict(complete=True, profile=case['profile'], checkpoint=case['checkpoint'],
                checkpoint_sha256=case['checkpoint_sha256'], checkpoint_unchanged=True,
                probes=raw, audit=checked, samples=generated, native_commands=9, artifact_sha256=artifacts,
                raw_sample_math=spec['generation']['arithmetic'], learning_commands=0,
                test_set_scored=False, reproduction_admitted=False)
            write(directory / 'result.json', record)
            results.append(record)
            write(out / 'completed-models.json', results)
            print(case['profile'], 'correct groups:', checked['correct_groups'], '/32; greedy items:',
                  checked['greedy_exact_items'], '/128', flush=True)
        authenticate(plan['authenticated_inputs'])
        result = dict(complete=True, protocol_sha256=protocol_sha256, protocol=plan, rows=results,
            predecessor_completion=predecessor, native_commands=18, learning_commands=0,
            test_set_scored=False, reproduction_admitted=False, limits=spec['limits'])
        write(out / 'result.json', result)
        write(out / 'execution.json', dict(protocol_sha256=protocol_sha256, phase='complete',
            native_work_started=True, completed_models=2))
        print('Quantitative baselines complete; all scores and raw samples retained.', flush=True)
        return result
    except BaseException as error:
        write(out / 'failure.json', dict(error=str(error), completed_models=len(results),
              native_work_started=native_started, artifacts_preserved=True))
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
    create.add_argument('--wait-pid', type=int, required=True)
    create.add_argument('--wait-executable', type=Path, required=True)
    run = commands.add_parser('run')
    run.add_argument('--out', type=Path, required=True)
    run.add_argument('--protocol-sha256', required=True)
    args = parser.parse_args()
    if args.command == 'declare':
        declare(args.out, args.workspace, args.wait_pid, args.wait_executable)
    else:
        execute(args.out, args.protocol_sha256)
