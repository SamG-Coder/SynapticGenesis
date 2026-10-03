"""Serialize candidate mechanism acceptance after the whole 411M study.

No model computation is implemented here. Native CUDA commands are launched
only after a held process handle exits successfully and saved completion is
independently verified. Synthetic test artifacts cannot become training sources.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

from large_founder_verification import verify as verify_predecessor
from process_gate import ProcessGate

ROOT = Path(__file__).resolve().parents[1]
CTEST_NAMES = ('spiking-native', 'spiking-live', 'spiking-graph', 'spiking-replay', 'spiking-adaptive',
               'spiking-synaptic', 'spiking-evolution', 'spiking-curriculum', 'spiking-feedback',
               'spiking-distillation', 'spiking-teacher-replay', 'spiking-trace', 'spiking-gated',
               'spiking-selective', 'spiking-associative', 'spiking-association-layout',
               'spiking-stage-replay', 'spiking-reductions')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8', newline='\n')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def authenticate(pins):
    for name, expected in pins.items():
        require(sha(name) == expected, f'Acceptance input changed: {name}')


def command_plan(out, python, ctest, old, new, probe):
    out = Path(out).resolve()
    commands = []

    def add(label, *args):
        commands.append(dict(label=label, argv=list(map(str, args))))

    fixtures = out / 'membrane-gpu'
    add('membrane-native', probe, 'gpu-test', '--out', fixtures)
    for cell in range(3, 7):
        for batch in (1, 2):
            add(f'membrane-oracle-{cell}-{batch}', python, '-X', 'utf8', ROOT / 'tests/oracle.py',
                fixtures / f'cell-{cell}-batch-{batch}')
    add('native-suite', ctest, '--test-dir', ROOT / 'build', '--parallel', '1', '--no-tests=error',
        '--output-on-failure', '--output-junit', out / 'ctest.xml')
    for folder in ('test', 'adaptive-test', 'trace-test', 'gated-test', 'selective-test', 'associative-test'):
        add(f'legacy-oracle-{folder}', python, '-X', 'utf8', ROOT / 'tests/oracle.py', ROOT / 'build' / f'{folder}-results')
    add('disabled-runtime-parity', python, '-X', 'utf8', ROOT / 'tests/live_views.py',
        '--out', out / 'disabled-parity', '--old-exe', old, '--new-exe', new)
    for side, executable in (('old', old), ('new', new)):
        add(f'teacher-cli-{side}', python, '-X', 'utf8', ROOT / 'tests/teacher_replay_cli.py',
            '--exe', executable, '--out', out / f'teacher-{side}')
    add('membrane-teacher-cli', python, '-X', 'utf8', ROOT / 'tests/membrane_teacher_cli.py',
        '--exe', new, '--out', out / 'membrane-teacher')
    return commands


def checked_ctest(ctest, new):
    value = json.loads(subprocess.check_output([str(ctest), '--test-dir', str(ROOT / 'build'),
                                               '--show-only=json-v1'], text=True, encoding='utf-8'))
    require(tuple(t['name'] for t in value['tests']) == CTEST_NAMES, 'Unexpected CTest coverage')
    require(all(Path(t['command'][0]).resolve() == Path(new).resolve() for t in value['tests']),
            'CTest points at a different runtime')
    return value['tests']


def declare(out, workspace, predecessor, old, wait_pid, wait_executable):
    out, workspace, predecessor, old = map(lambda p: Path(p).resolve(), (out, workspace, predecessor, old))
    require(not out.exists(), 'Acceptance output already exists')
    new = ROOT / 'build/synapticgenesis.exe'
    probe = ROOT / 'build/membrane-checkpoint-test/synaptic-membrane-checkpoint-test.exe'
    python, ctest = Path(sys.executable).resolve(), Path(shutil.which('ctest') or '').resolve()
    require(old != new and sha(old) != sha(new), 'Acceptance requires distinct preserved and candidate runtimes')
    integration = read(ROOT / 'reports/membrane-policy-integration.json')
    require(sha(new) == integration['candidate_binaries']['build/synapticgenesis.exe'] and
            sha(probe) == integration['candidate_binaries']['build/membrane-checkpoint-test/synaptic-membrane-checkpoint-test.exe'],
            'Candidate binaries differ from the host-verified integration')
    for name, expected in integration['source_and_evidence_sha256'].items():
        require(sha(ROOT / name) == expected, f'Integrated source/evidence changed: {name}')
    prior = read(predecessor / 'protocol.json')
    require(prior['specification']['version'] == 'prose-411m-v1' and
            old == (workspace / prior['specification']['runtime']).resolve() and
            sha(old) == prior['specification']['runtime_sha256'], 'Preserved runtime differs from the 411M driver')
    gate = ProcessGate(wait_pid, wait_executable) if wait_pid else None
    try:
        if gate:
            require(Path(gate.identity['executable']).name.lower() == 'python.exe',
                    'Wait for the Python study driver, not its native learning child')
        else:
            verify_predecessor(workspace, predecessor, sha(predecessor / 'protocol.json'))
        tests = checked_ctest(ctest, new)
        inputs = [new, probe, old, python, ctest, predecessor / 'protocol.json',
                  ROOT / 'build/CTestTestfile.cmake', ROOT / 'reports/membrane-policy-integration.json',
                  ROOT / 'CMakeLists.txt', ROOT / 'build.ps1',
                  *sorted((ROOT / 'src').glob('*.cu')), *sorted((ROOT / 'src').glob('*.cuh')),
                  *sorted((ROOT / 'scripts').rglob('*.py')), *sorted((ROOT / 'tests').glob('*.py')),
                  ROOT / 'tests/membrane_checkpoint.cu', ROOT / 'tests/membrane_learning.cuh']
        plan = dict(version='membrane-acceptance-v1', status='declared_before_gpu_execution',
                    created_utc=datetime.now(timezone.utc).isoformat(),
                    source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                    root=str(ROOT), workspace=str(workspace), predecessor=str(predecessor),
                    predecessor_protocol_sha256=sha(predecessor / 'protocol.json'),
                    wait_for_driver=gate.identity if gate else None,
                    executables={k: str(v) for k, v in dict(old=old, new=new, probe=probe, python=python, ctest=ctest).items()},
                    ctest=tests, commands=command_plan(out, python, ctest, old, new, probe),
                    authenticated_inputs={str(p): sha(p) for p in inputs},
                    new_sources_admitted=False, language_trial=False, reproduction_admitted=False,
                    scope='GPU derivative/restart checks and disabled-runtime compatibility, not language quality or speed.')
        out.mkdir(parents=True)
        write(out / 'protocol.json', plan)
        print('Declared', len(plan['commands']), 'sequential acceptance commands; no CUDA work.', flush=True)
        return plan
    finally:
        if gate:
            gate.close()


def ctest_result(path):
    suite = ET.parse(path).getroot()
    tests = suite.findall('testcase')
    require(suite.tag == 'testsuite' and int(suite.attrib['tests']) == len(CTEST_NAMES) and
            int(suite.attrib.get('failures', 0)) == 0 and int(suite.attrib.get('disabled', 0)) == 0 and
            int(suite.attrib.get('skipped', 0)) == 0 and
            int(suite.attrib.get('errors', 0)) == 0 and len(tests) == len(CTEST_NAMES) and
            sorted(t.attrib['name'] for t in tests) == sorted(CTEST_NAMES), 'Native CTest result is incomplete')
    require(all(t.find('failure') is None and t.find('error') is None and t.find('skipped') is None for t in tests),
            'Native CTest case failed or was skipped')
    return dict(passed=True, cases=len(tests), junit_sha256=sha(path))


def teacher_parity(out):
    left, right = Path(out) / 'teacher-old', Path(out) / 'teacher-new'
    require(read(left / 'result.json')['passed'] and read(right / 'result.json')['passed'],
            'Teacher CLI checks did not pass')
    names = lambda root: {p.relative_to(root).as_posix() for p in root.rglob('*')
                          if p.is_file() and (p.suffix == '.ckpt' or p.name == 'transcript.txt')}
    selected = names(left)
    require(selected and selected == names(right), 'Teacher compatibility artifact sets differ')
    hashes = {}
    for name in sorted(selected):
        require(sha(left / name) == sha(right / name), f'Disabled teacher checkpoint or speech differs: {name}')
        hashes[name] = sha(left / name)
    return dict(passed=True, exact_artifacts=hashes)


def collect_results(out, plan):
    out = Path(out)
    native = read(out / 'membrane-gpu/native-result.json')
    require(native['passed'] is True and native['gradient_fixtures'] == 8 and native['exact_restart_cases'] == 8,
            'Membrane native checks are incomplete')
    oracles = []
    for cell in range(3, 7):
        for batch in (1, 2):
            path = out / 'membrane-gpu' / f'cell-{cell}-batch-{batch}'
            fixture, value = read(path / 'fixture.json'), read(path / 'oracle.json')
            require(value['passed'] is True and value['cell'] == cell and fixture['batch'] == batch and
                    value['membrane']['cost'] == .2 and value['membrane']['band'] == 1.5 and
                    value['weighted_targets'] and value['nonzero_boundary_state'] and
                    fixture['outside_surrogate_support'] > 0, 'Membrane oracle case did not pass')
            oracles.append(dict(cell=cell, batch=batch, result=value, sha256=sha(path / 'oracle.json')))
    for cell, folder in enumerate(('test', 'adaptive-test', 'trace-test', 'gated-test', 'selective-test', 'associative-test'), 1):
        value = read(ROOT / 'build' / f'{folder}-results/oracle.json')
        require(value['passed'] and value['cell'] == cell, 'Legacy oracle failed or has the wrong cell')
    disabled = read(out / 'disabled-parity/result.json')
    require(disabled['passed'] and disabled['native_commands'] == 48 and len(disabled['cases']) == 12 and
            [(r['cell'], r['tf32']) for r in disabled['cases']] ==
                [(cell, fast) for cell in ('lif', 'alif', 'trace', 'gated', 'selective', 'associative') for fast in (False, True)] and
            disabled['old_sha256'] == plan['authenticated_inputs'][plan['executables']['old']] and
            disabled['new_sha256'] == plan['authenticated_inputs'][plan['executables']['new']] and
            all(r['complete_old_checkpoint_exact'] and r['old_speech_exact'] and r['complete_restart_exact']
                for r in disabled['cases']), 'Disabled runtime or restart parity failed')
    teacher = read(out / 'membrane-teacher/membrane-result.json')
    require(teacher['passed'] and teacher['native_commands'] == 11 and teacher['source_updates'] == 64 and
            teacher['regularized_checkpoint_exact_on_restart'] and teacher['speech_exact_on_restart'] and
            teacher['enabled_cost_changes_learned_weights'] and teacher['two_teachers_active'] and
            teacher['optional_si_exercised'] and teacher['read_only_commands_preserve_checkpoint'],
            'Regularized teacher/restart checks did not pass')
    return dict(passed=True, native=native, oracles=oracles, ctest=ctest_result(out / 'ctest.xml'),
                disabled_parity=disabled, teacher_parity=teacher_parity(out),
                regularized_teacher=teacher,
                language_quality_established=False, reproduction_admitted=False)


def execute(out, protocol_sha256):
    out = Path(out).resolve()
    require(not (out / 'execution.json').exists(), 'Acceptance already started; preserve its evidence')
    require(sha(out / 'protocol.json') == protocol_sha256, 'Acceptance protocol identity changed')
    plan = read(out / 'protocol.json')
    require(plan['version'] == 'membrane-acceptance-v1' and plan['status'] == 'declared_before_gpu_execution' and
            Path(plan['root']).resolve() == ROOT, 'Unexpected acceptance protocol')
    e = plan['executables']
    require(plan['commands'] == command_plan(out, e['python'], e['ctest'], e['old'], e['new'], e['probe']),
            'Acceptance command plan changed')
    authenticate(plan['authenticated_inputs'])
    gate, completed = None, []
    try:
        waiting = plan['wait_for_driver']
        if waiting:
            gate = ProcessGate(waiting['pid'], waiting['executable'])
            require(gate.identity == waiting, 'Study driver PID identity changed')
        write(out / 'execution.json', dict(protocol_sha256=sha(out / 'protocol.json'),
              driver=waiting, native_work_started=False, phase='waiting-for-study' if gate else 'verify-study'))
        if gate:
            print('Waiting for the complete 411M driver:', waiting['pid'], flush=True)
            code = gate.wait()
            write(out / 'predecessor-exit.json', dict(**gate.identity, exit_code=code))
            gate.close(); gate = None
        previous = verify_predecessor(plan['workspace'], plan['predecessor'], plan['predecessor_protocol_sha256'])
        write(out / 'predecessor-verified.json', previous)
        authenticate(plan['authenticated_inputs'])
        require(checked_ctest(e['ctest'], e['new']) == plan['ctest'], 'CTest declaration changed')
        for index, command in enumerate(plan['commands'], 1):
            authenticate(plan['authenticated_inputs'])
            write(out / 'execution.json', dict(protocol_sha256=sha(out / 'protocol.json'), driver=waiting,
                  native_work_started=True, phase=command['label'], completed=len(completed)))
            write(out / 'commands.json', plan['commands'][:index])
            print('Acceptance:', command['label'], flush=True)
            result = subprocess.run(command['argv'], cwd=ROOT, capture_output=True)
            log = out / f'command-{index:03d}.log'
            log.write_bytes(result.stdout + result.stderr)
            require(result.returncode == 0, f'Acceptance command failed: {command["label"]}; see {log}')
            completed.append(dict(label=command['label'], returncode=result.returncode, log_sha256=sha(log)))
            write(out / 'completed-commands.json', completed)
        result = collect_results(out, plan)
        authenticate(plan['authenticated_inputs'])
        result.update(complete=True, protocol_sha256=sha(out / 'protocol.json'), commands=completed,
                      predecessor_verification_sha256=sha(out / 'predecessor-verified.json'))
        write(out / 'result.json', result)
        write(out / 'execution.json', dict(protocol_sha256=sha(out / 'protocol.json'), driver=waiting,
              native_work_started=True, phase='complete', completed=len(completed)))
        print('Candidate acceptance complete; no language trial or promotion performed.', flush=True)
    except BaseException as error:
        write(out / 'failure.json', dict(error=str(error), completed_commands=completed, artifacts_preserved=True))
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
    create.add_argument('--predecessor', type=Path, required=True)
    create.add_argument('--old', type=Path, required=True)
    create.add_argument('--wait-pid', type=int, default=0)
    create.add_argument('--wait-executable', type=Path)
    run = commands.add_parser('run')
    run.add_argument('--out', type=Path, required=True)
    run.add_argument('--protocol-sha256', required=True)
    args = parser.parse_args()
    if args.command == 'declare':
        declare(args.out, args.workspace, args.predecessor, args.old, args.wait_pid, args.wait_executable)
    else:
        execute(args.out, args.protocol_sha256)
