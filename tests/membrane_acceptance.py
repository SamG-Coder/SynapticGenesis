"""Host-only gate/comparator failure injection plus real Windows process handles."""
import argparse
from contextlib import redirect_stdout
from copy import deepcopy
import io
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
spec = importlib.util.spec_from_file_location('acceptance_driver', ROOT / 'scripts/membrane_acceptance.py')
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)
from large_founder_verification import completed_header
from process_gate import ProcessGate


def check(out, checkpoint_fixtures=None):
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    rejected = []
    snapshot_checks = []
    if checkpoint_fixtures:
        from membrane_teacher_cli import snapshot
        checkpoint_fixtures = Path(checkpoint_fixtures).resolve()
        assert driver.read(checkpoint_fixtures / 'result.json')['passed']
        for si in (0, 1):
            paths = [checkpoint_fixtures / f'6-6-{si}-{label}.ckpt' for label in ('legacy', 'membrane')]
            old, new = map(snapshot, paths)
            assert old['weight_sha256'] == new['weight_sha256'] and old['teacher'] == new['teacher'] and old['extra'] == new['extra']
            assert old['policy'] is None and new['policy'][2] == 6
            snapshot_checks.append(dict(optional_si=bool(si), files={str(p): driver.sha(p) for p in paths}))

    def fails(name, action, message):
        try:
            action()
        except (ValueError, RuntimeError) as error:
            assert message in str(error), (name, error)
            rejected.append(name)
        else:
            raise AssertionError(f'Accepted negative control: {name}')

    actual = driver.checked_ctest(Path('ctest'), ROOT / 'build/synapticgenesis.exe')
    assert tuple(row['name'] for row in actual) == driver.CTEST_NAMES
    commands = driver.command_plan(out, 'python.exe', 'ctest.exe', 'old.exe', 'new.exe', 'probe.exe')
    assert len(commands) == 20 and len({c['label'] for c in commands}) == 20
    assert [c['label'] for c in commands[:9]] == ['membrane-native'] + [
        f'membrane-oracle-{cell}-{batch}' for cell in range(3, 7) for batch in (1, 2)]
    assert commands[9]['argv'][commands[9]['argv'].index('--parallel') + 1] == '1'
    assert '--no-tests=error' in commands[9]['argv']
    xml = out / 'ctest.xml'
    suite = ET.Element('testsuite', tests='18', failures='0', errors='0', disabled='0')
    for name in driver.CTEST_NAMES:
        ET.SubElement(suite, 'testcase', name=name)
    ET.ElementTree(suite).write(xml)
    assert driver.ctest_result(xml)['cases'] == 18
    for label, change in [
            ('missing_case', lambda r: r.remove(r[-1])),
            ('duplicate_case', lambda r: r[0].set('name', r[1].get('name'))),
            ('skipped_case', lambda r: ET.SubElement(r[0], 'skipped')),
            ('failed_case', lambda r: ET.SubElement(r[0], 'failure')),
            ('disabled_suite', lambda r: r.set('disabled', '1')),
            ('skipped_suite', lambda r: r.set('skipped', '1'))]:
        altered = deepcopy(suite); change(altered); ET.ElementTree(altered).write(xml)
        fails('ctest_' + label, lambda: driver.ctest_result(xml), 'CTest')
    for side in ('old', 'new'):
        folder = out / f'teacher-{side}'
        folder.mkdir()
        driver.write(folder / 'result.json', dict(passed=True))
        (folder / 'latest.ckpt').write_bytes(b'fixture only')
        (folder / 'transcript.txt').write_bytes(b'test speech')
    assert len(driver.teacher_parity(out)['exact_artifacts']) == 2
    (out / 'teacher-new/latest.ckpt').write_bytes(b'changed')
    fails('teacher_checkpoint_changed', lambda: driver.teacher_parity(out), 'checkpoint or speech differs')
    (out / 'teacher-new/latest.ckpt').write_bytes(b'fixture only')
    (out / 'teacher-new/extra.ckpt').write_bytes(b'extra')
    fails('teacher_extra_checkpoint', lambda: driver.teacher_parity(out), 'artifact sets differ')

    prior = dict(status='declared_before_full_411m_learning', specification=dict(version='prose-411m-v1'))
    completion = dict(complete=True, protocol=prior, baseline_assessments_exact=True,
        matched_source_replay_and_speech_exposure=True, matched_replay_payloads=True,
        random_initialization=True, imported_weights=False, reserved_tests_scored=False,
        reproduction_admitted=False, native_learning_commands=1, native_assessment_commands=80,
        controls=[dict(stage=i, profile='105m') for i in range(1, 5)],
        rows=[dict(stage=i, profile='411m') for i in range(1, 5)])
    completed_header(prior, completion)
    for key in ('complete', 'baseline_assessments_exact', 'matched_source_replay_and_speech_exposure',
                'matched_replay_payloads', 'random_initialization', 'imported_weights',
                'reserved_tests_scored', 'reproduction_admitted'):
        changed = deepcopy(completion); changed[key] = not changed[key]
        fails('predecessor_' + key, lambda: completed_header(prior, changed), 'redecessor' if
              key in ('imported_weights', 'reserved_tests_scored', 'reproduction_admitted') else 'Predecessor')
    for key in ('native_learning_commands', 'native_assessment_commands'):
        changed = deepcopy(completion); changed[key] -= 1
        fails('predecessor_' + key, lambda: completed_header(prior, changed), 'coverage differs')
    for key in ('controls', 'rows'):
        changed = deepcopy(completion); changed[key].pop()
        fails('predecessor_' + key, lambda: completed_header(prior, changed), 'missing assessment')

    events = []
    identity = dict(pid=4242, executable='python.exe', creation_filetime=1234)

    class Gate:
        def __init__(self, pid, executable):
            events.append('open')
            self.identity = dict(identity)
        def wait(self):
            events.append('wait')
            return 0
        def close(self):
            events.append('close')

    def plan_for(folder):
        folder.mkdir()
        marker = folder / 'pinned.txt'; marker.write_bytes(b'unchanged')
        e = dict(old='old.exe', new='new.exe', python='python.exe', ctest='ctest.exe', probe='probe.exe')
        value = dict(version='membrane-acceptance-v1', status='declared_before_gpu_execution',
            root=str(ROOT), workspace='workspace', predecessor='predecessor', predecessor_protocol_sha256='pin',
            wait_for_driver=dict(identity), executables=e, ctest=actual,
            authenticated_inputs={str(marker): driver.sha(marker)},
            commands=driver.command_plan(folder, e['python'], e['ctest'], e['old'], e['new'], e['probe']))
        driver.write(folder / 'protocol.json', value)
        return value

    def no_launch(folder, expected, protocol_sha256=None):
        with patch.object(driver.subprocess, 'run', side_effect=AssertionError('Native launch before acceptance gates')) as launch:
            with redirect_stdout(io.StringIO()) as output:
                fails(folder.name, lambda: driver.execute(folder, protocol_sha256 or driver.sha(folder / 'protocol.json')), expected)
            (folder / 'host-mocked-dispatch.txt').write_text(output.getvalue(), encoding='utf-8')
            assert launch.call_count == 0
        assert not (folder / 'commands.json').exists()

    folder = out / 'changed_input'; plan_for(folder)
    (folder / 'pinned.txt').write_bytes(b'changed')
    no_launch(folder, 'input changed')
    folder = out / 'changed_protocol'; plan = plan_for(folder)
    identity_before = driver.sha(folder / 'protocol.json')
    plan['unexpected'] = True; driver.write(folder / 'protocol.json', plan)
    no_launch(folder, 'protocol identity changed', identity_before)
    folder = out / 'changed_command'; plan = plan_for(folder)
    plan['commands'].reverse(); driver.write(folder / 'protocol.json', plan)
    no_launch(folder, 'command plan changed')
    folder = out / 'changed_driver'; plan = plan_for(folder)
    plan['wait_for_driver']['creation_filetime'] += 1; driver.write(folder / 'protocol.json', plan)
    with patch.object(driver, 'ProcessGate', Gate):
        no_launch(folder, 'PID identity changed')
    folder = out / 'failed_driver'; plan_for(folder)
    with patch.object(driver, 'ProcessGate', Gate), patch.object(Gate, 'wait', side_effect=RuntimeError('predecessor exit 7')):
        no_launch(folder, 'predecessor exit 7')
    folder = out / 'incomplete_predecessor'; plan_for(folder)
    with patch.object(driver, 'ProcessGate', Gate), patch.object(driver, 'verify_predecessor', side_effect=ValueError('completion absent')):
        no_launch(folder, 'completion absent')
    folder = out / 'changed_after_wait'; plan_for(folder)

    def mutate():
        (folder / 'pinned.txt').write_bytes(b'changed while waiting')
        return 0

    with patch.object(driver, 'ProcessGate', Gate), patch.object(Gate, 'wait', side_effect=mutate), \
            patch.object(driver, 'verify_predecessor', return_value=dict(passed=True)):
        no_launch(folder, 'input changed')
    folder = out / 'changed_ctest'; plan_for(folder)
    with patch.object(driver, 'ProcessGate', Gate), patch.object(driver, 'verify_predecessor', return_value=dict(passed=True)), \
            patch.object(driver, 'checked_ctest', return_value=[]):
        no_launch(folder, 'CTest declaration changed')
    folder = out / 'mocked_success'; plan_for(folder)
    with patch.object(driver, 'ProcessGate', Gate), patch.object(driver, 'verify_predecessor', return_value=dict(passed=True)), \
            patch.object(driver, 'checked_ctest', return_value=actual), \
            patch.object(driver, 'collect_results', return_value=dict(passed=True)), \
            patch.object(driver.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, b'ok', b'')) as launch:
        with redirect_stdout(io.StringIO()) as output:
            driver.execute(folder, driver.sha(folder / 'protocol.json'))
        (folder / 'host-mocked-dispatch.txt').write_text(output.getvalue(), encoding='utf-8')
        assert launch.call_count == 20 and len(driver.read(folder / 'completed-commands.json')) == 20
        assert driver.read(folder / 'execution.json')['phase'] == 'complete'
    fails('duplicate_execution', lambda: driver.execute(folder, driver.sha(folder / 'protocol.json')), 'already started')

    # Real, short CPU-only processes exercise handle identity, normal exit and
    # nonzero exit. No sleep in this test exceeds one second.
    process_cases = []
    for code in (0, 7):
        process = subprocess.Popen([sys.executable, '-c', f'import time; time.sleep(.3); raise SystemExit({code})'])
        gate = ProcessGate(process.pid, sys.executable)
        try:
            if code:
                fails('real_nonzero_driver', gate.wait, 'exited with code 7')
            else:
                assert gate.wait() == 0
            process_cases.append(dict(exit_code=code, identity=gate.identity))
        finally:
            gate.close(); process.wait()
    process = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(.3)'])
    try:
        fails('real_wrong_executable', lambda: ProcessGate(process.pid, ROOT / 'build/synapticgenesis.exe'),
              'executable differs')
    finally:
        process.wait()
    driver.write(out / 'result.json', dict(passed=True, cuda_executed=False,
        native_acceptance_executed=False, mocked_success_not_device_evidence=True,
        command_count=20, ctest_cases=18, rejections=rejected, real_cpu_process_cases=process_cases,
        native_fixture_snapshot_comparisons=snapshot_checks,
        source_sha256={str(p): driver.sha(p) for p in (Path(__file__), ROOT / 'scripts/membrane_acceptance.py',
            ROOT / 'scripts/large_founder_verification.py', ROOT / 'scripts/process_gate.py', ROOT / 'tests/membrane_teacher_cli.py')}))
    print('PASS:', len(rejected), 'host negative controls; 20 commands checked, no CUDA work.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--checkpoint-fixtures', type=Path)
    args = parser.parse_args()
    check(args.out, args.checkpoint_fixtures)
