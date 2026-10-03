"""Serialize native residency checks after the direct correction experiment."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import statistics
import subprocess
import sys
import time

from native_experiment import NativeCommands, read
from process_gate import ProcessGate
from prose_founder import file_hash, write

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tests'))
from resident_conversation import check_events, quoted, require, run as native_checks, session_file

PREDECESSOR = Path('runs/conversation-correction-v1')
PREDECESSOR_HASH = '0e159d226c62ad46fc9ba206c3b29131ec4118b787c7f3717aa909f9babf1c46'
BASELINE = Path('D:/SynapticGenesis/runs/membrane-acceptance-fix-worktree/build/synapticgenesis.exe')
CANDIDATE = ROOT / 'build/resident-conversation/synaptic-resident-conversation.exe'


def authenticate(pins):
    for path, digest in pins.items():
        require(file_hash(path) == digest, 'Resident acceptance input changed: ' + str(path))


def completed_correction():
    require(file_hash(PREDECESSOR / 'protocol.json') == PREDECESSOR_HASH and
            not (PREDECESSOR / 'failure.json').exists(), 'Correction predecessor changed or failed')
    plan, result, execution = [read(PREDECESSOR / name) for name in ('protocol.json', 'result.json', 'execution.json')]
    require(result['complete'] and result['protocol_sha256'] == PREDECESSOR_HASH and result['protocol'] == plan and
            execution['phase'] == 'complete' and execution['completed_rounds'] == 3 and
            result['native_learning_commands'] == 3 and result['native_assessment_commands'] == 94 and
            len(result['rounds']) == 3 and [r['passes'] for r in result['rounds']] == [1, 16, 256] and
            result['ancestor_unchanged'] and not result['reproduction_admitted'],
            'Direct correction experiment has not completed its declared comparisons')
    for stage in [result['baseline'], *(r['assessment'] for r in result['rounds'])]:
        require(stage['complete'] and stage['checkpoint_unchanged'] and stage['native_commands'] == 22 and
                file_hash(stage['checkpoint']) == stage['checkpoint_sha256'],
                'Correction endpoint or assessment changed')
    return result


def compare_real(out, result):
    """Repeat the entire 105M correction trajectory in one process."""
    out.mkdir()
    plan = result['protocol']; material = plan['material']; spec = material['specification']
    generation = spec['generation']
    require(generation == dict(bytes=96, top_k=1, temperature=1, seed=42, graph=True),
            'Expected the fixed direct-correction sampling policy')
    stages = [('before', result['baseline'])] + [(f"after-{r['passes']}", r['assessment']) for r in result['rounds']]
    lines, questions, comparisons = [], {}, []
    for index, (label, stage) in enumerate(stages):
        require(len(stage['answers']) == len(material['questions']) == 18, 'Missing correction questions or answers')
        if index:
            lines.extend([f"learn {stage['state']['counters']['online_updates']}", f'save {label}'])
        for row, reference in zip(material['questions'], stage['answers']):
            ident = label + '-' + row['id']
            require(row['id'] == reference['id'] and row['question'] == reference['question'], 'Question order differs')
            lines.append('ask ' + ident + ' ' + quoted(row['question']))
            questions[ident] = row['question']
            comparisons.append((ident, reference, stage))
    lines.append('quit')
    script = out / 'session.sgconversation'
    session_file(script, lines)
    measured = out / 'resident'
    native = NativeCommands(CANDIDATE, out)
    started = time.perf_counter()
    native('run', '--resume', spec['base'], '--curriculum', spec['original_curriculum'],
        '--extend-curriculum', PREDECESSOR.resolve() / 'material/curriculum/curriculum.sg',
        '--script', script, '--out', measured, '--answer-bytes', 96, '--answer-top-k', 1,
        '--answer-temperature', 1, '--answer-seed', 42)
    process_seconds = time.perf_counter() - started
    events, answers = check_events(measured, questions)
    for answer, (ident, reference, stage) in zip(answers, comparisons):
        require(answer['id'] == ident and answer['answer'].encode('latin-1').hex() == reference['answer_hex'] and
                answer['online_updates'] == stage['state']['counters']['online_updates'] and
                answer['global_updates'] == stage['state']['counters']['global_updates'],
                'Resident answer bytes or learning counters differ from the correction experiment')
    for label, stage in stages[1:]:
        require(file_hash(measured / f'checkpoint-{label}.ckpt') == stage['checkpoint_sha256'],
                'Resident full checkpoint differs from the recorded 105M correction trajectory')
    require(file_hash(measured / 'final.ckpt') == stages[-1][1]['checkpoint_sha256'],
            'Final questions changed the 105M model')
    rates = [a['resident_decode_bytes_per_second'] for a in answers]
    learned = [r for r in events if r['event'] == 'learned']
    comparison = dict(passed=True, profile='105m', parameters=events[0]['parameters'],
        full_checkpoints_exact=3, raw_answers_exact=len(answers), checkpoint_unchanged_by_final_questions=True,
        native_commands=1, questions_and_learning_share_one_process=True,
        process_seconds=process_seconds, question_scratch_bytes=events[0]['question_scratch_bytes'],
        resident_decode_bytes_per_second=dict(median=statistics.median(rates), minimum=min(rates), maximum=max(rates)),
        answer_metrics=answers, learning_metrics=learned,
        timing_limits='Resident decode wall time excludes checkpoint load, graph capture and prompt prefill, which are separately recorded. It includes CPU sampling and host/device synchronization. Whole-process time includes learning and checkpoint saves. Shared-desktop timing is not a hardware-isolated benchmark.',
        language_quality={label:stage['first_line_scores'] for label, stage in stages},
        quality_limits='Byte and state equivalence to the direct-correction experiment, not evidence of better language quality.')
    write(out / 'result.json', comparison)
    return comparison


def declare(out, wait_pid, wait_executable):
    out = Path(out).resolve()
    require(not out.exists(), 'Use a fresh resident acceptance directory')
    require(file_hash(PREDECESSOR / 'protocol.json') == PREDECESSOR_HASH, 'Correction declaration changed')
    previous = read(PREDECESSOR / 'protocol.json')
    pins = dict(previous['authenticated_inputs'])
    paths = [PREDECESSOR / 'protocol.json', CANDIDATE, BASELINE, Path(sys.executable),
             ROOT / 'build/resident-conversation/synaptic-conversation-protocol-test.exe',
             ROOT / 'scripts/resident_conversation_acceptance.py', ROOT / 'tests/resident_conversation.py',
             ROOT / 'scripts/process_gate.py', ROOT / 'scripts/native_experiment.py', ROOT / 'scripts/prose_founder.py',
             ROOT / 'CMakeLists.txt', ROOT / 'build.ps1', ROOT / 'tests/conversation_protocol.cpp',
             ROOT / 'tests/resident_conversation_host.py', ROOT / 'reports/resident-conversation-host-checks.json',
             ROOT / 'reports/resident-conversation-build.json']
    tracked = subprocess.check_output(['git', 'ls-files', 'src', 'experiments'], cwd=ROOT, text=True).splitlines()
    paths.extend(ROOT / name for name in tracked)
    pins.update({str(p.resolve()):file_hash(p) for p in paths})
    require(BASELINE.resolve() == Path(previous['runtime']).resolve(), 'Preserved runtime differs')
    authenticate(pins)
    gate = ProcessGate(wait_pid, wait_executable)
    try:
        require(gate.kernel.WaitForSingleObject(gate.handle, 0) == 258, 'Declare while the correction driver is live')
        plan = dict(version='resident-conversation-acceptance-v1', created_utc=datetime.now(timezone.utc).isoformat(),
            source_checkout=str(ROOT), source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            workspace=str(Path.cwd().resolve()), predecessor=str(PREDECESSOR), predecessor_protocol_sha256=PREDECESSOR_HASH,
            wait_for_driver=gate.identity, authenticated_inputs=pins, native_fixture_cases=4,
            fixture_cases='Ordinary and membrane-cost 0.001 associative models, each trained in FP32 and TF32.',
            fixture_native_commands=33, real_model_native_commands=1, real_model_questions=72,
            real_model_checkpoints=3, reproduction_admitted=False)
        out.mkdir(parents=True)
        write(out / 'protocol.json', plan)
        print('Declared resident state/answer equivalence, stdin checks and one complete 105M correction replay.', flush=True)
        return plan
    finally:
        gate.close()


def execute(out, protocol_sha256):
    out = Path(out).resolve()
    require(file_hash(out / 'protocol.json') == protocol_sha256 and not (out / 'execution.json').exists(),
            'Resident acceptance declaration changed or already started')
    plan = read(out / 'protocol.json')
    require(plan['version'] == 'resident-conversation-acceptance-v1' and
            plan['source_checkout'] == str(ROOT) and plan['workspace'] == str(Path.cwd().resolve()),
            'Resident acceptance checkout or workspace differs')
    authenticate(plan['authenticated_inputs'])
    gate, started = None, False
    try:
        waiting = plan['wait_for_driver']
        gate = ProcessGate(waiting['pid'], waiting['executable'])
        require(gate.identity == waiting, 'Correction driver identity changed')
        write(out / 'execution.json', dict(phase='waiting-for-correction', native_work_started=False,
            protocol_sha256=protocol_sha256, predecessor=waiting))
        print('Waiting on the held correction-driver handle:', waiting['pid'], flush=True)
        gate.wait()
        write(out / 'predecessor-exit.json', dict(**gate.identity, exit_code=0))
        gate.close(); gate = None
        correction = completed_correction()
        authenticate(plan['authenticated_inputs'])
        started = True
        write(out / 'execution.json', dict(phase='native-contract-checks', native_work_started=True,
            protocol_sha256=protocol_sha256))
        checks = native_checks(out / 'native', CANDIDATE, BASELINE)
        require(checks['passed'] and checks['legacy_commands'] == 28 and checks['resident_file_commands'] == 4 and
                checks['interactive_commands'] == 1, 'Native fixture coverage differs')
        write(out / 'execution.json', dict(phase='real-model-replay', native_work_started=True,
            protocol_sha256=protocol_sha256))
        measured = compare_real(out / '105m', correction)
        authenticate(plan['authenticated_inputs'])
        result = dict(passed=True, protocol=plan, protocol_sha256=protocol_sha256, native_contract=checks,
            real_model=measured, predecessor_result_sha256=file_hash(PREDECESSOR / 'result.json'),
            native_commands=34, reproduction_admitted=False)
        write(out / 'result.json', result)
        write(out / 'execution.json', dict(phase='complete', native_work_started=True, protocol_sha256=protocol_sha256))
        print('Resident conversation acceptance complete.', flush=True)
        return result
    except BaseException as error:
        write(out / 'failure.json', dict(error=str(error), native_work_started=started, artifacts_preserved=True))
        raise
    finally:
        if gate: gate.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    creation = sub.add_parser('declare')
    creation.add_argument('--out', required=True, type=Path)
    creation.add_argument('--wait-pid', required=True, type=int)
    creation.add_argument('--wait-executable', required=True, type=Path)
    execution = sub.add_parser('run')
    execution.add_argument('--out', required=True, type=Path)
    execution.add_argument('--protocol-sha256', required=True)
    args = parser.parse_args()
    if args.command == 'declare': declare(args.out, args.wait_pid, args.wait_executable)
    else: execute(args.out, args.protocol_sha256)
