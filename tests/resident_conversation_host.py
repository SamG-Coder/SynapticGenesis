"""Host checks for the queued resident acceptance driver; no model execution."""
from pathlib import Path
import json
import sys
import tempfile
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'tests'))
import resident_conversation_acceptance as runner
from resident_conversation import check_events, session_file, quoted
from native_experiment import read
from prose_founder import file_hash, write


def run(report):
    if report.exists(): raise ValueError('Preserve previous reports')
    rejections = []
    def reject(name, call):
        try: call()
        except (ValueError, RuntimeError): rejections.append(name)
        else: raise AssertionError('Accepted invalid fixture: ' + name)
    with tempfile.TemporaryDirectory(prefix='resident-host-') as temporary:
        directory = Path(temporary)
        before = dict(event='start', single_parameter_workspace=True, independent_question_recurrence=True, question_math='strict FP32')
        question, raw = 'What is water?', b'\xffHydrogen.\n'
        (directory / 'answer-a.bin').write_bytes(('Question: '+question+'\nAnswer: ').encode() + raw)
        answer = dict(event='answer', id='a', answer=raw.decode('latin-1'), raw_file='answer-a.bin',
            generated_bytes=len(raw), captured_now=True, decode_seconds=.1, resident_decode_bytes_per_second=110.)
        complete = dict(event='complete', questions=1, query_outputs_are_training_targets=False)
        def put(a=answer):
            (directory / 'events.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in (before,a,complete)))
        put(); check_events(directory, {'a':question})
        put(dict(answer, answer='invented output'))
        reject('changed-raw-answer', lambda: check_events(directory, {'a':question}))
        put(dict(answer, captured_now=False))
        reject('missing-first-graph-capture', lambda: check_events(directory, {'a':question}))
        put(dict(answer, decode_seconds=0))
        reject('nonpositive-generation-duration', lambda: check_events(directory, {'a':question}))
        put(); reject('missing-question', lambda: check_events(directory, {'a':question,'b':'another'}))
        identity = dict(pid=123, executable='synthetic-python.exe', creation_filetime=456)
        gate = Mock(identity=identity); gate.wait.return_value=0
        def declaration(label):
            out=directory/label; out.mkdir()
            write(out/'protocol.json',dict(version='resident-conversation-acceptance-v1', source_checkout=str(ROOT), workspace=str(Path.cwd().resolve()),
                authenticated_inputs={}, wait_for_driver=identity))
            return out, file_hash(out/'protocol.json')
        with patch.object(runner, 'ProcessGate', return_value=gate), patch.object(runner,'authenticate'), \
             patch.object(runner,'completed_correction') as predecessor, patch.object(runner,'native_checks') as native:
            gate.wait.side_effect=RuntimeError('synthetic failed predecessor')
            out,digest=declaration('failed-exit')
            reject('failed-predecessor',lambda:runner.execute(out,digest))
            assert not native.called and not read(out/'failure.json')['native_work_started']
            gate.wait.side_effect=None; gate.identity=dict(identity,creation_filetime=999)
            out,digest=declaration('reused-pid')
            reject('reused-predecessor-pid',lambda:runner.execute(out,digest))
            assert not native.called
            gate.identity=identity; predecessor.side_effect=ValueError('synthetic incomplete correction')
            out,digest=declaration('incomplete-correction')
            reject('incomplete-correction',lambda:runner.execute(out,digest))
            assert not native.called and not read(out/'failure.json')['native_work_started']
        path=directory/'commands.sgconversation'
        session_file(path, ['ask a '+quoted('A "quoted" question?'), 'quit'])
        assert path.read_bytes() == b'SGCONVERSATION1\nask a "A \\"quoted\\" question?"\nquit\n'
    result=dict(passed=True,rejected_inputs=rejections,raw_non_utf8_bytes_preserved=True,
        failed_predecessors_start_no_native_work=True,native_commands_launched=0,model_quality_verified=False,
        source_sha256={str(p):file_hash(p) for p in (Path(__file__),ROOT/'scripts/resident_conversation_acceptance.py',ROOT/'tests/resident_conversation.py')})
    write(report,result)
    print('Resident host checks passed:',len(rejections),'rejections; zero native commands.',flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--report',required=True,type=Path)
    run(parser.parse_args().report)
