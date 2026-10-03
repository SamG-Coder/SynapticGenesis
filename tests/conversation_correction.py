"""Host checks for correction isolation, durable history and process handoff.

Synthetic replies check the score definition only; they are not model results.
"""
import argparse
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

import conversation_experiment as runner
from conversation_material import material, prepare, prompt
from conversation_measure import score, assess
from membrane_study_state import f32, read_state
from native_experiment import read
from prose_founder import file_hash, write


def reject(label, action, records):
    try:
        action()
    except (ValueError, RuntimeError):
        records.append(label)
    else:
        raise AssertionError('Accepted invalid fixture: ' + label)


def scoring_checks():
    assert score(b'Five.\nwrong second sentence', ['Five.'])['first_line_matches']
    assert score(b'\n\n five!\n', ['Five.'])['first_line_matches']
    for answer in (b'Three.\n', b'Five, actually six.\n', b'The answer might be five.\n', b'\xffFive.\n', b''):
        assert not score(answer, ['Five.'])['first_line_matches']
    return dict(first_line_only=True, following_text_not_judged=True, invalid_utf8_never_matches=True,
                synthetic_replies_only=True)


def history_checks(prepared, rejections):
    initial = prepared['base']['state']
    # Construct the intended arithmetic of one direct exposure independently.
    candidate = deepcopy(initial)
    candidate['counters'].update(online_updates=216295, global_updates=270368,
        observed_pairs=27680129 + prepared['target_pairs_per_pass'],
        replay_updates=54073, curriculum_stage=5)
    candidate['hyperparameters'][0] = f32(initial['hyperparameters'][7] / 4)
    candidate['replay_groups'].append([61, 6, 6, 0, 0])
    session = dict(shared_weights=True, persistent_membranes=True, trains_on_generated_text=False,
                   answer_emphasized_documents=6, online_first_document=55, online_document_count=6,
                   global_updates=270368)
    runner.verify_learning(candidate, initial, prepared, 1, session)
    for label, mutate in (
        ('reset-old-online-counter', lambda s: s['counters'].update(online_updates=6)),
        ('lost-replay-history', lambda s: s['counters'].update(replay_updates=1)),
        ('wrong-source-byte-count', lambda s: s['counters'].update(observed_pairs=27680129)),
        ('changed-learning-rate', lambda s: s['hyperparameters'].__setitem__(0, f32(.0003))),
        ('lost-old-stage-exposures', lambda s: s['replay_groups'][0].__setitem__(1, 0)),
        ('replaced-model-shape', lambda s: s.update(parameters=1)),
    ):
        bad = deepcopy(candidate); mutate(bad)
        reject(label, lambda: runner.verify_learning(bad, initial, prepared, 1, session), rejections)
    reject('learning-from-generated-mistakes', lambda: runner.verify_learning(candidate, initial, prepared, 1,
        dict(session, trains_on_generated_text=True)), rejections)
    return dict(ancestor_online_updates=216289, one_pass_source_updates=6, one_pass_replay_updates=1,
                expected_global_updates=270368, fixture_only=True)


def measurement_checks(directory, prepared):
    # Exercise raw-byte preservation and complete question/book coverage without CUDA.
    checkpoint = directory / 'fake.ckpt'; checkpoint.write_bytes(b'unchanged synthetic checkpoint')
    calls = []
    class Native:
        def __init__(self, executable, out): self.commands, self.out = [], out
        def __call__(self, *args):
            command = list(map(str, args)); self.commands.append(command); calls.append(command)
            output = Path(command[command.index('--output') + 1])
            if command[0] == 'sample':
                prefix = command[command.index('--prompt') + 1].encode('ascii')
                output.write_bytes(prefix + b'\xff' + b'x' * 95)
            else:
                write(output, dict(evaluated_bytes=65536, step=270361, loss_nats_per_byte=2., bits_per_byte=2.88539))
    with patch('conversation_measure.NativeCommands', Native), \
         patch('conversation_measure.read_state', return_value=prepared['base']['state']):
        result = assess('synthetic.exe', checkpoint, prepared, directory / 'assessment')
    assert len(calls) == 22 and len(result['answers']) == 18 and len(result['books']) == 4
    assert all(not x['first_line_matches'] and x['answer_hex'].startswith('ff') for x in result['answers'])
    assert result['checkpoint_unchanged'] and checkpoint.read_bytes() == b'unchanged synthetic checkpoint'
    return dict(mocked_native_commands=22, native_commands_launched=0, raw_bytes_preserved=True)


def handoff_checks(directory, prepared, rejections):
    identity = dict(pid=789, executable='synthetic-python.exe', creation_filetime=123)
    gate = Mock(identity=identity)
    gate.wait.side_effect = RuntimeError('synthetic failed predecessor')
    def declaration(name):
        out = directory / name; out.mkdir()
        write(out / 'protocol.json', dict(version='conversation-plan-v1', source_checkout=str(ROOT),
            workspace=str(Path.cwd().resolve()), material=prepared, authenticated_inputs={}, wait_for_driver=identity))
        return out, file_hash(out / 'protocol.json')
    with patch.object(runner, 'ProcessGate', return_value=gate), patch.object(runner, 'authenticate'), \
         patch.object(runner, 'previous_complete') as completion, patch.object(runner, 'assess') as evaluation:
        out, digest = declaration('failed-predecessor')
        reject('nonzero-predecessor-exit', lambda: runner.execute(out, digest), rejections)
        assert not evaluation.called and not read(out / 'failure.json')['native_work_started']
        gate.identity = dict(identity, creation_filetime=999)
        out, digest = declaration('reused-pid')
        reject('reused-predecessor-pid', lambda: runner.execute(out, digest), rejections)
        assert not evaluation.called
        gate.identity = identity; gate.wait.side_effect = None; gate.wait.return_value = 0
        completion.side_effect = ValueError('synthetic incomplete baseline')
        out, digest = declaration('incomplete-baseline')
        reject('incomplete-baseline', lambda: runner.execute(out, digest), rejections)
        assert not evaluation.called and not read(out / 'failure.json')['native_work_started']
    return dict(failed_reused_or_incomplete_predecessor_starts_no_native_work=True)


def run(report):
    require_new = not report.exists()
    if not require_new: raise ValueError('Preserve previous check reports')
    source = read(ROOT / 'data/conversation-corrections-v1.json')
    targets, questions, pairs = material(source)
    assert len(targets.split(b'\x1e')) == 6 and len(questions) == 18
    # Independent, reviewed expectations rather than deriving gold from model output.
    answers = {row['id']: row['answer'] for row in source['lessons']}
    assert answers['addition'] == 'Two plus three is five.' and 2 + 3 == 5
    assert answers['subtraction'] == 'Five minus two is three.' and 5 - 2 == 3
    assert answers['name'] == 'My name is SynapticGenesis.'
    assert answers['water'] == 'Water is made of hydrogen and oxygen.'
    rejections = []
    for label, mutate in (
        ('duplicate-lesson-id', lambda s: s['lessons'][1].update(id='greeting')),
        ('evaluation-question-in-target', lambda s: s['evaluation_only'][0].update(question='Hello!')),
        ('oversize-correction-window', lambda s: s['lessons'][0].update(question='x' * 200)),
        ('separator-in-correction', lambda s: s['lessons'][0].update(question='x\x1ey')),
        ('duplicate-evaluation-id', lambda s: s['evaluation_only'][0].update(id='name')),
    ):
        bad = deepcopy(source); mutate(bad)
        reject(label, lambda: material(bad), rejections)
    with tempfile.TemporaryDirectory(prefix='sg-conversation-') as temporary:
        directory = Path(temporary)
        prepared = prepare(directory / 'material')
        history = history_checks(prepared, rejections)
        measurement = measurement_checks(directory, prepared)
        handoff = handoff_checks(directory, prepared, rejections)
        reject('overwrite-prepared-edition', lambda: prepare(directory / 'material'), rejections)
        expected = runner.learning_arguments('base.ckpt', 'old.sg', 'new.sg', 'out', 216295, True)
        assert '--resume' in expected and '--checkpoint' not in expected and '--extend-curriculum' in expected
        assert runner.learning_arguments('later.ckpt', 'new.sg', 'new.sg', 'out', 216385, False).count('--extend-curriculum') == 0
        actual = dict(base_checkpoint_sha256=prepared['base']['checkpoint_sha256'],
            authenticated_inputs=len(prepared['authenticated_inputs']),
            target_bytes=len(targets), target_pairs_per_pass=pairs,
            original_stages_preserved=4, correction_documents=6, evaluation_questions=18,
            quantitative_contexts_protected=prepared['protected_quantitative_contexts'],
            quantitative_statements_protected=prepared['protected_quantitative_statements'])
    result = dict(passed=True, source_and_checkpoint_admission=actual, scoring=scoring_checks(),
        history=history, measurement=measurement, handoff=handoff, rejected_inputs=rejections,
        native_commands_launched=0, model_quality_verified=False,
        implementation_sha256={p.as_posix():file_hash(p) for p in (Path(__file__),
            *(ROOT / 'scripts' / name for name in ('conversation_material.py', 'conversation_measure.py', 'conversation_experiment.py')))})
    write(report, result)
    print('Conversation host checks passed:', len(rejections), 'rejections; zero native commands.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--report', required=True, type=Path)
    run(parser.parse_args().report)
