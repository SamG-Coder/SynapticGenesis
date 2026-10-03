"""Run the declared correction comparisons immediately, without queue dependencies."""
from pathlib import Path
from datetime import datetime, timezone
from conversation_experiment import (authenticate, assess, contextual_control, learning_arguments,
    verify_learning, read_state, NativeCommands, read, write, file_hash, require)

out = Path('runs/conversation-correction-immediate-v1').resolve()
require(not out.exists(), 'Use a fresh output directory')
source = Path('runs/conversation-correction-v1').resolve()
plan = read(source / 'protocol.json')
authenticate(plan['authenticated_inputs'])
material = plan['material']; spec = material['specification']
out.mkdir()
write(out / 'protocol.json', dict(mode='immediate-user-request', created_utc=datetime.now(timezone.utc).isoformat(),
    original_protocol_sha256=file_hash(source / 'protocol.json'), original_plan=plan,
    concurrent_gpu_workload=True, isolated_timing=False, predecessor_required=False))
results = []
try:
    write(out / 'execution.json', dict(phase='before-correction', native_work_started=True))
    baseline = assess(plan['runtime'], spec['base'], material, out / 'before')
    print('Before correction:', baseline['first_line_scores'], flush=True)
    contextual = contextual_control(plan['runtime'], spec['base'], material, out / 'context-only')
    dialogue = [dict(question=lesson['question'], model_before=answer, teacher_answer=lesson['answer'],
        feedback_kind='confirm' if answer['first_line_matches'] else 'correct')
        for lesson, answer in zip(material['source']['lessons'], baseline['answers'][:6])]
    write(out / 'dialogue.json', dialogue)
    checkpoint = spec['base']; schedule = source / 'material/curriculum/curriculum.sg'
    for index, passes in enumerate(spec['passes']):
        directory = out / f'practice-{passes}'; directory.mkdir()
        write(out / 'execution.json', dict(phase='correction-learning', passes=passes, native_work_started=True, completed_rounds=len(results)))
        native = NativeCommands(plan['runtime'], directory)
        native(*learning_arguments(checkpoint, Path(spec['original_curriculum']) if index == 0 else schedule,
            schedule, directory, spec['base_online_updates'] + 6 * passes, index == 0))
        checkpoint = directory / 'latest.ckpt'
        session = read(directory / 'session.json')
        verify_learning(read_state(checkpoint), material['base']['state'], material, passes, session)
        measured = assess(plan['runtime'], checkpoint, material, out / f'after-{passes}')
        record = dict(passes=passes, assessment=measured, session=session,
            retention_changes=[dict(book=now['book'], loss_change_nats_per_byte=now['loss_nats_per_byte']-old['loss_nats_per_byte'])
                for old, now in zip(baseline['books'], measured['books'])])
        results.append(record); write(out / 'completed-rounds.json', results)
        print('After', passes, 'passes:', measured['first_line_scores'], flush=True)
    authenticate(plan['authenticated_inputs'])
    require(file_hash(spec['base']) == spec['base_sha256'], 'Ancestor changed')
    write(out / 'result.json', dict(complete=True, baseline=baseline, contextual_control=contextual,
        dialogue=dialogue, rounds=results, native_learning_commands=3, native_assessment_commands=94,
        concurrent_gpu_workload=True, isolated_timing=False, ancestor_unchanged=True))
    write(out / 'execution.json', dict(phase='complete', completed_rounds=3))
except BaseException as error:
    write(out / 'failure.json', dict(error=str(error), completed_rounds=len(results)))
    raise
