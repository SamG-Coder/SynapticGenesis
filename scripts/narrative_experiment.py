"""Continue all declared models on selected books; native CUDA performs learning."""
import argparse
from pathlib import Path
import shutil
import subprocess

from experiment_checkpoint import distribution, state_record
from extend_curriculum import prepare, read_schedule
from native_experiment import (NativeCommands, binding_scores, read, sha, verified_book_manifest,
                               verified_manifest, write)


BOOKS = {'reader': 13853, 'geography': 12228, 'narrative': 11757}
PROMPTS = ['The bird ', 'Once upon a time ']


def assess_books(native, checkpoint_path, prepared, out, endpoint, batches):
    records = {}
    for name, book in BOOKS.items():
        target = out / f'{name}-{endpoint}.json'
        native('evaluate', '--checkpoint', checkpoint_path, '--data', prepared / f'{book}.txt',
               '--batch', 16, '--context', 128, '--batches', batches, '--output', target)
        records[name] = read(target)
        assert records[name]['evaluated_bytes'] == 16 * 128 * batches
    return records


def samples(native, checkpoint_path, out, endpoint, count):
    records = []
    for index, prompt in enumerate(PROMPTS):
        target = out / f'sample-{endpoint}-{index}.txt'
        native('sample', '--checkpoint', checkpoint_path, '--prompt', prompt, '--tokens', count,
               '--temperature', .8, '--top-k', 40, '--seed', 42, '--graph', '--output', target)
        raw = target.read_bytes()
        assert raw.startswith(prompt.encode('ascii')) and len(raw) == len(prompt) + count
        generated = raw[len(prompt):]
        records.append(dict(prompt=prompt, generated_bytes=count, file_sha256=sha(target),
                            generated_latin1=generated.decode('latin1'),
                            generated_utf8_display=generated.decode('utf-8', errors='backslashreplace')))
    return records


def run(exe, parent, prepared, binding, out, smoke):
    exe, parent, prepared, binding = map(lambda path: path.resolve(), (exe, parent, prepared, binding))
    source = read(parent / 'comparison.json')
    prior = source['protocol']
    assert prior['status'] == ('smoke_only' if smoke else 'declared_before_training')
    assert prior['online_endpoints'] == ([64, 96, 128] if smoke else [34000, 67000, 130000])
    assert prior['seeds'] == ([1337] if smoke else [1337, 2026, 31415])
    assert prior['arms'] == ['selective', 'wide-selective', 'associative']
    assert sha(exe) == prior['executable_sha256']
    assert sha(parent / 'curriculum.sg') == prior['schedule_sha256']
    assert sha(binding / 'manifest.json') == prior['prepared_manifest_sha256']
    verified_manifest(binding)
    books = verified_book_manifest(prepared, Path('data/sources-stories-v1.json'))
    assert {r['id'] for r in books['sources'] if r['split'] == 'train'} == {572, 5312, 43936, 11, 236, 17314, 17396}
    assert {r['id'] for r in books['sources'] if r['split'] == 'validation'} == set(BOOKS.values())
    assert {r['id'] for r in books['sources'] if r['split'] == 'test'} == {15659, 902}
    baseline = prior['online_endpoints'][-1]
    endpoints = [256, 384] if smoke else [160000, 190000]
    count, batches = (64, 2) if smoke else (384, 32)
    log, save = (64, 128) if smoke else (5000, 5000)
    parents = [row for row in source['runs'] if row['online_updates'] == baseline]
    assert len(parents) == len(prior['seeds']) * 3
    for row in parents:
        path = parent / f'{row["seed"]}-{row["arm"]}/checkpoint-{baseline}.ckpt'
        assert sha(path) == row['checkpoint_sha256']
    out.mkdir(parents=True, exist_ok=False)
    admission = prepare(parent / 'curriculum.sg', prepared / 'train.dat', out / 'edition', endpoints[-1],
                        rate_scale=.25, scope='new', answer_scale=1,
                        holdouts=[prepared / f'{r["id"]}.txt' for r in books['sources'] if r['split'] != 'train'])
    assert admission['previous_stages'] == 3 and admission['added_documents'] == 7
    schedule = out / 'edition/curriculum.sg'
    _, stages = read_schedule(schedule)
    boundaries = [len(stage['content'].split(b'\x1e')) for stage in stages]
    paths = ['scripts/narrative_experiment.py', 'scripts/native_experiment.py',
             'scripts/extend_curriculum.py', 'scripts/experiment_checkpoint.py']
    protocol = dict(status='smoke_only' if smoke else 'declared_before_training',
        seeds=prior['seeds'], arms=prior['arms'], models=prior['models'], baseline_online_updates=baseline,
        online_endpoints=endpoints, parent_comparison_sha256=sha(parent / 'comparison.json'),
        parent_directory=parent.as_posix(), parent_schedule_sha256=prior['schedule_sha256'],
        parent_checkpoints={f'{r["seed"]}-{r["arm"]}': r['checkpoint_sha256'] for r in parents},
        executable_sha256=sha(exe), source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        source_sha256={name: sha(name) for name in paths}, source_admission=admission,
        prepared_directory=prepared.as_posix(), prepared_manifest_sha256=sha(prepared / 'manifest.json'),
        book_validation={name: dict(id=book, sha256=sha(prepared / f'{book}.txt')) for name, book in BOOKS.items()},
        binding_directory=binding.as_posix(), probe_sha256=prior['probe_sha256'],
        chunk=prior['chunk'], learning_rate=prior['learning_rate'], later_stage_multiplier=.25,
        new_answer_emphasis=1, previous_lesson_answer_emphasis=64,
        replay_policy='stage', replay_every=4, replay_slots=prior['replay_slots'],
        replay_quotas=[prior['replay_slots'] // 4] * 4,
        speak_every=prior['speak_every'], generated_bytes_per_speech=96, live_prompt='The bird ',
        graph_speech=True, learning_tf32=True, validation_strict_fp32=True,
        evaluation_batch=16, evaluation_context=128, evaluation_batches=batches,
        sample_prompts=PROMPTS, sample_bytes=count, sample_temperature=.8, sample_top_k=40, sample_seed=42,
        log_every=log, save_every=save,
        primary_endpoint='Change in held-out Velveteen Rabbit byte loss from the authenticated parent to the final endpoint.',
        secondary_endpoints='Earlier reader and geography loss change; retained development binding; all fixed samples; '
                            'live-loop cost and final seven-round 512-byte graph decoding.',
        comparison='Continue every existing final architecture/seed, without selecting the best model. '
                   'One pooled narrative stage; this does not test short-to-long ordering or teacher feedback.',
        state_policy='Resume complete learning history through native append-only admission. The fourth stage '
                     'rebalances stage reservoirs to equal quarters while preserving lifetime counters. '
                     'The usual document reset clears transient state at the new source boundary.',
        timing='Sequential GPU commands; live timing includes replay, speech, logging and saves, '
               'but excludes setup, held-out assessment and fixed sample generation.',
        reserved_test_evaluated=False,
        limits='Three paired seeds, fixed exposure, no hyperparameter search or stopping based on intermediate quality. '
               'Validation uses deterministic sampled 128-byte windows, not full-book or conversational evaluation. '
               'Continuation uses this project\'s own from-scratch models, not imported pretrained weights.')
    write(out / 'protocol.json', protocol)
    native, rows = NativeCommands(exe, out), []
    for seed_index, seed in enumerate(prior['seeds']):
        arms = prior['arms'][seed_index % 3:] + prior['arms'][:seed_index % 3]
        for arm in arms:
            name = f'{seed}-{arm}'
            directory = out / name
            directory.mkdir()
            ancestor = parent / name / f'checkpoint-{baseline}.ckpt'
            current = directory / f'checkpoint-{baseline}.ckpt'
            shutil.copyfile(ancestor, current)
            initial = state_record(current)
            assert initial['checkpoint_sha256'] == protocol['parent_checkpoints'][name]
            shutil.copyfile(parent / name / f'development-{baseline}.json', directory / f'development-{baseline}.json')
            baseline_binding = read(directory / f'development-{baseline}.json')
            baseline_record = dict(seed=seed, arm=arm, **initial, cumulative_narrative_seconds=0.,
                books=assess_books(native, current, prepared, directory, baseline, batches),
                development={k: v for k, v in baseline_binding.items() if k != 'results'},
                samples=samples(native, current, directory, baseline, count))
            rows.append(baseline_record)
            write(out / 'partial.json', rows)
            elapsed = 0.
            for index, endpoint in enumerate(endpoints):
                extra = ['--extend-curriculum', schedule] if index == 0 else []
                native('live', '--resume', current, '--curriculum', parent / 'curriculum.sg' if index == 0 else schedule,
                       *extra, '--out', directory, '--updates', endpoint, '--prompt', 'The bird ',
                       '--validation', prepared / '13853.txt', '--eval-batches', batches,
                       '--log-every', log, '--save-every', save)
                session = read(directory / 'session.json')
                assert session['online_updates'] == endpoint and session['curriculum_stage'] == 4
                assert session['online_document_count'] == 7
                current = directory / f'checkpoint-{endpoint}.ckpt'
                shutil.copyfile(directory / 'latest.ckpt', current)
                state = state_record(current)
                assert distribution(current, boundaries) == protocol['replay_quotas']
                elapsed += session['elapsed_seconds']
                row = dict(seed=seed, arm=arm, **state, cumulative_narrative_seconds=elapsed, session=session,
                           books=assess_books(native, current, prepared, directory, endpoint, batches),
                           samples=samples(native, current, directory, endpoint, count))
                row.update(binding_scores(native, current, binding, directory, endpoint, splits=('development',)))
                if endpoint == endpoints[-1]:
                    native('decode-bench', '--checkpoint', current, '--out', directory / 'decode',
                           '--tokens', 96 if smoke else 512, '--rounds', 2 if smoke else 7)
                    row['decode'] = read(directory / 'decode/benchmark.json')
                assert sha(current) == state['checkpoint_sha256']
                rows.append(row)
                write(out / 'partial.json', rows)
                print(f'{seed} {arm} at {endpoint}: narrative={row["books"]["narrative"]["loss_nats_per_byte"]:.5f}, '
                      f'reader={row["books"]["reader"]["loss_nats_per_byte"]:.5f}, '
                      f'binding={row["development"]["joint_accuracy"]:.4f}, segment={session["elapsed_seconds"]:.2f}s', flush=True)
            assert sha(ancestor) == initial['checkpoint_sha256']
    assert sha(exe) == protocol['executable_sha256']
    assert sha(parent / 'comparison.json') == protocol['parent_comparison_sha256']
    assert sha(parent / 'curriculum.sg') == prior['schedule_sha256']
    assert sha(schedule) == admission['schedule_sha256']
    assert all(sha(name) == identity for name, identity in protocol['source_sha256'].items())
    verified_book_manifest(prepared, Path('data/sources-stories-v1.json'))
    verified_manifest(binding)
    write(out / 'comparison.json', dict(protocol=protocol, native_commands=len(native.commands), runs=rows))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exe', type=Path, default=Path('build/synapticgenesis.exe'))
    parser.add_argument('--parent', type=Path, default=Path('runs/associative-long-panel'))
    parser.add_argument('--prepared', type=Path, default=Path('data/prepared/stories-v1'))
    parser.add_argument('--binding', type=Path, default=Path('data/prepared/binding-diversity-v1'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    run(args.exe, args.parent, args.prepared, args.binding, args.out, args.smoke)
