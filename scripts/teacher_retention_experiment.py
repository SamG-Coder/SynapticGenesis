"""Fixed paired narrative retention study; all neural computation is native CUDA."""
import argparse
from pathlib import Path
import shutil
import subprocess

from experiment_checkpoint import checkpoint, distribution, state_record
from extend_curriculum import prepare, read_schedule
from narrative_experiment import BOOKS, PROMPTS, assess_books, samples
from native_experiment import NativeCommands, binding_scores, read, sha, verified_book_manifest, verified_manifest, write


VARIANTS = ['control', 'teacher']
TEMPERATURE, STRENGTH = 2., .5
SOURCE_FILES = ['scripts/teacher_retention_experiment.py', 'scripts/experiment_checkpoint.py',
                'scripts/narrative_experiment.py', 'scripts/native_experiment.py', 'scripts/extend_curriculum.py',
                'tests/teacher_retention_experiment.py', 'tests/binding_learned_oracle.py']


def run(args):
    exe, parent, prepared, binding, previous = [p.resolve() for p in
        (args.exe, args.parent, args.prepared, args.binding, args.previous)]
    out = args.out.resolve()
    source, old_study = read(parent/'comparison.json'), read(previous/'comparison.json')
    prior, earlier = source['protocol'], old_study['protocol']
    expected_status = 'smoke_only' if args.smoke else 'declared_before_training'
    assert prior['status'] == earlier['status'] == expected_status
    assert prior['seeds'] == earlier['seeds'] == ([1337] if args.smoke else [1337, 2026, 31415])
    architectures = ['selective', 'wide-selective', 'associative']
    assert prior['arms'] == earlier['arms'] == architectures
    baseline = 128 if args.smoke else 130000
    endpoints = [256, 384] if args.smoke else [160000, 190000]
    assert prior['online_endpoints'][-1] == earlier['baseline_online_updates'] == baseline
    assert earlier['online_endpoints'] == endpoints
    assert sha(parent/'curriculum.sg') == prior['schedule_sha256'] == earlier['parent_schedule_sha256']
    assert sha(parent/'comparison.json') == earlier['parent_comparison_sha256']
    assert sha(binding/'manifest.json') == prior['prepared_manifest_sha256']
    verified_manifest(binding)
    books = verified_book_manifest(prepared, Path('data/sources-stories-v1.json'))
    assert {r['id'] for r in books['sources'] if r['split'] == 'train'} == {572, 5312, 43936, 11, 236, 17314, 17396}
    assert {r['id'] for r in books['sources'] if r['split'] == 'validation'} == set(BOOKS.values())
    assert {r['id'] for r in books['sources'] if r['split'] == 'test'} == {902, 15659}
    parents = [r for r in source['runs'] if r['online_updates'] == baseline]
    assert len(parents) == 3*len(prior['seeds'])
    for row in parents:
        name = f'{row["seed"]}-{row["arm"]}'
        assert sha(parent/name/f'checkpoint-{baseline}.ckpt') == row['checkpoint_sha256'] == earlier['parent_checkpoints'][name]
    out.mkdir(parents=True, exist_ok=False)
    admission = prepare(parent/'curriculum.sg', prepared/'train.dat', out/'edition', endpoints[-1],
                        rate_scale=.25, scope='new', answer_scale=1,
                        holdouts=[prepared/f'{r["id"]}.txt' for r in books['sources'] if r['split'] != 'train'])
    schedule = out/'edition/curriculum.sg'
    assert sha(schedule) == earlier['source_admission']['schedule_sha256']
    _, stages = read_schedule(schedule)
    boundaries = [len(s['content'].split(b'\x1e')) for s in stages]
    count, batches = (64, 2) if args.smoke else (384, 32)
    log, save = (64, 128) if args.smoke else (5000, 5000)
    arms = [f'{a}-{v}' for a in architectures for v in VARIANTS]
    policy = dict(kind='one frozen pre-narrative self', temperature=TEMPERATURE, strength=STRENGTH,
                  teacher_math='strict_fp32', eligible_source='All three previously observed curriculum stages',
                  source_sha256=sha(stages[2]['source']), eligible_documents=boundaries[2],
                  memory_budget_mib=512, generated_text_targets=False,
                  actual_target_coefficient=1, answer_weights_apply_to_both_terms=True)
    protocol = dict(status=expected_status, seeds=prior['seeds'], architectures=architectures,
        variants=VARIANTS, arms=arms, models={f'{a}-{v}': prior['models'][a] for a in architectures for v in VARIANTS},
        baseline_online_updates=baseline, online_endpoints=endpoints,
        parent_directory=parent.as_posix(), parent_comparison_sha256=sha(parent/'comparison.json'),
        parent_schedule_sha256=prior['schedule_sha256'],
        parent_checkpoints={f'{r["seed"]}-{r["arm"]}': r['checkpoint_sha256'] for r in parents},
        previous_narrative_directory=previous.as_posix(), previous_narrative_sha256=sha(previous/'comparison.json'),
        executable_sha256=sha(exe), source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        source_sha256={p: sha(p) for p in SOURCE_FILES}, source_admission=admission,
        prepared_directory=prepared.as_posix(), prepared_manifest_sha256=sha(prepared/'manifest.json'),
        binding_directory=binding.as_posix(), binding_manifest_sha256=sha(binding/'manifest.json'),
        probe_sha256=prior['probe_sha256'], teacher_policy=policy,
        book_validation={name: dict(id=book, sha256=sha(prepared/f'{book}.txt')) for name, book in BOOKS.items()},
        chunk=128, learning_rate=prior['learning_rate'], later_stage_multiplier=.25,
        new_answer_emphasis=1, previous_lesson_answer_emphasis=64,
        replay_policy='stage', replay_every=4, replay_slots=prior['replay_slots'],
        replay_quotas=[prior['replay_slots']//4]*4, source_boundaries=boundaries,
        speak_every=prior['speak_every'], generated_bytes_per_speech=96, live_prompt='The bird ',
        graph_speech=True, learning_tf32=True, validation_strict_fp32=True,
        evaluation_batch=16, evaluation_context=128, evaluation_batches=batches,
        sample_prompts=PROMPTS, sample_bytes=count, sample_temperature=.8, sample_top_k=40, sample_seed=42,
        log_every=log, save_every=save, reserved_test_evaluated=False,
        primary_endpoint='Teacher minus matched control complete development binding-group accuracy at the final endpoint, separately for each architecture and seed.',
        secondary_endpoints='Paired new-narrative, earlier-reader and geography byte loss; greedy complete groups; all fixed samples; live cost and teacher buffers; final decoding.',
        candidate_gate='A promising setting must improve final binding-group accuracy in all nine pairs and keep each of the three book losses within +0.02 nats/byte of its matched control, while improving narrative loss from its own parent. This is an engineering gate on repeated development data, not broad reliability.',
        order='Rotate architecture order by seed; alternate branch order by seed index plus canonical architecture index; GPU commands sequential.',
        timing='Full live segments include source/replay updates, speech, logging and saves; exclude setup, teacher packaging, held-out scoring and fixed samples. Report teacher GPU bytes separately.',
        limits='No coefficient search, early stopping, best-seed selection, external pretrained teacher, population crossover or extra selected content. This isolates retention guidance from the previous self. Three seeds and repeated development probes cannot establish general conversation or lifelong learning.')
    write(out/'protocol.json', protocol)  # Before packaging, evaluation or learning.
    native, rows, bundles = NativeCommands(exe, out), [], {}
    for seed_index, seed in enumerate(prior['seeds']):
        rotated = architectures[seed_index % 3:] + architectures[:seed_index % 3]
        for architecture in rotated:
            architecture_index = architectures.index(architecture)
            name = f'{seed}-{architecture}'
            ancestor = parent/name/f'checkpoint-{baseline}.ckpt'
            bundle = out/(name+'-teacher-bundle')
            native('teacher-pack', '--teacher-a', ancestor, '--data', stages[2]['source'], '--out', bundle,
                   '--temperature', TEMPERATURE, '--strength', STRENGTH)
            bundles[name] = {p.name: sha(p) for p in bundle.iterdir() if p.is_file()}
            write(out/'bundles.json', bundles)
            base_dir = out/(name+'-baseline'); base_dir.mkdir()
            baseline_record = dict(seed=seed, architecture=architecture, **state_record(ancestor),
                cumulative_narrative_seconds=0., books=assess_books(native, ancestor, prepared, base_dir, baseline, batches),
                samples=samples(native, ancestor, base_dir, baseline, count))
            baseline_record.update(binding_scores(native, ancestor, binding, base_dir, baseline, splits=('development',)))
            variants = VARIANTS if (seed_index+architecture_index) % 2 == 0 else list(reversed(VARIANTS))
            for variant in variants:
                arm = f'{architecture}-{variant}'
                directory = out/f'{seed}-{arm}'; directory.mkdir()
                current = directory/f'checkpoint-{baseline}.ckpt'; shutil.copyfile(ancestor, current)
                for source_file in base_dir.iterdir():
                    shutil.copyfile(source_file, directory/source_file.name)
                rows.append(dict(**baseline_record, arm=arm, variant=variant))
                write(out/'partial.json', rows)
                elapsed = 0.
                teacher_flags = ['--teacher-bundle', bundle, '--teacher-memory-mib', 512] if variant == 'teacher' else []
                for index, endpoint in enumerate(endpoints):
                    assert sha(exe) == protocol['executable_sha256']
                    extension = ['--extend-curriculum', schedule] if index == 0 else []
                    native('live', '--resume', current, '--curriculum', parent/'curriculum.sg' if index == 0 else schedule,
                           *extension, '--out', directory, '--updates', endpoint, '--prompt', 'The bird ',
                           '--validation', prepared/'13853.txt', '--eval-batches', batches,
                           '--log-every', log, '--save-every', save, *teacher_flags)
                    session = read(directory/'session.json')
                    assert session['online_updates'] == endpoint and session['online_document_count'] == 7
                    current = directory/f'checkpoint-{endpoint}.ckpt'
                    shutil.copyfile(directory/'latest.ckpt', current)
                    state = state_record(current)
                    assert distribution(current, boundaries) == protocol['replay_quotas']
                    assert ('teaching' in state) == (variant == 'teacher')
                    elapsed += session['elapsed_seconds']
                    record = dict(seed=seed, architecture=architecture, variant=variant, arm=arm, **state,
                                  cumulative_narrative_seconds=elapsed, session=session,
                                  books=assess_books(native, current, prepared, directory, endpoint, batches),
                                  samples=samples(native, current, directory, endpoint, count))
                    record.update(binding_scores(native, current, binding, directory, endpoint, splits=('development',)))
                    if endpoint == endpoints[-1]:
                        native('decode-bench', '--checkpoint', current, '--out', directory/'decode',
                               '--tokens', 96 if args.smoke else 512, '--rounds', 2 if args.smoke else 7)
                        record['decode'] = read(directory/'decode/benchmark.json')
                    assert sha(current) == state['checkpoint_sha256']
                    rows.append(record); write(out/'partial.json', rows)
                    print(f'{seed} {arm} at {endpoint}: binding={record["development"]["joint_accuracy"]:.4f}, '
                          f'narrative={record["books"]["narrative"]["loss_nats_per_byte"]:.5f}, '
                          f'segment={session["elapsed_seconds"]:.2f}s', flush=True)
            assert sha(ancestor) == protocol['parent_checkpoints'][name]
    assert sha(exe) == protocol['executable_sha256']
    assert all(sha(p) == h for p, h in protocol['source_sha256'].items())
    assert sha(parent/'comparison.json') == protocol['parent_comparison_sha256']
    assert sha(previous/'comparison.json') == protocol['previous_narrative_sha256']
    for name, files in bundles.items():
        assert all(sha(out/(name+'-teacher-bundle')/file) == h for file, h in files.items())
    verified_manifest(binding); verified_book_manifest(prepared, Path('data/sources-stories-v1.json'))
    write(out/'comparison.json', dict(protocol=protocol, bundles=bundles, native_commands=len(native.commands), runs=rows))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--exe', type=Path, default=Path('build/synapticgenesis.exe'))
    p.add_argument('--parent', type=Path, default=Path('runs/associative-long-panel'))
    p.add_argument('--previous', type=Path, default=Path('runs/narrative-panel'))
    p.add_argument('--prepared', type=Path, default=Path('data/prepared/stories-v1'))
    p.add_argument('--binding', type=Path, default=Path('data/prepared/binding-diversity-v1'))
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--smoke', action='store_true')
    run(p.parse_args())
