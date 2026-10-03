"""Admit selected stories into complete live histories; compare rehearsal budgets."""
import argparse
from pathlib import Path
import shutil
import subprocess

from adaptation_sources import prepare_schedule, verified_edition
from experiment_checkpoint import checkpoint, state_record
from extend_curriculum import prepare, read_schedule
from narrative_experiment import assess_books, samples
from native_experiment import NativeCommands, binding_scores, read, sha, verified_manifest, write


def assess(native, probe, model, directory, endpoint, protocol, edition):
    """Score a disposable zero-update view without replacing a live checkpoint."""
    before = sha(model)
    destination = directory / f'reading-{endpoint}'
    probe('run', '--checkpoint', model, '--mode', 'carry', '--seed', 1337,
          '--train', edition / 'train.dat', '--validation', edition / 'validation.dat',
          '--schedule', edition / 'evaluation.sg', '--out', destination,
          '--steps', 0, '--endpoints', '0', '--lr', .000075)
    result = read(destination / 'result.json')
    assert result['session_presented_pairs'] == result['end_updates'] == 0
    reading = read(destination / 'evaluation-0.json')
    assert reading['weights_optimizer_unchanged']
    row = dict(**state_record(model), reading=reading,
               reading_report_sha256=sha(destination / 'evaluation-0.json'),
               books=assess_books(native, model, Path(protocol['book_directory']), directory,
                                  endpoint, protocol['evaluation_batches']))
    row.update(binding_scores(native, model, Path(protocol['binding_directory']), directory,
                              endpoint, splits=('development',)))
    assert sha(model) == before
    return row


def run(args):
    parent = Path('runs/teacher-retention-panel').resolve()
    previous = read(parent / 'comparison.json')
    old = previous['protocol']
    assert old['online_endpoints'][-1] == 190000
    old_schedule = parent / 'edition/curriculum.sg'
    assert sha(old_schedule) == old['source_admission']['schedule_sha256']
    stories = Path('data/prepared/early-readers-v1').resolve()
    manifest = verified_edition(stories, Path('data/sources-early-readers-v1.json'))
    books, binding = Path(old['prepared_directory']), Path(old['binding_directory'])
    verified_manifest(binding)
    assert sha(binding / 'manifest.json') == old['binding_manifest_sha256']
    for value in old['book_validation'].values():
        assert sha(books / f"{value['id']}.txt") == value['sha256']
    seeds = [1337] if args.smoke else [1337, 2026, 31415]
    common = [16, 64, 512] if args.smoke else [64, 256, 1024, 4096]
    matched = 320 if args.smoke else 2560
    arms = ['every-4', 'every-1']
    endpoints = {'every-4': common, 'every-1': sorted(common + [matched])}
    ancestors = {}
    for seed in seeds:
        name = f'{seed}-associative-control'
        path = parent / name / 'checkpoint-190000.ckpt'
        recorded = next(r for r in previous['runs'] if r['seed'] == seed
                        and r['arm'] == 'associative-control' and r['online_updates'] == 190000)
        assert sha(path) == recorded['checkpoint_sha256']
        meta, extra, _, _ = checkpoint(path)
        assert meta[17] == 5 and meta[24] == 190000 and extra[2] == 4 and extra[16] == 4
        ancestors[str(seed)] = dict(path=path.as_posix(), sha256=sha(path), state=state_record(path))
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    edition = out / 'reading-edition'
    edition.mkdir()
    for name in ('train.dat', 'validation.dat', 'manifest.json', 'source-spec.json', 'ATTRIBUTION.md'):
        shutil.copyfile(stories / name, edition / name)
    schedule = prepare_schedule(edition / 'train.dat', 1337, 1, edition / 'evaluation.sg')
    holdouts = [stories / 'validation.dat', stories / 'test.dat']
    holdouts += [books / f"{v['id']}.txt" for v in old['book_validation'].values()]
    admission = prepare(old_schedule, edition / 'train.dat', out / 'curriculum', 190000 + common[-1],
                        rate_scale=.25, scope='new', answer_scale=1, holdouts=holdouts)
    assert admission['previous_stages'] == 4 and admission['added_documents'] == 10
    _, stages = read_schedule(out / 'curriculum/curriculum.sg')
    native_sources = sorted(Path('src').glob('*.cu')) + sorted(Path('src').glob('*.cuh'))
    sources = [str(p).replace('\\', '/') for p in native_sources] + [
        'scripts/live_reading_experiment.py', 'scripts/experiment_checkpoint.py',
        'scripts/native_experiment.py', 'scripts/adaptation_sources.py', 'scripts/extend_curriculum.py',
        'scripts/narrative_experiment.py', 'tests/live_reading_experiment.py',
        'tests/stage_replay_reference.py', 'scripts/summarize_live_reading.py',
        'docs/live-reading-replay.md', 'experiments/adaptation_probe.cu',
        'experiments/adaptation_io.cuh']
    protocol = dict(status='smoke_only' if args.smoke else 'declared_before_training',
        seeds=seeds, arms=arms, baseline_online_updates=190000, additional_endpoints=endpoints,
        online_endpoints=[190000 + v for v in common],
        common_exposure_endpoints=common, matched_optimizer_count_pair={'every-4': common[-1], 'every-1': matched},
        matched_additional_optimizer_updates=common[-1] * 5 // 4,
        ancestors=ancestors, parent_comparison_sha256=sha(parent / 'comparison.json'),
        parent_schedule=old_schedule.as_posix(), parent_schedule_sha256=sha(old_schedule),
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        source_sha256={p: sha(p) for p in sources}, executable_sha256=sha(args.exe),
        diagnostic_executable_sha256=sha(args.probe),
        source_edition=manifest, source_admission=admission,
        source_boundaries=[len(s['content'].split(b'\x1e')) for s in stages],
        evaluation_schedule_sha256=schedule['schedule_sha256'],
        reading_train_sha256=sha(edition / 'train.dat'), reading_validation_sha256=sha(edition / 'validation.dat'),
        book_directory=books.as_posix(), book_validation=old['book_validation'],
        binding_directory=binding.as_posix(), binding_manifest_sha256=sha(binding / 'manifest.json'),
        probe_sha256=old['probe_sha256'], evaluation_batches=2 if args.smoke else 32,
        chunk=128, inherited_base_lr=.0003, new_stage_multiplier=.25,
        replay_slots=1024, replay_every={'every-4': 4, 'every-1': 1},
        replay_policy='stage', graph_speech=True, speak_every=500, speech_bytes=96,
        prompt='The bird ', sample_prompts=['The bird ', 'Once upon a time '],
        sample_bytes=64 if args.smoke else 384, log_every=16 if args.smoke else 256,
        save_every=100000000, learning_math='inherited TF32; ordered gradient reductions',
        assessment_math='strict FP32; reading diagnostic resets each 128-target window and exact shorter tail',
        learning_source_order='Complete documents in selected edition order, including final short chunks; repeats sequentially.',
        state_policy='Native --resume and append-only admission preserve complete history; both arms change only explicit replay cadence.',
        order='Alternate arm order by seed. GPU commands are sequential. Original checkpoints remain immutable.',
        primary_comparison='Final binding retention and new-story development loss at equal 4096 additional observations.',
        budget_comparison='Every-4 at 4096 versus every-1 at 2560: equal 5120 optimizer updates, not equal target bytes, speech or wall time.',
        candidate_gate='At equal final new-source exposure, every-1 must improve binding by at least 5 percentage points in all three seeds, '
                       'keep new development loss within +0.03 nats/byte and each earlier book within +0.02 of every-4, '
                       'and improve new development loss from its own ancestor in each seed. Also report the complete matched-update comparison. '
                       'Passing is a candidate result, not automatic promotion.',
        timing='Native live segments include observation, replay, speech, logging and saves; exclude process setup and held-out assessments. '
               'More segment boundaries in every-1 add setup/save overhead. Other desktop GPU contexts are active.',
        limits='One small new edition and three related learned ancestors. Repeated development probes; no global plasticity or general-language claim. '
               'Cadence changes replay selections and quantity; this is not a pure timing experiment with identical rehearsed windows.',
        teacher_feedback=False, reserved_tests_scored=False, generated_text_targets=False)
    write(out / 'protocol.json', protocol)
    for name in ('native-commands', 'diagnostic-commands'):
        (out / name).mkdir()
    native = NativeCommands(args.exe, out / 'native-commands')
    probe = NativeCommands(args.probe, out / 'diagnostic-commands')
    baselines, records = [], []
    for index, seed in enumerate(seeds):
        original = Path(ancestors[str(seed)]['path'])
        baseline_dir = out / f'{seed}-baseline'
        baseline_dir.mkdir()
        baseline = dict(seed=seed, **assess(native, probe, original, baseline_dir, 190000, protocol, edition))
        baselines.append(baseline)
        write(out / 'baselines.json', baselines)
        order = arms if index % 2 == 0 else arms[::-1]
        for arm in order:
            directory = out / f'{seed}-{arm}'
            directory.mkdir()
            current = directory / 'checkpoint-190000.ckpt'
            shutil.copyfile(original, current)
            elapsed = 0.
            for point_index, additional in enumerate(endpoints[arm]):
                endpoint = 190000 + additional
                old_path = old_schedule if point_index == 0 else out / 'curriculum/curriculum.sg'
                flags = ['--extend-curriculum', out / 'curriculum/curriculum.sg'] if point_index == 0 else []
                native('live', '--resume', current, '--curriculum', old_path, *flags,
                       '--out', directory, '--updates', endpoint, '--prompt', protocol['prompt'],
                       '--replay-every', protocol['replay_every'][arm],
                       '--validation', books / '13853.txt', '--eval-batches', protocol['evaluation_batches'],
                       '--log-every', protocol['log_every'], '--save-every', protocol['save_every'])
                current = directory / f'checkpoint-{endpoint}.ckpt'
                shutil.copyfile(directory / 'latest.ckpt', current)
                session = read(directory / 'session.json')
                write(directory / f'session-{endpoint}.json', session)
                assert session['online_updates'] == endpoint and session['online_document_count'] == 10
                elapsed += session['elapsed_seconds']
                row = dict(seed=seed, arm=arm, additional_observations=additional,
                           cumulative_live_seconds=elapsed, session=session,
                           **assess(native, probe, current, directory, endpoint, protocol, edition))
                if additional == common[-1]:
                    row['samples'] = samples(native, current, directory, endpoint, protocol['sample_bytes'])
                    native('decode-bench', '--checkpoint', current, '--out', directory / 'decode',
                           '--tokens', 64 if args.smoke else 512, '--rounds', 2 if args.smoke else 7)
                    row['decode'] = read(directory / 'decode/benchmark.json')
                records.append(row)
                write(out / 'partial.json', records)
                print(seed, arm, additional, 'observations complete', flush=True)
    assert all(sha(p) == digest for p, digest in protocol['source_sha256'].items())
    assert sha(args.exe) == protocol['executable_sha256'] and sha(args.probe) == protocol['diagnostic_executable_sha256']
    assert all(sha(r['path']) == r['sha256'] for r in ancestors.values())
    verified_edition(stories, Path('data/sources-early-readers-v1.json'))
    write(out / 'comparison.json', dict(protocol=protocol, baselines=baselines, runs=records,
          native_commands=len(native.commands), diagnostic_commands=len(probe.commands),
          originals_unchanged=True, complete=True))
    print(out / 'comparison.json', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exe', type=Path, default=Path('build/synapticgenesis.exe'))
    parser.add_argument('--probe', type=Path, default=Path('build/adaptation-probe/synaptic-adaptation-probe.exe'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    run(parser.parse_args())
