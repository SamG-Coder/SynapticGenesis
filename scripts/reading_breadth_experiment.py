"""Compare bounded native live reading from small and expanded selections."""
import argparse
from pathlib import Path
import shutil
import subprocess

from experiment_checkpoint import checkpoint, state_record
from extend_curriculum import prepare
from live_reading_experiment import assess
from narrative_experiment import samples
from native_experiment import NativeCommands, read, sha, verified_manifest, write
from reading_breadth_sources import prepare_assessment, source_exposure, verify_sources


def run(args):
    parent = Path('runs/teacher-retention-panel').resolve()
    previous = read(parent / 'comparison.json')
    old = previous['protocol']
    old_schedule = parent / 'edition/curriculum.sg'
    assert old['online_endpoints'][-1] == 190000
    assert sha(old_schedule) == old['source_admission']['schedule_sha256']
    books, binding = Path(old['prepared_directory']), Path(old['binding_directory'])
    verified_manifest(binding)
    assert sha(binding / 'manifest.json') == old['binding_manifest_sha256']
    editions = verify_sources()
    seeds = [1337] if args.smoke else [1337, 2026, 31415]
    endpoints = [16, 116, 546] if args.smoke else [116, 546, 1092]
    arms = ['starter-4', 'broader-4', 'starter-1', 'broader-1']
    policies = {arm: dict(source=arm.split('-')[0], replay_every=int(arm.split('-')[1])) for arm in arms}
    ancestors = {}
    for seed in seeds:
        path = parent / f'{seed}-associative-control/checkpoint-190000.ckpt'
        recorded = next(r for r in previous['runs'] if r['seed'] == seed
                        and r['arm'] == 'associative-control' and r['online_updates'] == 190000)
        assert sha(path) == recorded['checkpoint_sha256']
        meta, extra, *_ = checkpoint(path)
        assert meta[17] == 5 and meta[24] == 190000 and extra[2] == 4 and extra[16] == 4
        ancestors[str(seed)] = dict(path=path.as_posix(), sha256=sha(path), state=state_record(path))
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    assessment = prepare_assessment(out / 'assessment', editions)
    starter = Path(editions['starter']['directory']) / 'train.dat'
    broader = out / 'broader-train.dat'
    broader.write_bytes((out / 'assessment/train.dat').read_bytes())
    additions = {'starter': starter, 'broader': broader}
    protected = [Path(e['directory']) / (split + '.dat') for e in editions.values()
                 for split in ('validation', 'test')]
    book_manifest = read(books / 'manifest.json')
    protected += [books / f"{s['id']}.txt" for s in book_manifest['sources'] if s['split'] != 'train']
    admissions = {name: prepare(old_schedule, source, out / ('curriculum-' + name), 190000 + endpoints[-1],
                               rate_scale=.25, scope='new', answer_scale=1, holdouts=protected)
                  for name, source in additions.items()}
    exposure = {name: {str(end): source_exposure(path.read_bytes(), end) for end in endpoints}
                for name, path in additions.items()}
    assert source_exposure(broader.read_bytes(), 546)['complete_passes'] == 1
    assert source_exposure(starter.read_bytes(), 116)['complete_passes'] == 1
    paths = [str(p).replace('\\', '/') for p in sorted(Path('src').glob('*.cu')) + sorted(Path('src').glob('*.cuh'))]
    paths += ['scripts/reading_breadth_experiment.py', 'scripts/reading_breadth_sources.py',
              'scripts/live_reading_experiment.py', 'scripts/adaptation_sources.py',
              'scripts/native_experiment.py', 'scripts/narrative_experiment.py', 'scripts/experiment_checkpoint.py',
              'scripts/extend_curriculum.py', 'tests/reading_breadth_experiment.py',
              'tests/live_reading_experiment.py', 'tests/stage_replay_reference.py',
              'scripts/summarize_reading_breadth.py', 'docs/reading-breadth-experiment.md',
              'experiments/adaptation_probe.cu', 'experiments/adaptation_io.cuh',
              'scripts/sample_live_reading_questions.py', 'scripts/native_demonstration.py']
    protocol = dict(status='smoke_only' if args.smoke else 'declared_before_training',
        seeds=seeds, arms=arms, arm_policies=policies, baseline_online_updates=190000,
        additional_endpoints=endpoints, online_endpoints=[190000 + n for n in endpoints], primary_endpoint=546,
        ancestors=ancestors, parent_comparison_sha256=sha(parent / 'comparison.json'),
        parent_schedule=old_schedule.as_posix(), parent_schedule_sha256=sha(old_schedule),
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        source_sha256={path: sha(path) for path in paths}, executable_sha256=sha(args.exe),
        diagnostic_executable_sha256=sha(args.probe), source_editions=editions,
        admissions=admissions, exposure=exposure, assessment=assessment,
        book_directory=books.as_posix(), book_validation=old['book_validation'],
        binding_directory=binding.as_posix(), binding_manifest_sha256=sha(binding / 'manifest.json'),
        probe_sha256=old['probe_sha256'], evaluation_batches=2 if args.smoke else 32,
        chunk=128, learning_rate=.000075, replay_slots=1024, prompt='The bird ', speak_every=500, speech_bytes=96,
        sample_bytes=64 if args.smoke else 384, log_every=16 if args.smoke else 256, save_every=100000000,
        learning_math='Inherited TF32 and ordered gradient reductions; native shared forward path.',
        assessment_math='Strict FP32, reset at each 128-target window and exact short tail; no updates.',
        source_order='Starter ten whole stories first; broader appends 34 whole training stories in selected order. Sequential cycles.',
        state_policy='Complete native resume and append-only admission retain weights, Adam, recurrence, speech RNG and replay history.',
        primary_comparison='Broader versus starter after 546 observations, separately for replay every 4 and every 1. '
                           'Broader sees 67188 new targets once; starter sees 66704 targets with repetition. This is not exact equal-byte exposure.',
        continuation_comparison='After 1092 observations: two complete broader passes versus about 9.4 starter passes. '
                                'No checkpoint chosen by assessment score.',
        gate=dict(endpoint=546, development_improves_from_ancestor=True, both_development_sets_no_worse_than_starter=True,
                  binding_ancestor_drop_limit_pp=5, binding_starter_drop_limit_pp=2.5,
                  earlier_book_ancestor_loss_limit=.05, earlier_book_starter_loss_limit=.02,
                  required_seeds='all three, evaluated separately per replay cadence', automatic_promotion=False),
        generation='Two fixed 384-byte story continuations and seven-round 512-byte decode at final endpoints; '
                   'all six prior question prompts on all twelve final models, 192 bytes each, no retries.',
        timing='Sequential GPU runs, condition order reversed on alternating seeds. Live segment time includes replay, speech, logging and saves; '
               'excludes startup and assessment. Different short tails and replay selections prevent pure compute or hardware claims.',
        limits='Content identity, breadth and repetition change together; one order and three related ancestors. '
               'Shared development probes and same-source creators limit generalization. Passing is a candidate result, not general conversation or lifelong learning.',
        teacher_feedback=False, generated_text_targets=False, reserved_tests_scored=False)
    write(out / 'protocol.json', protocol)
    for name in ('native-commands', 'diagnostic-commands'):
        (out / name).mkdir()
    native, probe = NativeCommands(args.exe, out / 'native-commands'), NativeCommands(args.probe, out / 'diagnostic-commands')
    baselines, records = [], []
    for index, seed in enumerate(seeds):
        original = Path(ancestors[str(seed)]['path'])
        baseline = out / f'{seed}-baseline'
        baseline.mkdir()
        baselines.append(dict(seed=seed, **assess(native, probe, original, baseline, 190000, protocol, out / 'assessment')))
        write(out / 'baselines.json', baselines)
        for arm in arms if index % 2 == 0 else arms[::-1]:
            policy = policies[arm]
            directory = out / f'{seed}-{arm}'
            directory.mkdir()
            current = directory / 'checkpoint-190000.ckpt'
            shutil.copyfile(original, current)
            schedule = out / ('curriculum-' + policy['source']) / 'curriculum.sg'
            elapsed = 0.
            for point, additional in enumerate(endpoints):
                end = 190000 + additional
                flags = ['--extend-curriculum', schedule] if point == 0 else []
                native('live', '--resume', current, '--curriculum', old_schedule if point == 0 else schedule, *flags,
                       '--out', directory, '--updates', end, '--prompt', protocol['prompt'],
                       '--replay-every', policy['replay_every'], '--validation', books / '13853.txt',
                       '--eval-batches', protocol['evaluation_batches'], '--log-every', protocol['log_every'],
                       '--save-every', protocol['save_every'])
                current = directory / f'checkpoint-{end}.ckpt'
                shutil.copyfile(directory / 'latest.ckpt', current)
                session = read(directory / 'session.json')
                write(directory / f'session-{end}.json', session)
                elapsed += session['elapsed_seconds']
                row = dict(seed=seed, arm=arm, additional_observations=additional, session=session,
                           cumulative_live_seconds=elapsed,
                           **assess(native, probe, current, directory, end, protocol, out / 'assessment'))
                if additional == endpoints[-1]:
                    row['samples'] = samples(native, current, directory, end, protocol['sample_bytes'])
                    native('decode-bench', '--checkpoint', current, '--out', directory / 'decode',
                           '--tokens', 64 if args.smoke else 512, '--rounds', 2 if args.smoke else 7)
                    row['decode'] = read(directory / 'decode/benchmark.json')
                records.append(row)
                write(out / 'partial.json', records)
                print(seed, arm, additional, 'observations complete', flush=True)
    assert all(sha(path) == digest for path, digest in protocol['source_sha256'].items())
    assert sha(args.exe) == protocol['executable_sha256'] and sha(args.probe) == protocol['diagnostic_executable_sha256']
    assert all(sha(r['path']) == r['sha256'] for r in ancestors.values())
    assert verify_sources() == editions
    write(out / 'comparison.json', dict(protocol=protocol, baselines=baselines, runs=records,
          native_commands=len(native.commands), diagnostic_commands=len(probe.commands), originals_unchanged=True, complete=True))
    print(out / 'comparison.json', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--exe', type=Path, default=Path('build/synapticgenesis.exe'))
    p.add_argument('--probe', type=Path, default=Path('build/adaptation-probe/synaptic-adaptation-probe.exe'))
    run(p.parse_args())
