"""Check complete native breadth-study histories, common prefixes and assessments."""
import argparse
from collections import Counter
import importlib.util
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from adaptation_sources import fnv
from experiment_checkpoint import checkpoint, state_record
from extend_curriculum import read_schedule
from native_experiment import NativeCommands, read, sha, write
from reading_breadth_sources import grouped_reading, source_exposure, verify_sources
from stage_replay_reference import ReplayReference, schedule_identity

_spec = importlib.util.spec_from_file_location('live_reading_execution_audit',
                                             Path(__file__).with_name('live_reading_experiment.py'))
_audit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_audit)
argument, assessment, speech = _audit.argument, _audit.assessment, _audit.speech


def check(root, exe, probe, continuation_control=False):
    data = read(root / 'comparison.json')
    p = data['protocol']
    assert data['complete'] and data['originals_unchanged']
    assert not p['reserved_tests_scored'] and not p['teacher_feedback'] and not p['generated_text_targets']
    assert verify_sources() == p['source_editions']
    assert all(sha(path) == value for path, value in p['source_sha256'].items())
    assert sha(exe) == p['executable_sha256'] and sha(probe) == p['diagnostic_executable_sha256']
    assert sha('runs/teacher-retention-panel/comparison.json') == p['parent_comparison_sha256']
    assert sha(p['parent_schedule']) == p['parent_schedule_sha256']
    common = root / 'assessment'
    assert read(common / 'manifest.json') == p['assessment']
    assert all(sha(common / name) == digest for name, digest in p['assessment']['files'].items())
    docs = {split: (common / (split + '.dat')).read_bytes().split(b'\x1e') for split in ('train', 'validation')}
    for split in docs:
        expected, metadata = [], []
        for edition_name, edition in p['source_editions'].items():
            for source in edition['manifest']['sources']:
                if source['split'] == split:
                    expected.append((Path(edition['directory']) / (source['id'] + '.txt')).read_bytes())
                    metadata.append(dict(edition=edition_name, id=source['id'], body_sha256=source['body_sha256']))
        assert expected == docs[split]
        assert read(common / (split + '-documents.json')) == metadata
    assert len(docs['train']) == 44 and len(docs['validation']) == 7
    schedules = {}
    for name, admission in p['admissions'].items():
        path = root / ('curriculum-' + name) / 'curriculum.sg'
        raw, stages = read_schedule(path)
        assert len(stages) == 5 and admission['previous_stages'] == 4
        assert sha(path) == admission['schedule_sha256']
        new_docs = docs['train'][:10] if name == 'starter' else docs['train']
        assert stages[-1]['content'] == stages[-2]['content'] + b'\x1e' + b'\x1e'.join(new_docs)
        assert admission['added_documents'] == len(new_docs)
        assert [float(s['answer']) for s in stages] == [1, 64, 64, 1, 1]
        assert stages[-1]['scope'] == 'new' and float(stages[-1]['rate']) == .25
        assert stages[-1]['end_update'] == p['online_endpoints'][-1]
        for stage, recorded in zip(stages, admission['editions']):
            assert sha(stage['source']) == recorded['sha256']
        schedules[name] = (raw, stages)
        for end in p['additional_endpoints']:
            assert source_exposure(b'\x1e'.join(new_docs), end) == p['exposure'][name][str(end)]
    for book in p['book_validation'].values():
        assert sha(Path(p['book_directory']) / f"{book['id']}.txt") == book['sha256']
    assert sha(Path(p['binding_directory']) / 'manifest.json') == p['binding_manifest_sha256']
    assert sha(Path(p['binding_directory']) / 'development.sgprobe') == p['probe_sha256']['development.sgprobe']
    commands, diagnostics = read(root / 'native-commands/commands.json'), read(root / 'diagnostic-commands/commands.json')
    seeds, arms, count = len(p['seeds']), len(p['arms']), len(p['arms']) * len(p['additional_endpoints'])
    assert len(commands) == data['native_commands'] == seeds * (5 * count + 4 + 3 * arms)
    assert len(diagnostics) == data['diagnostic_commands'] == seeds * (count + 1)
    assert Counter(c[1] for c in commands) == dict(live=seeds * count, evaluate=seeds * (count + 1) * 3,
        **{'language-probes': seeds * (count + 1), 'sample': seeds * arms * 2, 'decode-bench': seeds * arms})
    for index, command in enumerate(commands, 1):
        assert Path(command[0]) == exe.resolve()
        assert (root / f'native-commands/command-{index:03d}.log').is_file()
        if command[1] == 'evaluate':
            assert Path(argument(command, 'data')).name in ('13853.txt', '12228.txt', '11757.txt')
            assert int(argument(command, 'batches')) == p['evaluation_batches']
        if command[1] == 'language-probes':
            assert sha(argument(command, 'probes')) == p['probe_sha256']['development.sgprobe']
    for index, command in enumerate(diagnostics, 1):
        assert Path(command[0]) == probe.resolve() and command[1] == 'run'
        assert argument(command, 'mode') == 'carry' and argument(command, 'steps') == argument(command, 'endpoints') == '0'
        assert Path(argument(command, 'train')) == common / 'train.dat'
        assert Path(argument(command, 'validation')) == common / 'validation.dat'
        assert Path(argument(command, 'schedule')) == common / 'evaluation.sg'
        assert (root / f'diagnostic-commands/command-{index:03d}.log').is_file()
    expected_order = [(f'{seed}-{arm}', 190000 + n) for index, seed in enumerate(p['seeds'])
                      for arm in (p['arms'] if index % 2 == 0 else p['arms'][::-1]) for n in p['additional_endpoints']]
    live = [c for c in commands if c[1] == 'live']
    assert [(Path(argument(c, 'out')).name, int(argument(c, 'updates'))) for c in live] == expected_order
    assert len(data['runs']) == seeds * count
    assert {(r['seed'], r['arm'], r['additional_observations']) for r in data['runs']} == {
        (seed, arm, n) for seed in p['seeds'] for arm in p['arms'] for n in p['additional_endpoints']}
    checked, prefix_checks, baseline_checks = [], [], []
    previous = read('runs/live-reading-panel/comparison.json')
    for seed in p['seeds']:
        ancestor = Path(p['ancestors'][str(seed)]['path'])
        assert sha(ancestor) == p['ancestors'][str(seed)]['sha256']
        baseline = next(r for r in data['baselines'] if r['seed'] == seed)
        assessment(baseline, root / f'{seed}-baseline', ancestor, p, docs)
        old_baseline = next(r for r in previous['baselines'] if r['seed'] == seed)
        assert baseline['reading']['train']['documents'][:10] == old_baseline['reading']['train']['documents']
        assert baseline['reading']['validation']['documents'][:3] == old_baseline['reading']['validation']['documents']
        baseline_checks.append(dict(seed=seed, previous_starter_assessment_identical=True))
        for arm in p['arms']:
            policy, directory = p['arm_policies'][arm], root / f'{seed}-{arm}'
            assert sha(directory / 'checkpoint-190000.ckpt') == sha(ancestor)
            schedule_raw, stages = schedules[policy['source']]
            reference = ReplayReference(ancestor, policy['replay_every'], schedule_identity(schedule_raw, stages),
                                        fnv(stages[-1]['content']), [len(d) for d in stages[-1]['content'].split(b'\x1e')])
            previous_end, elapsed = 190000, 0.
            for additional in p['additional_endpoints']:
                end = 190000 + additional
                row = next(r for r in data['runs'] if (r['seed'], r['arm'], r['online_updates']) == (seed, arm, end))
                model = directory / f'checkpoint-{end}.ckpt'
                reference.run_until(end)
                proof = reference.matches(model)
                assert ancestor.read_bytes()[256:288] == model.read_bytes()[256:288]
                assessment(row, directory, model, p, docs)
                grouped_reading(row['reading'], p['assessment'])
                session = read(directory / f'session-{end}.json')
                assert row['session'] == session and session['session_online_updates'] == end - previous_end
                assert row['observed_pairs'] - baseline['observed_pairs'] == p['exposure'][policy['source']][str(additional)]['presented_targets']
                assert session['shared_weights'] and session['persistent_membranes'] and not session['trains_on_generated_text']
                assert session['gradient_reductions'] == 'ordered_v1' and session['replay_every'] == policy['replay_every']
                assert session['online_document_count'] == (10 if policy['source'] == 'starter' else 44)
                assert abs(session['final_validation_loss'] - row['books']['reader']['loss_nats_per_byte']) < 1e-6
                elapsed += session['elapsed_seconds']
                assert elapsed == row['cumulative_live_seconds']
                command = next(c for c in live if Path(argument(c, 'out')) == directory and int(argument(c, 'updates')) == end)
                assert Path(argument(command, 'resume')) == directory / f'checkpoint-{previous_end}.ckpt'
                assert int(argument(command, 'replay-every')) == policy['replay_every']
                assert ('--extend-curriculum' in command) == (previous_end == 190000)
                if '--extend-curriculum' in command:
                    assert Path(argument(command, 'extend-curriculum')) == root / ('curriculum-' + policy['source']) / 'curriculum.sg'
                previous_end = end
                if additional == p['additional_endpoints'][-1]:
                    assert row['decode'] == read(directory / 'decode/benchmark.json')
                    assert row['decode']['generated_bytes_identical'] and row['decode']['state_logits_max_error'] == 0
                    for i, sample in enumerate(row['samples']):
                        path = directory / f'sample-{end}-{i}.txt'
                        raw = path.read_bytes()
                        assert sha(path) == sample['file_sha256']
                        assert raw.startswith(sample['prompt'].encode())
                        assert raw[len(sample['prompt']):].decode('latin1') == sample['generated_latin1']
                checked.append(dict(seed=seed, arm=arm, additional_observations=additional, checkpoint_sha256=sha(model), **proof))
        for cadence in (4, 1):
            a, b = [root / f'{seed}-{source}-{cadence}/checkpoint-190116.ckpt' for source in ('starter', 'broader')]
            assert checkpoint(a)[2] == checkpoint(b)[2], 'Same initial observations must preserve identical weights, moments and recurrence'
            prefix_checks.append(dict(seed=seed, replay_every=cadence, observations=116, learned_payload_identical=True))
    result = dict(passed=True, comparison_sha256=sha(root / 'comparison.json'),
        native_commands=len(commands), diagnostic_commands=len(diagnostics), branches=seeds * arms,
        checkpoint_assessments=seeds * (count + 1), exact_history_checks=checked,
        common_prefix_checks=prefix_checks, previous_baseline_checks=baseline_checks,
        native_sources_and_executables_authenticated=True, ancestors_unchanged=True,
        assessment_weights_and_moments_identical=True, source_splits_authenticated=True, reserved_tests_not_scored=True)
    controls = root / 'continuation-controls'
    if continuation_control or controls.exists():
        assert p['status'] == 'smoke_only'
        fresh = not controls.exists()
        if fresh:
            controls.mkdir()
        native = NativeCommands(exe, controls) if fresh else None
        recorded = [] if fresh else read(controls / 'commands.json')
        matches, seed = [], p['seeds'][0]
        for index, arm in enumerate(p['arms']):
            policy = p['arm_policies'][arm]
            destination = controls / arm
            arguments = ['live', '--resume', p['ancestors'][str(seed)]['path'], '--curriculum', p['parent_schedule'],
                '--extend-curriculum', root / ('curriculum-' + policy['source']) / 'curriculum.sg',
                '--out', destination, '--updates', p['online_endpoints'][-1], '--prompt', p['prompt'],
                '--replay-every', policy['replay_every'], '--validation', Path(p['book_directory']) / '13853.txt',
                '--eval-batches', p['evaluation_batches'], '--log-every', p['log_every'], '--save-every', p['save_every']]
            if native:
                native(*arguments)
            else:
                assert recorded[index] == [str(exe.resolve()), *map(str, arguments)]
            expected = root / f'{seed}-{arm}' / f"checkpoint-{p['online_endpoints'][-1]}.ckpt"
            assert expected.read_bytes() == (destination / 'latest.ckpt').read_bytes()
            assert speech(destination / 'transcript.txt') == speech(root / f'{seed}-{arm}/transcript.txt')
            assert state_record(expected)['generated_bytes'] - p['ancestors'][str(seed)]['state']['generated_bytes'] >= 96
            matches.append(dict(arm=arm, full_checkpoint_identical=True, speech_identical=True, crosses_speech_interval=True))
        result['uninterrupted_controls'] = matches
    write(root / 'execution-check.json', result)
    print({k:v for k,v in result.items() if k not in ('exact_history_checks', 'common_prefix_checks', 'previous_baseline_checks', 'uninterrupted_controls')}, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--exe', type=Path, default=Path('build/synapticgenesis.exe'))
    p.add_argument('--probe', type=Path, default=Path('build/adaptation-probe/synaptic-adaptation-probe.exe'))
    p.add_argument('--continuation-control', action='store_true')
    a = p.parse_args()
    check(a.root.resolve(), a.exe.resolve(), a.probe.resolve(), a.continuation_control)
