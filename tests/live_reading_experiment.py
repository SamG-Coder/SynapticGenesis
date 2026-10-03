"""Audit exact live exposure/replay, assessment isolation and complete restart."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from adaptation_sources import fnv, verified_edition
from audit_binding import audit
from experiment_checkpoint import checkpoint, state_record
from extend_curriculum import read_schedule
from native_experiment import NativeCommands, read, sha, write
from stage_replay_reference import ReplayReference, schedule_identity


def argument(command, name):
    return command[command.index('--' + name) + 1]


def assessment(row, directory, model, protocol, documents):
    endpoint = row['online_updates']
    expected = state_record(model)
    assert all(row[k] == v for k, v in expected.items())
    reading_dir = directory / f'reading-{endpoint}'
    native = read(reading_dir / 'result.json')
    score = read(reading_dir / 'evaluation-0.json')
    assert score == row['reading'] and sha(reading_dir / 'evaluation-0.json') == row['reading_report_sha256']
    assert native['completed'] and native['inputs_unchanged']
    assert native['session_presented_pairs'] == native['end_updates'] == native['start_updates'] == 0
    assert score['weights_optimizer_unchanged'] and score['global_updates'] == expected['global_updates']
    original, view = model.read_bytes(), (reading_dir / 'checkpoint-0.ckpt').read_bytes()
    meta = checkpoint(model)[0]
    assert original[288:288 + 12 * meta[14]] == view[288:288 + 12 * meta[14]], 'Assessment changed weights or moments'
    for split, docs in documents.items():
        measured = score[split]
        assert len(docs) == len(measured['documents'])
        assert measured['pairs'] == sum(len(d) - 1 for d in docs)
        weighted = 0
        for index, (entry, doc) in enumerate(zip(measured['documents'], docs)):
            assert entry['index'] == index and entry['pairs'] == len(doc) - 1
            assert math.isfinite(entry['loss']) and entry['loss'] >= 0
            weighted += entry['pairs'] * entry['loss']
        assert abs(weighted / measured['pairs'] - measured['loss']) < 1e-12
    for name, value in row['books'].items():
        assert read(directory / f'{name}-{endpoint}.json') == value
        assert value['evaluated_bytes'] == 16 * 128 * protocol['evaluation_batches']
    binding = read(directory / f'development-{endpoint}.json')
    assert {k: v for k, v in binding.items() if k != 'results'} == row['development']
    assert binding['parameters_and_optimizer_unchanged'] and binding['strict_fp32']
    assert audit(directory / f'development-{endpoint}.json') == row['development_audit']


def speech(path):
    raw = path.read_bytes()
    return re.sub(rb'\n\[session starts at online update [0-9]+\]\n', b'', raw)


def check(root, exe, probe, continuation_control=False):
    data = read(root / 'comparison.json')
    p = data['protocol']
    assert data['complete'] and data['originals_unchanged']
    assert not p['reserved_tests_scored'] and not p['generated_text_targets'] and not p['teacher_feedback']
    assert all(sha(path) == identity for path, identity in p['source_sha256'].items())
    assert sha(exe) == p['executable_sha256'] and sha(probe) == p['diagnostic_executable_sha256']
    assert sha('runs/teacher-retention-panel/comparison.json') == p['parent_comparison_sha256']
    assert sha(p['parent_schedule']) == p['parent_schedule_sha256']
    source = root / 'reading-edition'
    edition = verified_edition(Path('data/prepared/early-readers-v1'), Path('data/sources-early-readers-v1.json'))
    assert edition == p['source_edition']
    for name in ('train.dat', 'validation.dat', 'manifest.json', 'source-spec.json', 'ATTRIBUTION.md'):
        assert (source / name).read_bytes() == (Path('data/prepared/early-readers-v1') / name).read_bytes()
    assert sha(source / 'evaluation.sg') == p['evaluation_schedule_sha256']
    docs = {split: (source / f'{split}.dat').read_bytes().split(b'\x1e') for split in ('train', 'validation')}
    schedule_raw, stages = read_schedule(root / 'curriculum/curriculum.sg')
    admission = p['source_admission']
    assert len(stages) == 5 and admission['previous_stages'] == 4 and admission['added_documents'] == 10
    assert sha(root / 'curriculum/curriculum.sg') == admission['schedule_sha256']
    assert [float(s['answer']) for s in stages] == [1, 64, 64, 1, 1]
    for stage, declared in zip(stages, admission['editions']):
        assert sha(stage['source']) == declared['sha256']
    assert stages[-1]['content'] == stages[-2]['content'] + b'\x1e' + (source / 'train.dat').read_bytes()
    assert stages[-1]['scope'] == 'new' and float(stages[-1]['rate']) == .25
    schedule_hash = schedule_identity(schedule_raw, stages)
    corpus_hash = fnv(stages[-1]['content'])
    lengths = [len(d) for d in stages[-1]['content'].split(b'\x1e')]
    for v in p['book_validation'].values():
        assert sha(Path(p['book_directory']) / f"{v['id']}.txt") == v['sha256']
    assert sha(Path(p['binding_directory']) / 'development.sgprobe') == p['probe_sha256']['development.sgprobe']
    commands = read(root / 'native-commands/commands.json')
    diagnostics = read(root / 'diagnostic-commands/commands.json')
    count = sum(len(p['additional_endpoints'][arm]) for arm in p['arms'])
    seeds = len(p['seeds'])
    assert len(commands) == data['native_commands'] == seeds * (5 * count + 10)
    assert len(diagnostics) == data['diagnostic_commands'] == seeds * (count + 1)
    assert Counter(c[1] for c in commands) == dict(live=seeds * count, evaluate=seeds * (count + 1) * 3,
        **{'language-probes': seeds * (count + 1), 'sample': seeds * 4, 'decode-bench': seeds * 2})
    for i, command in enumerate(commands, 1):
        assert Path(command[0]) == exe.resolve()
        assert (root / f'native-commands/command-{i:03d}.log').is_file()
        if command[1] == 'evaluate':
            assert Path(argument(command, 'data')).name in ('13853.txt', '12228.txt', '11757.txt')
            assert int(argument(command, 'batches')) == p['evaluation_batches']
        if command[1] == 'language-probes':
            assert sha(argument(command, 'probes')) == p['probe_sha256']['development.sgprobe']
    for i, command in enumerate(diagnostics, 1):
        assert Path(command[0]) == probe.resolve() and command[1] == 'run'
        assert argument(command, 'mode') == 'carry' and argument(command, 'steps') == argument(command, 'endpoints') == '0'
        assert Path(argument(command, 'train')) == source / 'train.dat'
        assert Path(argument(command, 'validation')) == source / 'validation.dat'
        assert (root / f'diagnostic-commands/command-{i:03d}.log').is_file()
    expected_order = []
    for i, seed in enumerate(p['seeds']):
        for arm in p['arms'] if i % 2 == 0 else p['arms'][::-1]:
            expected_order += [(f'{seed}-{arm}', p['baseline_online_updates'] + end) for end in p['additional_endpoints'][arm]]
    live_commands = [c for c in commands if c[1] == 'live']
    assert [(Path(argument(c, 'out')).name, int(argument(c, 'updates'))) for c in live_commands] == expected_order
    expected_rows = {(seed, arm, end) for seed in p['seeds'] for arm in p['arms'] for end in p['additional_endpoints'][arm]}
    assert len(data['runs']) == len(expected_rows)
    assert {(r['seed'], r['arm'], r['additional_observations']) for r in data['runs']} == expected_rows
    checked = []
    for seed in p['seeds']:
        ancestor = Path(p['ancestors'][str(seed)]['path'])
        assert sha(ancestor) == p['ancestors'][str(seed)]['sha256']
        original = state_record(ancestor)
        base = p['baseline_online_updates']
        baseline = next(r for r in data['baselines'] if r['seed'] == seed)
        assessment(baseline, root / f'{seed}-baseline', ancestor, p, docs)
        for arm in p['arms']:
            directory = root / f'{seed}-{arm}'
            assert sha(directory / f'checkpoint-{base}.ckpt') == sha(ancestor)
            reference = ReplayReference(ancestor, p['replay_every'][arm], schedule_hash, corpus_hash, lengths)
            elapsed, previous = 0., base
            for additional in p['additional_endpoints'][arm]:
                end = base + additional
                row = next(r for r in data['runs'] if (r['seed'], r['arm'], r['additional_observations']) == (seed, arm, additional))
                model = directory / f'checkpoint-{end}.ckpt'
                reference.run_until(end)
                proof = reference.matches(model)
                assert ancestor.read_bytes()[256:288] == model.read_bytes()[256:288], 'Inherited learning policy changed'
                assessment(row, directory, model, p, docs)
                session = read(directory / f'session-{end}.json')
                assert session == row['session']
                assert session['online_updates'] == end and session['session_online_updates'] == end - previous
                assert session['shared_weights'] and session['persistent_membranes'] and not session['trains_on_generated_text']
                assert session['online_document_count'] == 10 and session['replay_every'] == p['replay_every'][arm]
                assert session['gradient_reductions'] == 'ordered_v1'
                assert abs(session['final_validation_loss'] - row['books']['reader']['loss_nats_per_byte']) < 1e-6
                elapsed += session['elapsed_seconds']
                assert elapsed == row['cumulative_live_seconds']
                command = next(c for c in live_commands if Path(argument(c, 'out')) == directory and int(argument(c, 'updates')) == end)
                assert Path(argument(command, 'resume')) == directory / f'checkpoint-{previous}.ckpt'
                assert int(argument(command, 'replay-every')) == p['replay_every'][arm]
                assert ('--extend-curriculum' in command) == (previous == base)
                previous = end
                if additional == p['common_exposure_endpoints'][-1]:
                    assert row['decode'] == read(directory / 'decode/benchmark.json')
                    assert row['decode']['generated_bytes_identical'] and row['decode']['state_logits_max_error'] == 0
                    for i, sample in enumerate(row['samples']):
                        output = directory / f'sample-{end}-{i}.txt'
                        raw = output.read_bytes()
                        assert sha(output) == sample['file_sha256']
                        assert raw[:len(sample['prompt'])] == sample['prompt'].encode('ascii')
                        assert raw[len(sample['prompt']):].decode('latin1') == sample['generated_latin1']
                checked.append(dict(seed=seed, arm=arm, additional_observations=additional,
                    checkpoint_sha256=sha(model), **proof))
    result = dict(passed=True, comparison_sha256=sha(root / 'comparison.json'),
        branches=seeds * 2, checkpoint_assessments=seeds * (count + 1),
        native_commands=len(commands), diagnostic_commands=len(diagnostics),
        native_sources_and_executables_authenticated=True, ancestors_unchanged=True,
        complete_replay_descriptors_independently_reproduced=True,
        assessment_weights_and_moments_identical_to_live_checkpoint=True,
        reserved_tests_not_scored=True, checked=checked)
    controls = root / 'continuation-controls'
    if continuation_control or controls.exists():
        assert p['status'] == 'smoke_only'
        fresh = not controls.exists()
        if fresh:
            controls.mkdir()
        native = NativeCommands(exe, controls) if fresh else None
        saved_commands = [] if fresh else read(controls / 'commands.json')
        matches = []
        seed = p['seeds'][0]
        for index, arm in enumerate(p['arms']):
            destination = controls / arm
            args = ['live', '--resume', p['ancestors'][str(seed)]['path'], '--curriculum', p['parent_schedule'],
                '--extend-curriculum', root / 'curriculum/curriculum.sg', '--out', destination,
                '--updates', p['online_endpoints'][-1], '--prompt', p['prompt'],
                '--replay-every', p['replay_every'][arm], '--validation', Path(p['book_directory']) / '13853.txt',
                '--eval-batches', p['evaluation_batches'], '--log-every', p['log_every'], '--save-every', p['save_every']]
            if native:
                native(*args)
            else:
                assert saved_commands[index] == [str(exe.resolve()), *map(str, args)]
            expected = root / f'{seed}-{arm}' / f"checkpoint-{p['online_endpoints'][-1]}.ckpt"
            assert (destination / 'latest.ckpt').read_bytes() == expected.read_bytes()
            assert speech(destination / 'transcript.txt') == speech(root / f'{seed}-{arm}/transcript.txt')
            assert state_record(expected)['generated_bytes'] - p['ancestors'][str(seed)]['state']['generated_bytes'] >= 96
            matches.append(dict(arm=arm, full_checkpoint_identical=True, speech_identical=True,
                                crosses_speech_interval=True, checkpoint_sha256=sha(expected)))
        result['uninterrupted_controls'] = matches
    write(root / 'execution-check.json', result)
    print({k: v for k, v in result.items() if k not in ('checked', 'uninterrupted_controls')}, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--exe', type=Path, default=Path('build/synapticgenesis.exe'))
    p.add_argument('--probe', type=Path, default=Path('build/adaptation-probe/synaptic-adaptation-probe.exe'))
    p.add_argument('--continuation-control', action='store_true')
    a = p.parse_args()
    check(a.root.resolve(), a.exe.resolve(), a.probe.resolve(), a.continuation_control)
