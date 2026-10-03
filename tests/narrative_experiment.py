"""Audit narrative admission, exposure, replay history and exact continuation."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
from native_experiment import NativeCommands, read, sha, write
from narrative_experiment import state_record


def check(root, exe, continuation_control=False):
    data = read(root / 'comparison.json')
    p = data['protocol']
    parent = Path(p['parent_directory'])
    base, endpoints = p['baseline_online_updates'], p['online_endpoints']
    models = len(p['seeds']) * len(p['arms'])
    assert sha(exe) == p['executable_sha256']
    assert len(data['runs']) == models * (1 + len(endpoints))
    assert len({(r['seed'], r['arm'], r['online_updates']) for r in data['runs']}) == len(data['runs'])
    commands = read(root / 'commands.json')
    assert data['native_commands'] == len(commands) == models * 20
    assert Counter(c[1] for c in commands) == {
        'live': models * 2, 'evaluate': models * 9, 'sample': models * 6,
        'language-probes': models * 2, 'decode-bench': models}
    for command in commands:
        if '--data' in command:
            assert Path(command[command.index('--data') + 1]).name in ('13853.txt', '12228.txt', '11757.txt')
        if '--probes' in command:
            assert Path(command[command.index('--probes') + 1]).name == 'development.sgprobe'
    admission = p['source_admission']
    assert admission['previous_stages'] == 3 and admission['added_documents'] == 7
    assert admission['editions'][-1]['answer_scale'] == 1
    assert [r['answer_scale'] for r in admission['editions'][:-1]] == [1, 64, 64]
    for record in admission['editions']:
        assert sha(root / 'edition' / record['file']) == record['sha256']
    docs = (Path(p['prepared_directory']) / 'train.dat').read_bytes().split(b'\x1e')
    windows = [min(128, len(doc) - 1 - offset) for doc in docs for offset in range(0, len(doc) - 1, 128)]
    assert len(docs) == 7
    expected_pairs = {end: (end - base) // len(windows) * sum(windows) +
                      sum(windows[:(end - base) % len(windows)]) for end in endpoints}
    for seed in p['seeds']:
        for arm in p['arms']:
            directory = root / f'{seed}-{arm}'
            baseline = state_record(directory / f'checkpoint-{base}.ckpt')
            original = parent / f'{seed}-{arm}/checkpoint-{base}.ckpt'
            assert baseline['checkpoint_sha256'] == sha(original) == p['parent_checkpoints'][f'{seed}-{arm}']
            events = [json.loads(line) for line in (directory / 'metrics.jsonl').read_text().splitlines()]
            events = [event for event in events if event.get('event') == 'curriculum_extension']
            assert len(events) == 1
            event = events[0]
            assert event['online_update'] == base and event['previous_stages'] == 3 and event['stages'] == 4
            assert event['global_update'] == baseline['global_updates']
            assert event['replay_updates'] == baseline['replay_updates']
            assert event['replay_windows_preserved'] == p['replay_slots']
            assert event['generated_bytes'] == baseline['generated_bytes']
            for end in [base] + endpoints:
                row = next(r for r in data['runs'] if r['seed'] == seed and r['arm'] == arm and r['online_updates'] == end)
                state = state_record(directory / f'checkpoint-{end}.ckpt')
                assert all(row[key] == value for key, value in state.items())
                assert len(row['samples']) == 2
                for index, sample in enumerate(row['samples']):
                    assert sha(directory / f'sample-{end}-{index}.txt') == sample['file_sha256']
                if end == base:
                    assert (directory / f'development-{end}.json').read_bytes() == (
                        parent / f'{seed}-{arm}/development-{end}.json').read_bytes()
                    continue
                delta, extra_replays = end - base, end // 4 - base // 4
                assert state['global_updates'] == baseline['global_updates'] + delta + extra_replays
                assert state['replay_updates'] == baseline['replay_updates'] + extra_replays
                assert state['observed_pairs'] == baseline['observed_pairs'] + expected_pairs[end]
                assert state['generated_bytes'] == baseline['generated_bytes'] + 96 * (
                    end // p['speak_every'] - base // p['speak_every'])
                assert state['curriculum_stage'] == 4
                assert [r['stored_windows'] for r in state['replay_groups']] == p['replay_quotas']
                for earlier, later in zip(baseline['replay_groups'], state['replay_groups'][:3]):
                    assert earlier['document_end'] == later['document_end']
                    assert earlier['seen_windows'] == later['seen_windows']
                    assert later['replay_updates'] >= earlier['replay_updates']
                    assert later['replay_pairs'] >= earlier['replay_pairs']
                assert state['replay_groups'][-1]['seen_windows'] == delta
                assert row['session']['online_document_count'] == 7 and row['session']['shared_weights']
                assert row['session']['persistent_membranes'] and not row['session']['trains_on_generated_text']
                assert abs(row['session']['final_validation_loss'] - row['books']['reader']['loss_nats_per_byte']) < 1e-6
                if end == endpoints[-1]:
                    assert row['decode']['generated_bytes_identical'] and row['decode']['state_logits_max_error'] == 0
        for end in [base] + endpoints:
            rows = [r for r in data['runs'] if r['seed'] == seed and r['online_updates'] == end]
            for key in ('observed_pairs', 'global_updates', 'replay_updates', 'replay_pairs', 'generated_bytes'):
                assert len({r[key] for r in rows}) == 1, (seed, end, key)
            assert all(r['replay_groups'] == rows[0]['replay_groups'] for r in rows)
    result = dict(passed=True, smoke_only=p['status'] == 'smoke_only', native_commands=len(commands),
                  models=models, checkpoint_rows=len(data['runs']), parent_files_unchanged=True,
                  selected_new_documents=7, independent_expected_narrative_pairs=expected_pairs,
                  exposure_matches=True, inherited_replay_counters_preserved=True,
                  declared_replay_quotas_verified=True, validation_reader_repeat_matches=True,
                  graph_generation_exact=True, samples_authenticated=True, reserved_tests_not_scored=True)
    if continuation_control:
        assert result['smoke_only'], 'Run the additional uninterrupted control in the shortened rehearsal'
        controls = root / 'continuation-controls'
        controls.mkdir(exist_ok=False)
        native = NativeCommands(exe, controls)
        records = []
        for arm in p['arms']:
            name = f'{p["seeds"][0]}-{arm}'
            directory = controls / name
            native('live', '--resume', parent / name / f'checkpoint-{base}.ckpt',
                   '--curriculum', parent / 'curriculum.sg', '--extend-curriculum', root / 'edition/curriculum.sg',
                   '--out', directory, '--updates', endpoints[-1], '--prompt', 'The bird ',
                   '--validation', Path(p['prepared_directory']) / '13853.txt', '--eval-batches', p['evaluation_batches'],
                   '--log-every', p['log_every'], '--save-every', p['save_every'])
            expected = root / name / f'checkpoint-{endpoints[-1]}.ckpt'
            assert (directory / 'latest.ckpt').read_bytes() == expected.read_bytes(), arm
            records.append(dict(arm=arm, complete_checkpoint_identical=True, checkpoint_sha256=sha(expected)))
        result['uninterrupted_controls'] = records
        result['control_native_commands'] = len(native.commands)
    write(root / 'execution-check.json', result)
    print(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--exe', type=Path, default=Path('build/synapticgenesis.exe'))
    parser.add_argument('--continuation-control', action='store_true')
    args = parser.parse_args()
    check(args.root, args.exe, args.continuation_control)
