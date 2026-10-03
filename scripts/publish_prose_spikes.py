"""Authenticate completed native trace artifacts and publish the exact result."""
import argparse
import math
from pathlib import Path
import struct

from native_experiment import read
from prose_founder import file_hash, write
from prose_spike_panel import authenticate


def checked_trace(directory, expected, steps, channels, hidden, layers):
    shapes = {'logits.f32': steps * 256, 'losses.f32': steps}
    for layer in range(layers):
        for name in ('norm', 'z', 'gate', 'u', 'spikes', 'emission'):
            shapes[f'layer-{layer}-{name}.f32'] = steps * (channels if name == 'norm' else hidden)
    assert set(expected) == set(shapes) == {p.name for p in directory.iterdir()}
    byte_count = 0
    for name, count in shapes.items():
        path = directory / name
        assert path.stat().st_size == count * 4, path
        assert file_hash(path) == expected[name], path
        byte_count += count * 4
    return len(shapes), byte_count


def header(path):
    with Path(path).open('rb') as stream:
        return struct.unpack('<32Q', stream.read(256))


def publish(root, report, audit):
    if report.exists() or audit.exists():
        raise ValueError('Publication outputs already exist')
    result = read(root / 'result.json')
    plan = result['protocol']
    assert plan == read(root / 'protocol.json')
    authenticate(plan)
    assert result['complete'] and result['all_checkpoints_unchanged']
    assert result['compatibility_trace_byte_identical'] and result['learning_updates'] == 0
    assert not result['reserved_tests_scored'] and result['native_commands'] == 73
    comparison_path = Path(plan['comparison']) / 'comparison.json'
    assert file_hash(comparison_path) == result['comparison_sha256']
    comparison = read(comparison_path)
    commands, wanted, cases = read(root / 'commands.json'), [], []
    trace_files = trace_bytes = 0
    legacy = read(plan['compatibility']['result'])
    cp, source = Path(plan['compatibility']['checkpoint']), Path(plan['compatibility']['source'])
    assert file_hash(cp) == legacy['protocol']['checkpoint_sha256']
    assert file_hash(source) == legacy['protocol']['source_sha256']
    meta = header(cp)
    legacy_steps = source.stat().st_size - 1
    count, size = checked_trace(root / 'compatibility', legacy['trace_sha256'], legacy_steps, *meta[2:5])
    assert count == 26 and legacy_steps == 256
    trace_files += count
    trace_bytes += size
    wanted.append(list(map(str, [cp, source, root / 'compatibility'])))
    log_lines = [f'Read-only neuron traces: {legacy_steps} bytes, {meta[4]} layers\n']
    order = [(p, s) for p in plan['profiles'] for s in plan['stages']]
    assert [(r['profile'], r['stage']) for r in result['records']] == order
    assert len(order) == 9 and len(plan['windows']) == 8
    simple_means = ('spike_negative_fraction', 'spike_zero_fraction', 'spike_positive_fraction',
                    'spike_event_fraction', 'within_window_ternary_change_fraction',
                    'within_window_direct_sign_flip_fraction', 'zero_local_spike_surrogate_fraction',
                    'local_spike_surrogate_mean')
    for row in result['records']:
        checkpoint = Path(row['checkpoint'])
        assert file_hash(checkpoint) == row['checkpoint_sha256']
        meta = header(checkpoint)
        channels, hidden, layers = meta[2:5]
        assert meta[1] == 6 and meta[5] == 1 and meta[14] == row['parameters']
        assert meta[24] == row['source_observations']
        if row['stage']:
            match = next(r for r in comparison['rows'] if (r['profile'], r['stage']) == (row['profile'], row['stage']))
            assert match['checkpoint_sha256'] == row['checkpoint_sha256']
        else:
            assert meta[7] == meta[22] == meta[24] == meta[30] == 0
        directory = root / f'{row["profile"]}-stage-{row["stage"]}'
        assert [w['window'] for w in row['windows']] == [w['name'] for w in plan['windows']]
        assert len(row['pooled_layers']) == layers
        for window, measured in zip(plan['windows'], row['windows']):
            assert window['target_bytes'] == 1024
            assert measured == read(directory / f'{window["name"]}.json')
            assert measured['loss_consistency_max_error'] < 3e-5
            count, size = checked_trace(directory / window['name'], measured['trace_sha256'],
                                        1024, channels, hidden, layers)
            trace_files += count
            trace_bytes += size
            wanted.append(list(map(str, [checkpoint, window['file'], directory / window['name']])))
            log_lines.append(f'Read-only neuron traces: 1024 bytes, {layers} layers\n')
        assert math.isclose(row['mean_window_loss_nats_per_byte'],
                            sum(w['loss_nats_per_byte'] for w in row['windows']) / 8, abs_tol=1e-14)
        for index, pooled in enumerate(row['pooled_layers']):
            assert pooled['layer_zero_based'] == index and pooled['hidden'] == hidden
            assert pooled['observed_steps'] == 8192 and pooled['independent_windows'] == 8
            assert pooled['within_window_adjacent_pairs'] == 8 * 1023 * hidden
            for key in simple_means:
                average = sum(w['layers'][index][key] for w in row['windows']) / 8
                assert math.isclose(pooled[key], average, rel_tol=0, abs_tol=1e-12), key
        cases.append(dict(profile=row['profile'], stage=row['stage'], checkpoint_sha256=row['checkpoint_sha256'],
                          layers=layers, windows=8, trace_files=8 * (6 * layers + 2),
                          all_raw_trace_hashes_match=True, simple_pooled_means_match_windows=True))
        print(row['profile'], row['stage'], 'trace artifacts verified', flush=True)
    assert len(commands) == len(wanted) == len(log_lines) == 73
    assert [c[1:] for c in commands] == wanted
    assert all(Path(c[0]).resolve() == Path(plan['diagnostic']).resolve() for c in commands)
    for index, line in enumerate(log_lines, 1):
        assert (root / f'command-{index:03d}.log').read_text(encoding='utf-8') == line
    assert read(root / 'predecessor-exit.json')['exit_code'] == 0
    authenticate(plan)
    report.write_bytes((root / 'result.json').read_bytes())
    write(audit, dict(passed=True, native_commands=73, windows=72, cases=cases,
        raw_trace_files_verified=trace_files, raw_trace_bytes_verified=trace_bytes,
        legacy_trace_files_byte_identical=26, all_raw_outputs_and_command_logs_match=True,
        comparison_sha256=result['comparison_sha256'], published_report_sha256=file_hash(report),
        exact_result_copy=True, audit_script_sha256=file_hash(Path(__file__)),
        new_native_commands=0, model_updates=0, reserved_tests_scored=False,
        limit='Artifact and declared-execution verification, with pooled count consistency checks. '
              'Not an independent recomputation of all neuron statistics, new learning result, '
              'new seed or held-out language benchmark.'))
    print('Published 73 verified native commands and', trace_files, 'raw trace files.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('runs/prose-spike-panel'))
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    args = parser.parse_args()
    publish(args.root, args.report, args.audit)
