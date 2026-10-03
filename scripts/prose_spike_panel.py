"""Matched read-only native traces, sequenced after the fixed size comparison."""
import argparse
import hashlib
from pathlib import Path
import struct
import subprocess

import numpy as np

from corpus.selection import require_training_spec, selection
from native_experiment import NativeCommands, read, verified_book_manifest
from process_gate import ProcessGate
from prose_founder import file_hash, write
from spike_regime import regime


def prepare(comparison, diagnostic, out):
    source = Path('data/sources-prose-scale-v1.json')
    require_training_spec(source)
    prepared = Path('data/prepared/prose-scale-v1-pinned')
    manifest = verified_book_manifest(prepared, source)
    comparison_plan = read(comparison / 'protocol.json')
    assert comparison_plan['profiles_in_order'] == ['105m', '2m', '27m']
    assert comparison_plan['stage_order'] == [1, 2, 3, 4]
    out.mkdir(parents=True, exist_ok=False)
    (out / 'inputs').mkdir()
    windows = []
    for stage in manifest['stages']:
        docs = (prepared / stage['file']).read_bytes().split(b'\x1e')
        books = [r for r in manifest['sources'] if r['split'] == 'train' and r['stage'] == stage['id']]
        assert [(prepared / f"{r['id']}.txt").read_bytes() for r in books] == docs
        for slot, numerator in enumerate((1, 3)):
            index = len(docs) * numerator // 4
            document = docs[index]
            if len(document) < 1025:
                raise ValueError('Selected trace document is too short')
            offset = (len(document) - 1025) // 2
            payload = document[offset:offset + 1025]
            assert b'\x1e' not in payload
            name = f"stage-{stage['id']}-{slot}"
            filename = out / 'inputs' / f'{name}.dat'
            filename.write_bytes(payload)
            windows.append(dict(name=name, stage=stage['id'], book=books[index]['id'],
                file=filename.as_posix(), document_sha256=hashlib.sha256(document).hexdigest(),
                byte_offset=offset, input_bytes=1025, target_bytes=1024, sha256=file_hash(filename)))
    legacy = read('reports/native-activity-panel.json')
    row = next(r for r in legacy['records'] if r['label'] == '1337-associative-control')
    window = next(r for r in row['windows'] if r['window'] == 'reading-0')
    legacy_result = Path('runs/native-activity-panel/1337-associative-control/reading-0/result.json')
    assert file_hash(legacy_result) == window['result_sha256']
    old = read(legacy_result)
    assert old['protocol']['checkpoint_sha256'] in {r['sha256'] for r in selection()['continuation_bases']}
    compatibility = dict(result=legacy_result.as_posix(), result_sha256=file_hash(legacy_result),
                          checkpoint=old['protocol']['checkpoint'], source=old['protocol']['source'])
    inputs = [source, prepared / 'manifest.json', comparison / 'protocol.json', diagnostic,
              Path(__file__), Path('scripts/spike_regime.py'), Path('scripts/process_gate.py'),
              Path('scripts/native_experiment.py'), Path('scripts/prose_founder.py'),
              Path('scripts/corpus/selection.py'),
              Path('tests/learned_trace_dump.cu'), Path('tests/spike_regime.py'),
              Path('reports/native-activity-panel.json'), legacy_result,
              *(Path('src').glob('*.cu')), *(Path('src').glob('*.cuh'))]
    protocol = dict(status='declared_before_matched_traces', comparison=comparison.as_posix(),
        diagnostic=diagnostic.as_posix(), source_commit=subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], text=True).strip(),
        authenticated_inputs={p.as_posix(): file_hash(p) for p in inputs},
        profiles=['105m', '2m', '27m'], stages=[0, 1, 4], windows=windows,
        compatibility=compatibility, native_commands_planned=73,
        policy='One unchanged production forward per window, strict FP32, fresh recurrence. '
               'Initialization, first stage and completed curriculum for each model size. '
               'All eight 1024-target-byte windows are identical across checkpoints. '
               'No native model work starts until the size-comparison process has exited successfully '
               'and its completed comparison authenticates the learned checkpoints.',
        pooling='All eight windows contribute equally. Temporal transitions across reset-window boundaries '
                'are excluded. Marginal spike entropy is not joint information or task importance.',
        limits='Selected training-source windows, not a new held-out quality benchmark or live-stream '
               'trajectory. Reset windows are shorter than full documents. Zero local spike surrogate '
               'does not imply zero full gradient. Dense CUDA projections do not become sparse merely '
               'because spike count falls. No controller or new learning rule is tested here.')
    write(out / 'protocol.json', protocol)
    print('Declared nine checkpoints and eight fixed 1024-byte trace windows per checkpoint.', flush=True)


def load_array(trace, name, shape):
    value = np.fromfile(trace / name, dtype='<f4')
    if value.size != int(np.prod(shape)) or not np.isfinite(value).all():
        raise ValueError('Malformed native array: ' + str(trace / name))
    return value.reshape(shape)


def authenticate(plan):
    require_training_spec('data/sources-prose-scale-v1.json')
    for path, digest in plan['authenticated_inputs'].items():
        if file_hash(path) != digest:
            raise ValueError('Declared diagnostic input changed: ' + path)
    for window in plan['windows']:
        if file_hash(window['file']) != window['sha256']:
            raise ValueError('Declared trace window changed: ' + window['file'])


def inspect_window(trace, window, hidden, layers):
    count = window['target_bytes']
    logits = load_array(trace, 'logits.f32', (count, 256)).astype(np.float64)
    losses = load_array(trace, 'losses.f32', (count,)).astype(np.float64)
    target = np.frombuffer(Path(window['file']).read_bytes()[1:], dtype=np.uint8)
    maxima = logits.max(axis=1)
    expected = maxima + np.log(np.exp(logits - maxima[:, None]).sum(axis=1))
    expected -= logits[np.arange(count), target]
    error = float(np.max(np.abs(expected - losses)))
    if error >= 3e-5:
        raise ValueError(f'Native loss/logit/target consistency error: {error}')
    rows = []
    for layer in range(layers):
        arrays = [load_array(trace, f'layer-{layer}-{name}.f32', (count, hidden))
                  for name in ('spikes', 'emission', 'u', 'z')]
        rows.append(dict(layer_zero_based=layer, **regime(*arrays)))
    return dict(window=window['name'], loss_nats_per_byte=float(losses.mean()),
                loss_consistency_max_error=error, layers=rows,
                trace_sha256={p.name: file_hash(p) for p in sorted(trace.iterdir())})


def execute(out, wait_pid, wait_executable):
    plan = read(out / 'protocol.json')
    assert plan['status'] == 'declared_before_matched_traces'
    authenticate(plan)
    gate = ProcessGate(wait_pid, wait_executable) if wait_pid else None
    try:
        if gate:
            write(out / 'waiting-process.json', gate.identity)
            print('Waiting for the complete size comparison:', gate.identity, flush=True)
            code = gate.wait()
            write(out / 'predecessor-exit.json', dict(**gate.identity, exit_code=code))
        authenticate(plan)
        parent = Path(plan['comparison'])
        result = read(parent / 'comparison.json')
        assert result['complete'] and result['matched_source_replay_and_speech_counters']
        assert result['protocol'] == read(parent / 'protocol.json')
        assert not result['reserved_tests_scored']
        native = NativeCommands(plan['diagnostic'], out)
        legacy = read(plan['compatibility']['result'])
        cp, source = Path(plan['compatibility']['checkpoint']), Path(plan['compatibility']['source'])
        assert file_hash(cp) == legacy['protocol']['checkpoint_sha256']
        assert file_hash(source) == legacy['protocol']['source_sha256']
        native(cp, source, out / 'compatibility')
        actual = {p.name: file_hash(p) for p in sorted((out / 'compatibility').iterdir())}
        if actual != legacy['trace_sha256']:
            write(out / 'compatibility-failure.json', dict(expected=legacy['trace_sha256'], actual=actual))
            raise ValueError('Rebuilt diagnostic differs from the preserved learned trace')
        records = []
        for profile in plan['profiles']:
            directory = Path(result['protocol']['original_founder']) if profile == '105m' else parent / f'founder-{profile}'
            founder = read(directory / 'protocol.json')
            assert founder['profile'] == profile and founder['random_initialization'] and not founder['imported_weights']
            for stage in plan['stages']:
                checkpoint = directory / ('initial.ckpt' if stage == 0 else f'stage-{stage}.ckpt')
                identity = file_hash(checkpoint)
                if stage:
                    quality = next(r for r in result['rows'] if r['profile'] == profile and r['stage'] == stage)
                    assert quality['checkpoint_sha256'] == identity
                with checkpoint.open('rb') as stream:
                    meta = struct.unpack('<32Q', stream.read(256))
                assert meta[1] == 6 and meta[5] == 1
                assert meta[2:5] == (founder['channels'], founder['neurons_per_layer'], founder['layers'])
                assert meta[24] == (0 if stage == 0 else founder['source_schedule']['stages'][stage - 1]['end_update'])
                if stage == 0:
                    assert meta[7] == meta[22] == meta[30] == 0
                hidden, layers = meta[3:5]
                model_out = out / f'{profile}-stage-{stage}'
                model_out.mkdir()
                windows = []
                for window in plan['windows']:
                    trace = model_out / window['name']
                    native(checkpoint, window['file'], trace)
                    measured = inspect_window(trace, window, hidden, layers)
                    write(model_out / f"{window['name']}.json", measured)
                    windows.append(measured)
                pooled = []
                for layer in range(layers):
                    arrays = [np.concatenate([load_array(model_out / w['name'], f'layer-{layer}-{name}.f32',
                                  (w['target_bytes'], hidden)) for w in plan['windows']])
                              for name in ('spikes', 'emission', 'u', 'z')]
                    pooled.append(dict(layer_zero_based=layer,
                        **regime(*arrays, segments=[w['target_bytes'] for w in plan['windows']])))
                assert file_hash(checkpoint) == identity
                row = dict(profile=profile, stage=stage, checkpoint=checkpoint.as_posix(),
                           checkpoint_sha256=identity, parameters=meta[14], source_observations=meta[24],
                           windows=windows, pooled_layers=pooled,
                           mean_window_loss_nats_per_byte=float(np.mean([w['loss_nats_per_byte'] for w in windows])))
                records.append(row)
                write(out / 'partial.json', records)
                print(profile, 'stage', stage, 'complete; mean diagnostic-window loss',
                      row['mean_window_loss_nats_per_byte'], flush=True)
        authenticate(plan)
        assert len(native.commands) == plan['native_commands_planned']
        write(out / 'result.json', dict(complete=True, protocol=plan, records=records,
              comparison_sha256=file_hash(parent / 'comparison.json'), native_commands=len(native.commands),
              compatibility_trace_byte_identical=True, all_checkpoints_unchanged=True,
              learning_updates=0, reserved_tests_scored=False))
        print('Matched native spike-regime panel complete.', flush=True)
    finally:
        if gate:
            gate.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subs = parser.add_subparsers(dest='command', required=True)
    declaration = subs.add_parser('prepare')
    declaration.add_argument('--comparison', type=Path, default=Path('runs/prose-size-panel'))
    declaration.add_argument('--diagnostic', type=Path, default=Path('build/trace-diagnostic/synaptic-trace-diagnostic.exe'))
    declaration.add_argument('--out', type=Path, required=True)
    execution = subs.add_parser('run')
    execution.add_argument('--out', type=Path, required=True)
    execution.add_argument('--wait-pid', type=int, default=0)
    execution.add_argument('--wait-executable', type=Path)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.comparison, args.diagnostic, args.out)
    else:
        if args.wait_pid and args.wait_executable is None:
            parser.error('--wait-executable is required with --wait-pid')
        execute(args.out, args.wait_pid, args.wait_executable)
