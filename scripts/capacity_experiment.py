"""Measure larger random founders through the shared native live loop."""
import argparse
import json
import math
from pathlib import Path
import statistics
import subprocess

from corpus.selection import require_training_spec
from native_experiment import NativeCommands, read, sha, verified_book_manifest

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SHAPES = [(256, 512, 4), (384, 1024, 6), (512, 2048, 8), (1024, 4096, 8)]


def write(path, value):
    path.write_bytes((json.dumps(value, indent=2) + '\n').encode())


def run(out, executable=None, shapes=None):
    executable = Path(executable or 'build/capacity-probe/synaptic-capacity-probe.exe').resolve()
    shapes = [tuple(shape) for shape in (DEFAULT_SHAPES if shapes is None else shapes)]
    if not shapes or len(shapes) != len(set(shapes)) or any(
            len(shape) != 3 or not (8 <= shape[0] <= 2048 and 8 <= shape[1] <= 8192 and 1 <= shape[2] <= 32)
            for shape in shapes):
        raise ValueError('Capacity shapes must be distinct supported channels/hidden/layers triples')
    specification = Path('data/sources.json')
    prepared = Path('data/prepared/selected-foundations-v1')
    require_training_spec(specification)
    verified_book_manifest(prepared, specification)
    source = prepared / 'train.dat'
    inputs = {str(p): sha(p) for p in (source, specification, prepared / 'manifest.json', executable,
              ROOT / 'experiments/capacity_probe.cu', Path(__file__), ROOT / 'tests/capacity_probe.py',
              ROOT / 'scripts/native_experiment.py', ROOT / 'scripts/corpus/selection.py',
              ROOT / 'data/training-selection.json', ROOT / 'CMakeLists.txt', ROOT / 'build.ps1',
              *sorted((ROOT / 'src').glob('*.cu')), *sorted((ROOT / 'src').glob('*.cuh')))}
    out.mkdir(parents=True, exist_ok=False)
    protocol = dict(status='declared_before_execution', shapes=shapes, cell='associative', seed=1337,
        source=str(source), authenticated_inputs=inputs, warmup=128, updates_per_round=512, rounds=3,
        chunk=128, replay_every=4, replay_capacity=1024, speak_every=128, speak_bytes=64, decode_bytes=1024,
        live_timing='Production source update, uniform reservoir replay, scheduled graph speech, synchronization '
                    'and small speech writes. Excludes initialization, first 128 updates and final saving.',
        decode_timing='Same live speaker after learning. Warm capture, fixed weights/state/RNG; includes prompt '
                      'and CPU sampling. Three timed 1024-byte runs after one untimed run.',
        memory='Explicit live arrays subtract shared weights, moments and recurrence; free VRAM sampled at round '
               'boundaries after CUDA context initialization. Neither is whole-process peak memory.',
        limitation='A short capacity/speed test on existing admitted source text; no language-quality scaling result. '
                   'Other desktop GPU contexts active. Views are lazily cached; more tail lengths can cost more VRAM.',
        random_initialization=True, native_cpp_cuda=True, generated_text_targets=False, checkpoints_saved=False,
        hardware=subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version,memory.total',
                                         '--format=csv,noheader'], text=True).strip())
    write(out / 'protocol.json', protocol)
    commands = out / 'commands'
    commands.mkdir()
    native = NativeCommands(executable, commands)
    rows = []
    for channels, hidden, layers in shapes:
        assert all(sha(p) == digest for p, digest in inputs.items()), 'Capacity input changed before learning'
        directory = out / f'c{channels}-h{hidden}-l{layers}'
        native('run', '--out', directory, '--data', source, '--cell', 'associative', '--channels', channels,
               '--hidden', hidden, '--layers', layers, '--chunk', 128, '--seed', 1337, '--lr', .0003,
               '--replay', 'reservoir', '--replay-every', 4, '--replay-capacity', 1024, '--speak-every', 128,
               '--tokens', 64, '--graph', '--fast', '--warmup', 128, '--updates', 512, '--rounds', 3,
               '--decode-bytes', 1024)
        result = read(directory / 'result.json')
        learning = [json.loads(line) for line in (directory / 'rounds.jsonl').read_text().splitlines()]
        decoding = [json.loads(line) for line in (directory / 'decode.jsonl').read_text().splitlines()]
        assert len(learning) == len(decoding) == 3 and result['complete'] and result['decode_repeated_bytes_exact']
        assert (result['channels'], result['neurons_per_layer'], result['layers']) == (channels, hidden, layers)
        assert result['spiking_neurons'] == hidden * layers
        assert result['warmup_updates'] == 128 and result['measured_updates_per_round'] == 512
        assert [r['round'] for r in learning] == [r['round'] for r in decoding] == list(range(3))
        assert all(math.isfinite(r['seconds']) and r['seconds'] > 0 for r in learning + decoding)
        assert all(math.isfinite(r['loss']) and r['source_pairs'] > 0 for r in learning)
        assert all(r['generated_bytes'] == 1024 for r in decoding)
        row = dict(**result, learning_rounds=learning, decode_rounds=decoding,
            median_source_bytes_per_second=statistics.median(r['source_pairs'] / r['seconds'] for r in learning),
            median_generated_bytes_per_second=statistics.median(r['generated_bytes'] / r['seconds'] for r in decoding))
        rows.append(row)
        write(out / 'partial.json', rows)
        print('parameters', row['parameters'], 'neurons', row['spiking_neurons'], 'learn bytes/s',
              round(row['median_source_bytes_per_second']), 'decode bytes/s',
              round(row['median_generated_bytes_per_second']), 'arrays MiB',
              round(row['explicit_arrays_bytes'] / 1048576, 1), flush=True)
    assert all(sha(p) == digest for p, digest in inputs.items())
    # Replay/source traversal depends on data and policy, not the model dimensions.
    for row in rows[1:]:
        for ref, actual in zip(rows[0]['learning_rounds'], row['learning_rounds']):
            assert all(ref[k] == actual[k] for k in ('source_pairs', 'replay_pairs', 'generated_bytes'))
    write(out / 'comparison.json', dict(complete=True, protocol=protocol, rows=rows, inputs_unchanged=True,
                                      all_shapes_same_exposure=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--probe-exe', type=Path, default=Path('build/capacity-probe/synaptic-capacity-probe.exe'))
    parser.add_argument('--shape', type=int, nargs=3, action='append', metavar=('CHANNELS', 'HIDDEN', 'LAYERS'),
                        help='Repeat for each matched shape; omitted keeps the original four-size panel')
    args = parser.parse_args()
    run(args.out, args.probe_exe, args.shape)
