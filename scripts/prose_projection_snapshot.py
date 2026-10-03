"""Read-only projection sensitivity and neuron-parameter snapshots on CPU."""
import argparse
from pathlib import Path
import struct
import subprocess

import numpy as np

from native_experiment import read
from prose_founder import file_hash, write


def layout(channels, hidden, layers):
    """Associative v6 layout; verified against the independent learned oracle."""
    c, h = channels, hidden
    if not (8 <= c <= 2048 and 8 <= h <= 8192 and 1 <= layers <= 32):
        raise ValueError('Invalid associative dimensions')
    shapes = [(c,), (h, c), (h,), (c, h), (c,), (h,), (h,), (h,),
              (h, c), (h,), (98, h), (98,), (c, 32), (c,)]
    blocks, offset = [], 256 * c
    for _ in range(layers):
        block = []
        for shape in shapes:
            end = offset + int(np.prod(shape))
            block.append((offset, end, shape))
            offset = end
        blocks.append(block)
    return blocks, offset + c + 256 * c + 256


def statistics(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError('Invalid statistic input')
    result = dict(zip(('min', 'p50', 'p90', 'p99', 'max'),
                      map(float, np.quantile(values, [0, .5, .9, .99, 1]))))
    result['rms'] = float(np.sqrt(np.mean(values * values)))
    return result


def projection_rows(matrix, gain):
    matrix, gain = np.asarray(matrix), np.asarray(gain)
    if (matrix.ndim != 2 or gain.shape != (matrix.shape[1],) or
            not np.isfinite(matrix).all() or not np.isfinite(gain).all()):
        raise ValueError('Invalid projection shape or values')
    # Sensitivity to the normalized residual coordinates, before learned gain.
    # This is not a variance estimate without assumptions about those inputs.
    return np.sqrt(np.sum(np.square(matrix.astype(np.float64) * gain), axis=1))


def block_statistics(block):
    gain, wi, bi, wo, _, leak, _, scale, gate = block[:9]
    return dict(gain=statistics(gain),
        effective_input_row_l2=statistics(projection_rows(wi, gain)),
        input_bias_absolute=statistics(np.abs(bi)),
        effective_retention_gate_row_l2=statistics(projection_rows(gate, gain)),
        output_column_l2=statistics(np.sqrt(np.sum(np.square(wo.astype(np.float64)), axis=0))),
        leak_beta=statistics(np.exp(-np.logaddexp(0, -leak.astype(np.float64)))),
        continuous_trace_gain=statistics(np.logaddexp(0, scale.astype(np.float64))))


def inspect(path, expected_hash=None):
    identity = file_hash(path)
    if expected_hash is not None and identity != expected_hash:
        raise ValueError('Checkpoint differs from its recorded native assessment')
    with path.open('rb') as stream:
        meta = struct.unpack('<32Q', stream.read(256))
        if meta[0] != 0x314d4c53434e5042 or meta[1] != 6 or meta[17] != 5 or meta[5] != 1:
            raise ValueError('Expected single-stream associative v5 live checkpoint')
        blocks, count = layout(*meta[2:5])
        if count != meta[14] or meta[18] != meta[4] * (2 * meta[3] + 1024) or not 17 <= meta[31] <= 16384:
            raise ValueError('Checkpoint layout differs')
        offset = 288 + 12 * count + 4 * meta[18]
        if path.stat().st_size != offset + 8 * meta[31]:
            raise ValueError('Unexpected checkpoint extent')
        stream.seek(offset)
        extra = struct.unpack(f'<{meta[31]}Q', stream.read(8 * meta[31]))
        if extra[1] != 3 or extra[9] != 0:
            raise ValueError('Expected ordinary stage replay without consolidation')
    weights = np.memmap(path, dtype='<f4', mode='r', offset=288, shape=(count,))
    try:
        rows = []
        for layer, fields in enumerate(blocks):
            values = [weights[a:b].reshape(shape) for a, b, shape in fields]
            if not all(np.isfinite(value).all() for value in values):
                raise ValueError('Nonfinite parameter in layer')
            rows.append(dict(layer_zero_based=layer, **block_statistics(values)))
        del values
    finally:
        weights._mmap.close()
    if file_hash(path) != identity:
        raise ValueError('Checkpoint changed during CPU inspection')
    return dict(checkpoint=path.as_posix(), checkpoint_sha256=identity, cell=meta[1],
                parameters=count, channels=meta[2], hidden=meta[3], layer_count=meta[4],
                source_observations=meta[24], seed=meta[10], layers=rows, checkpoint_unchanged=True)


def run(out):
    snapshots, inputs = [], {}
    for profile, folder in [('2m', Path('runs/prose-size-panel/founder-2m')),
                            ('105m', Path('runs/prose-105m-founder'))]:
        protocol_path = folder / 'protocol.json'
        protocol = read(protocol_path)
        if not protocol['random_initialization'] or protocol['imported_weights'] or protocol['seed'] != 1337:
            raise ValueError('Founder provenance differs')
        inputs[protocol_path.as_posix()] = file_hash(protocol_path)
        for stage in (0, 1, 3, 4):
            checkpoint = folder / ('initial.ckpt' if stage == 0 else f'stage-{stage}.ckpt')
            expected, quality = None, None
            if stage:
                assessment_path = Path(f'runs/prose-size-panel/{profile}-stage-{stage}/result.json')
                quality = read(assessment_path)
                inputs[assessment_path.as_posix()] = file_hash(assessment_path)
                expected = quality['checkpoint_sha256']
            row = inspect(checkpoint, expected)
            wanted = protocol['source_schedule']['stages'][stage - 1]['end_update'] if stage else 0
            if row['source_observations'] != wanted or row['seed'] != protocol['seed']:
                raise ValueError('Snapshot exposure differs')
            row.update(profile=profile, stage=stage,
                       validation_mean_nats_per_byte=quality['validation_mean_nats_per_byte'] if quality else None)
            snapshots.append(row)
            median_norms = [v['effective_input_row_l2']['p50'] for v in row['layers']]
            print(profile, stage, 'median effective input row norms by layer:', median_norms, flush=True)
    for profile in ('2m', '105m'):
        original = next(r for r in snapshots if r['profile'] == profile and r['stage'] == 0)
        for row in [r for r in snapshots if r['profile'] == profile]:
            for current, initial in zip(row['layers'], original['layers']):
                current['input_row_median_ratio_to_initial'] = (
                    current['effective_input_row_l2']['p50'] / initial['effective_input_row_l2']['p50'])
    if any(file_hash(path) != identity for path, identity in inputs.items()):
        raise ValueError('Snapshot evidence changed')
    paths = [Path(__file__), Path('scripts/prose_founder.py'), Path('scripts/native_experiment.py'),
             Path('src/spike_lm.cu'), Path('src/model.cuh'), Path('src/trace_neuron.cuh')]
    write(out, dict(kind='Exploratory CPU parameter inspection after observed prose regression',
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        inputs_sha256=inputs, code_sha256={p.as_posix(): file_hash(p) for p in paths}, snapshots=snapshots,
        native_commands=0, learning_updates=0, reserved_tests_scored=False, checkpoint_files_unchanged=True,
        completed_size_comparison=False, concurrent_27m_learning=True,
        definitions='Let y be the residual vector after division by its RMS and before learned gain g. '
                    'For z = W diag(g) y + b, each effective input row norm is the Euclidean norm '
                    'of a row of W diag(g). It bounds sensitivity of that scalar z to unit changes in y. '
                    'The gate uses the same definition; output column norms measure sensitivity of the '
                    'spiking output projection to one emission coordinate.',
        limits='This is not an activation distribution, parameter-update trajectory, local/full gradient, '
               'causal explanation or complete network Jacobian. Row norms alone do not give observed '
               'drive variance or firing rate: input covariance, direction, biases and recurrence matter. '
               'The 2M/105M pair differs in both depth and width. One seed, eight immutable checkpoints; '
               'the 27M control and matched native traces remain pending. Learned weights were not changed. '
               'The CPU reader does not replace native checkpoint checksum validation.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('Use a fresh output path')
    run(args.out)
