"""Read immutable saved recurrence without running or modifying the CUDA model."""
import argparse
from pathlib import Path
import struct
import subprocess

import numpy as np

from native_experiment import read
from prose_founder import file_hash, write


def summarize_state(state, hidden, layers, observed):
    stride = 2 * hidden + 32 * 32
    if state.shape != (layers * stride,) or not np.isfinite(state).all():
        raise ValueError('Invalid associative recurrent-state shape or values')
    rows = []
    for layer in range(layers):
        at = layer * stride
        reset = state[at:at + hidden].astype(np.float64)
        trace = state[at + hidden:at + 2 * hidden].astype(np.float64)
        fast = state[at + 2 * hidden:at + stride].astype(np.float64)
        if np.abs(trace).max() > 1 + 1e-5:
            raise ValueError('Saved signed convex trace exceeds its expected bounds')
        absolute = np.abs(reset)
        rows.append(dict(layer_zero_based=layer,
            post_spike_membrane_absolute_mean=float(absolute.mean()),
            post_spike_membrane_absolute_p50=float(np.median(absolute)),
            post_spike_membrane_absolute_p90=float(np.quantile(absolute, .9)),
            post_spike_membrane_absolute_maximum=float(absolute.max()),
            last_step_zero_local_spike_surrogate_fraction_lower_bound=(float((absolute >= 1).mean()) if observed else None),
            signed_trace_absolute_mean=float(np.abs(trace).mean()),
            fast_matrix_frobenius=float(np.linalg.norm(fast))))
    return rows


def inspect(path):
    before = file_hash(path)
    with path.open('rb') as stream:
        meta = struct.unpack('<32Q', stream.read(256))
        if meta[1] != 6 or meta[5] != 1 or meta[17] != 5:
            raise ValueError('This diagnostic expects a single-stream associative v5 live checkpoint')
        expected = meta[4] * (2 * meta[3] + 1024)
        if meta[18] != expected or expected > 1024 * 1024:
            raise ValueError('Unexpected recurrent-state length')
        stream.seek(288 + 12 * meta[14])
        state = np.fromfile(stream, dtype='<f4', count=expected)
    rows = summarize_state(state, meta[3], meta[4], meta[24] > 0)
    if file_hash(path) != before:
        raise ValueError('Checkpoint changed during read-only inspection')
    return dict(checkpoint=path.as_posix(), checkpoint_sha256=before,
                source_observations=meta[24], parameters=meta[14], channels=meta[2],
                hidden=meta[3], layer_count=meta[4], recurrent_floats=meta[18],
                layers=rows, checkpoint_unchanged=True)


def run(founder, stages, out):
    protocol = read(founder / 'protocol.json')
    if protocol['profile'] != '105m' or not protocol['random_initialization'] or protocol['imported_weights']:
        raise ValueError('Expected the declared larger random founder')
    paths = [founder / 'initial.ckpt', *(founder / f'stage-{n}.ckpt' for n in stages)]
    rows = [inspect(path) for path in paths]
    expected = [0, *(protocol['source_schedule']['stages'][n - 1]['end_update'] for n in stages)]
    if [r['source_observations'] for r in rows] != expected:
        raise ValueError('Snapshot source boundaries differ from the declared founder')
    report = dict(kind='Exploratory CPU inspection of saved native recurrence',
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        founder_protocol_sha256=file_hash(founder / 'protocol.json'),
        scripts={str(p): file_hash(p) for p in (Path(__file__), Path('scripts/prose_founder.py'))},
        native_equations={p: file_hash(p) for p in ('src/trace_neuron.cuh', 'src/spike_lm.cu',
                                                   'src/model.cuh', 'src/checkpoint.cuh')},
        derivation='Signed threshold spike s = 1[u >= 1] - 1[u <= -1]; stored reset r = u - s. '
                   'A stored absolute r >= 1 implies absolute u >= 2. The implemented triangular '
                   'local spike surrogate is zero at absolute u >= 2. The initial zero state has no last step.',
        snapshots=rows, native_commands=0, model_updates=0,
        limitation='One saved state per checkpoint, at different source positions. This is not a matched-window '
                   'activity comparison, a time-averaged firing rate, a full gradient calculation or evidence '
                   'that a neuron never learns. Other paths and earlier times can carry gradients. No cause '
                   'of weak language quality or benefit from a controller has been established. SHA-256 '
                   'proves read identity; the diagnostic does not replace native payload validation.')
    write(out, report)
    for row in rows[1:]:
        values = [r['last_step_zero_local_spike_surrogate_fraction_lower_bound'] for r in row['layers']]
        print(row['source_observations'], 'observations; per-layer last-step zero local surrogate lower bound',
              f'{min(values):.2%} to {max(values):.2%}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--founder', type=Path, default=Path('runs/prose-105m-founder'))
    parser.add_argument('--stages', type=int, nargs='+', choices=(1, 2, 3, 4), default=[1, 2, 3])
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if len(args.stages) != len(set(args.stages)):
        parser.error('Stage snapshots must be distinct')
    run(args.founder, args.stages, args.out)
