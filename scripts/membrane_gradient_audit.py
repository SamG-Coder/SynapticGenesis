"""Audit objective-gradient scale using immutable native final-layer traces."""
import argparse
from pathlib import Path
import struct

import numpy as np

from membrane_gradient import components, norms
from native_experiment import read
from prose_founder import file_hash, write


def require(condition, message):
    if not condition:
        raise ValueError(message)


def final_leaks(path):
    """Read only the associative final block's leak vector, never mutable state."""
    with Path(path).open('rb') as stream:
        meta = struct.unpack('<32Q', stream.read(256))
        hp = struct.unpack('<8f', stream.read(32))
        c, h, layers = meta[2:5]
        require(meta[0] == 0x314d4c53434e5042 and meta[1] == 6 and meta[5] == 1 and
                meta[17] == 5 and 8 <= c <= 2048 and 8 <= h <= 8192 and 1 <= layers <= 32,
                'Expected a preserved ordinary associative checkpoint')
        block_size = 3 * c * h + 5 * h + 35 * c + 98 * h + 98
        parameters = 512 * c + c + 256 + layers * block_size
        require(meta[14] == parameters, 'Associative parameter layout changed')
        offset = 256 * c + (layers - 1) * block_size + c + h * c + h + c * h + c
        require(offset + h <= parameters, 'Leak vector lies outside parameters')
        stream.seek(288 + 4 * offset)
        leak = np.frombuffer(stream.read(4 * h), dtype='<f4').copy()
        require(leak.size == h and np.isfinite(leak).all(), 'Invalid leak vector')
    return meta, hp, leak


def distribution(values):
    return {name: float(np.quantile(values, q)) for name, q in
            [('min', 0), ('p50', .5), ('p90', .9), ('max', 1)]}


def audit(policy_path, out):
    require(not out.exists(), 'Use a fresh gradient audit report')
    policy = read(policy_path)
    require(policy['version'] == 'membrane-gradient-audit-v1' and policy['chunk'] == 128 and
            policy['band'] == 1.5, 'Unknown gradient audit policy')
    for key in ('trace_report', 'publication_audit'):
        require(file_hash(policy[key]) == policy[key + '_sha256'], 'Published trace evidence changed')
    report, publication = read(policy['trace_report']), read(policy['publication_audit'])
    require(report['complete'] and publication['passed'] and
            publication['published_report_sha256'] == policy['trace_report_sha256'], 'Unverified trace panel')
    coordinates = [(r['profile'], r['stage']) for r in report['records']]
    require(coordinates == [(p, s) for p in policy['profiles'] for s in policy['stages']],
            'Missing, extra or reordered trace cases')
    windows = report['protocol']['windows']
    require(len(windows) == 8 and all(w['target_bytes'] == 1024 for w in windows), 'Unexpected trace geometry')
    input_hashes = {str(policy_path): file_hash(policy_path),
                    policy['trace_report']: policy['trace_report_sha256'],
                    policy['publication_audit']: policy['publication_audit_sha256']}
    for window in windows:
        require(file_hash(window['file']) == window['sha256'], 'Trace source changed')
        input_hashes[window['file']] = window['sha256']
    cases, array_count = [], 0
    for row in report['records']:
        path = Path(row['checkpoint'])
        identity = file_hash(path)
        require(identity == row['checkpoint_sha256'], 'Preserved checkpoint changed')
        input_hashes[path.as_posix()] = identity
        meta, hp, leak = final_leaks(path)
        require(meta[24] == row['source_observations'] and meta[14] == row['parameters'], 'Checkpoint identity differs')
        c, h, layers = map(int, meta[2:5])
        require(len(row['windows']) == len(windows), 'Incomplete native windows')
        chunks = []
        for observed, declared in zip(row['windows'], windows):
            require(observed['window'] == declared['name'], 'Native window order differs')
            root = Path(policy['trace_root']) / f"{row['profile']}-stage-{row['stage']}" / declared['name']
            arrays = {}
            for name, width in [('norm', c), ('u', h), ('spikes', h)]:
                source = root / f'layer-{layers - 1}-{name}.f32'
                identity = file_hash(source)
                require(identity == observed['trace_sha256'][source.name], 'Native array changed')
                raw = source.read_bytes()
                require(len(raw) == 1024 * width * 4, 'Native array shape differs')
                arrays[name] = np.frombuffer(raw, dtype='<f4').reshape(1024, width)
                input_hashes[source.as_posix()] = identity
                array_count += 1
            for start in range(0, 1024, policy['chunk']):
                stop = start + policy['chunk']
                boundary = (arrays['u'][start - 1] - arrays['spikes'][start - 1]
                            if start else np.zeros(h, dtype=np.float32))
                measured = {}
                for objective in ('activity', 'membrane'):
                    gradients, value = components(arrays['norm'][start:stop], arrays['u'][start:stop],
                        arrays['spikes'][start:stop], leak, boundary, layers=layers, band=policy['band'],
                        objective=objective)
                    measured[objective] = dict(**norms(gradients), final_layer_objective_contribution=value)
                    del gradients
                chunks.append(dict(window=declared['name'], chunk_start=start, **measured))
        summary = {objective: distribution([r[objective]['lower_bound'] for r in chunks])
                   for objective in ('activity', 'membrane')}
        strengths = []
        for strength in policy['illustrative_strengths']:
            values = [strength * r['membrane']['lower_bound'] for r in chunks]
            strengths.append(dict(strength=strength, final_block_gradient_norm=distribution(values),
                lower_bound_exceeds_saved_clip=sum(v > hp[2] for v in values), chunks=len(values)))
        require(file_hash(path) == row['checkpoint_sha256'], 'Checkpoint changed during audit')
        cases.append(dict(profile=row['profile'], stage=row['stage'], checkpoint=path.as_posix(),
            checkpoint_sha256=row['checkpoint_sha256'], source_observations=meta[24], parameters=meta[14],
            final_layer_zero_based=layers - 1, channels=c, hidden=h, layers=layers,
            saved_gradient_clip=hp[2], saved_activity_cost=hp[4], unit_gradient_lower_bound=summary,
            illustrative_strengths=strengths, chunks=chunks))
        print(row['profile'], row['stage'], 'unit membrane final-block gradient:', summary['membrane'], flush=True)
    sources = ['scripts/membrane_gradient.py', 'scripts/membrane_gradient_audit.py',
               'src/trace_neuron.cuh', 'src/model.cuh', 'experiments/membrane_penalty.cuh']
    result = dict(complete=True, policy=policy, policy_sha256=file_hash(policy_path), cases=cases,
        authenticated_inputs=input_hashes, native_arrays_read=array_count,
        implementation_sha256={name: file_hash(name) for name in sources},
        chunks_analyzed=sum(len(row['chunks']) for row in cases), new_native_forwards=0,
        new_model_updates=0, coefficients_selected=False, candidate_used_in_learning=False,
        scope='Double-precision analytical objective gradients on frozen native FP32 values; final-block '
              'input weight, input bias and membrane leak only. The incoming state is detached and the '
              'backward carry ends at each 128-byte boundary. Normalization includes all model layers.',
        limitations=[
            'These earlier 2M/27M/105M snapshots are not the active 411M model or its actual live training chunks.',
            'The coordinate norm is a lower bound on the candidate objective alone. It is not a lower bound '
            'on the combined language and regularizer gradient, which can reinforce or cancel.',
            'Other parameters may increase the full objective norm; a small bound does not prove a weak objective.',
            'Spike-activity gradients use the declared surrogate. They are not derivatives of the discontinuous '
            'hard-spike function under ordinary finite differences.',
            'Native CUDA gradient agreement, clipping, AdamW displacement and language/retention effects are not '
            'measured here. A gradient ratio does not select an optimal coefficient.',
            'Saved FP32 traces are interpreted in double precision; these are not bit-identical GPU reductions.',
        ])
    write(out, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', type=Path, default=Path('data/membrane-gradient-audit-v1.json'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    audit(args.policy, args.out)
