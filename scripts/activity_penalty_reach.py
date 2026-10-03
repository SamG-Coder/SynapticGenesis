"""Inspect regularizer reach in preserved native arrays; no neural computation."""
import argparse
from collections import Counter
import hashlib
from pathlib import Path
import struct

import numpy as np

from native_experiment import read
from prose_founder import file_hash, write


def require(condition, message):
    if not condition:
        raise ValueError(message)


def measure(membrane, spikes, chunk=128, band=1.5):
    u, s = np.asarray(membrane, dtype=np.float32), np.asarray(spikes, dtype=np.float32)
    require(u.ndim == 2 and all(u.shape) and s.shape == u.shape, 'Expected matching nonempty time/neuron arrays')
    require(isinstance(chunk, int) and chunk > 0 and u.shape[0] % chunk == 0, 'Invalid complete chunk geometry')
    require(np.isfinite(band) and 1 < band < 2, 'Candidate band must be inside the outer surrogate boundary')
    require(np.isfinite(u).all() and np.isfinite(s).all(), 'Nonfinite trace')
    expected = (u >= 1).astype(np.float32) - (u <= -1).astype(np.float32)
    require(np.array_equal(s, expected), 'Trace spike and membrane disagree')
    surrogate = np.float32(.3) * (np.maximum(0, 1 - np.abs(u - 1)) + np.maximum(0, 1 - np.abs(u + 1)))
    # This is the local contribution from the existing activity objective,
    # before recurrent propagation, global clipping, moments, or weight decay.
    old_local = s * surrogate
    excess = np.maximum(np.float32(0), np.abs(u) - np.float32(band))
    shape = (u.shape[0] // chunk, chunk, u.shape[1])
    no_signal = (old_local == 0).reshape(shape).all(axis=1)
    all_fired = (s != 0).reshape(shape).all(axis=1)
    saturated = no_signal & all_fired
    reaches = (excess != 0).reshape(shape).any(axis=1)
    return dict(positions=int(u.size), neuron_chunks=int(u.shape[0] // chunk * u.shape[1]),
        fired_positions=int(np.count_nonzero(s)), zero_surrogate_positions=int(np.count_nonzero(surrogate == 0)),
        zero_activity_local_positions=int(np.count_nonzero(old_local == 0)),
        fired_without_activity_local_positions=int(np.count_nonzero((s != 0) & (old_local == 0))),
        direct_penalty_positions=int(np.count_nonzero(excess)),
        direct_penalty_only_positions=int(np.count_nonzero((excess != 0) & (old_local == 0))),
        no_activity_signal_chunks=int(np.count_nonzero(no_signal)),
        fully_firing_without_activity_signal_chunks=int(np.count_nonzero(saturated)),
        direct_penalty_reaches_fully_firing_chunks=int(np.count_nonzero(saturated & reaches)),
        unit_penalty_sum=float(.5 * np.square(excess.astype(np.float64)).sum()))


def summarize(counts):
    positions, chunks = counts['positions'], counts['neuron_chunks']
    return dict(counts, spike_event_fraction=counts['fired_positions']/positions,
        zero_local_spike_surrogate_fraction=counts['zero_surrogate_positions']/positions,
        fired_without_activity_local_fraction=counts['fired_without_activity_local_positions']/positions,
        direct_penalty_only_fraction=counts['direct_penalty_only_positions']/positions,
        fully_firing_without_activity_signal_chunk_fraction=counts['fully_firing_without_activity_signal_chunks']/chunks,
        no_activity_signal_chunk_fraction=counts['no_activity_signal_chunks']/chunks,
        unit_penalty_mean=counts['unit_penalty_sum']/positions)


def inspect(spec_path, out):
    require(not out.exists(), 'Use a fresh activity-penalty report')
    spec = read(spec_path)
    require(spec['version']=='membrane-penalty-audit-v1' and spec['chunk']==128 and spec['candidate_band']==1.5,
            'Unknown diagnostic policy')
    for key in ('trace_report','publication_audit'):
        require(file_hash(spec[key])==spec[key+'_sha256'], 'Published trace evidence changed')
    report, audit = read(spec['trace_report']), read(spec['publication_audit'])
    require(report['complete'] and audit['passed'] and audit['published_report_sha256']==spec['trace_report_sha256'],
            'Trace publication is incomplete or inconsistent')
    require(report['protocol']['profiles']==spec['profiles'] and report['protocol']['stages']==spec['stages'],
            'Trace case selection changed')
    coordinates = [(r['profile'],r['stage']) for r in report['records']]
    require(coordinates == [(p,s) for p in spec['profiles'] for s in spec['stages']], 'Missing or duplicate trace case')
    input_hashes = {}
    result = []
    for row in report['records']:
        checkpoint = Path(row['checkpoint'])
        require(file_hash(checkpoint)==row['checkpoint_sha256'], 'Checkpoint changed')
        with checkpoint.open('rb') as stream:
            meta = struct.unpack('<32Q', stream.read(256))
            hp = struct.unpack('<8f', stream.read(32))
        require(meta[1]==6 and meta[5]==1 and meta[24]==row['source_observations'], 'Unexpected trace checkpoint')
        hidden, layers = meta[3:5]
        require(len(row['windows'])==len(report['protocol']['windows'])==8, 'Incomplete fixed windows')
        pooled = [Counter() for _ in range(layers)]
        windows = []
        for observed, declared in zip(row['windows'],report['protocol']['windows']):
            require(observed['window']==declared['name'] and declared['target_bytes']==1024, 'Window differs')
            require(file_hash(declared['file'])==declared['sha256'], 'Observed source bytes changed')
            root = Path(spec['trace_root'])/f"{row['profile']}-stage-{row['stage']}"/observed['window']
            per_layer = []
            for layer in range(layers):
                arrays = []
                for name in ('u','spikes'):
                    path = root/f'layer-{layer}-{name}.f32'
                    raw = path.read_bytes()
                    digest = hashlib.sha256(raw).hexdigest()
                    require(digest==observed['trace_sha256'][path.name], 'Native trace changed: '+str(path))
                    require(len(raw)==1024*hidden*4, 'Native trace size differs')
                    arrays.append(np.frombuffer(raw,dtype='<f4').reshape(1024,hidden))
                    input_hashes[path.as_posix()] = digest
                counts = measure(*arrays,chunk=spec['chunk'],band=spec['candidate_band'])
                require(counts['direct_penalty_reaches_fully_firing_chunks']==
                        counts['fully_firing_without_activity_signal_chunks'], 'Candidate misses a saturated chunk')
                statistics = summarize(counts)
                prior = observed['layers'][layer]
                require(abs(statistics['zero_local_spike_surrogate_fraction']-
                            prior['zero_local_spike_surrogate_fraction'])<1e-12, 'Earlier surrogate statistic differs')
                require(abs(statistics['spike_event_fraction']-prior['spike_event_fraction'])<1e-12,
                        'Earlier firing statistic differs')
                pooled[layer].update(counts)
                per_layer.append(dict(layer_zero_based=layer,**statistics))
            windows.append(dict(window=observed['window'],layers=per_layer))
        aggregate = Counter()
        for counts in pooled: aggregate.update(counts)
        measured = dict(profile=row['profile'],stage=row['stage'],parameters=row['parameters'],
            source_observations=row['source_observations'],checkpoint=checkpoint.as_posix(),
            checkpoint_sha256=row['checkpoint_sha256'],saved_activity_cost=hp[4],
            aggregate=summarize(aggregate),layers=[dict(layer_zero_based=i,**summarize(c)) for i,c in enumerate(pooled)],
            windows=windows)
        result.append(measured)
        print(row['profile'],row['stage'],'fully firing chunks without local activity signal:',
              measured['aggregate']['fully_firing_without_activity_signal_chunk_fraction'],flush=True)
    value = dict(complete=True,policy=spec,policy_sha256=file_hash(spec_path),cases=result,
        authenticated_native_arrays=input_hashes,native_arrays_read=len(input_hashes),
        implementation_sha256={p:file_hash(p) for p in ('scripts/activity_penalty_reach.py',
            'experiments/membrane_penalty.cuh','src/trace_neuron.cuh','src/model.cuh','src/spike_lm.cu')},
        new_native_forwards=0,new_model_updates=0,candidate_used_in_learning=False,
        limitations=[
            'Post-hoc analysis of earlier 2M/27M/105M reset-window traces, not the active 411M learner or its live chunks.',
            'All audited checkpoints have their recorded activity-cost setting; unit-strength reach is a counterfactual when that cost is zero.',
            'Chunk geometry partitions saved 1024-byte forwards into 128-byte gradient intervals, without rerunning forward recurrence.',
            'A zero own-layer activity contribution does not imply zero task gradients, upstream/downstream contributions, optimizer movement, or permanent neuron inactivity.',
            'The direct membrane penalty is a candidate objective; no language-quality, retention, CUDA throughput, or biological benefit is demonstrated.'
        ])
    write(out,value)
    return value


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--policy',type=Path,default=Path('data/membrane-penalty-audit-v1.json'))
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    inspect(args.policy,args.out)
