"""Independent scalar checks for the preserved-trace penalty-reach analysis."""
import argparse
from collections import Counter
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from activity_penalty_reach import measure, summarize
from prose_founder import file_hash, write


def scalar_counts(u, chunk, band):
    result = Counter()
    for start in range(0,len(u),chunk):
        for j in range(u.shape[1]):
            result['neuron_chunks'] += 1
            no_activity, all_fired, reached = True, True, False
            for t in range(start,start+chunk):
                v = float(u[t,j])
                s = int(v>=1)-int(v<=-1)
                # Test fixtures use exactly representable values, so the
                # scalar equations do not depend on vector reduction order.
                surrogate = .3*(max(0,1-abs(v-1))+max(0,1-abs(v+1)))
                old = s*surrogate
                excess = max(0,abs(v)-band)
                result['positions'] += 1
                result['fired_positions'] += s!=0
                result['zero_surrogate_positions'] += surrogate==0
                result['zero_activity_local_positions'] += old==0
                result['fired_without_activity_local_positions'] += s!=0 and old==0
                result['direct_penalty_positions'] += excess!=0
                result['direct_penalty_only_positions'] += excess!=0 and old==0
                result['unit_penalty_sum'] += .5*excess*excess
                no_activity &= old==0
                all_fired &= s!=0
                reached |= excess!=0
            result['no_activity_signal_chunks'] += no_activity
            result['fully_firing_without_activity_signal_chunks'] += no_activity and all_fired
            result['direct_penalty_reaches_fully_firing_chunks'] += no_activity and all_fired and reached
    return dict(result)


def check(out):
    assert not out.exists(), 'Use a fresh reach-check report'
    cases = [
        np.full((8,3),3,dtype=np.float32),
        np.full((8,3),-3,dtype=np.float32),
        np.zeros((8,3),dtype=np.float32),
        np.full((8,3),.5,dtype=np.float32),
        np.full((8,3),1.25,dtype=np.float32),
        np.array([[3,-3,0],[-3,3,.5],[2,-2,-.5],[1.75,-1.75,1.5]]*2,dtype=np.float32),
        np.array([[3]*3]*4+[[1.25]*3]*4,dtype=np.float32),
    ]
    checks = 0
    for u in cases:
        s=(u>=1).astype(np.float32)-(u<=-1).astype(np.float32)
        for chunk in (1,2,4,8):
            counts = measure(u,s,chunk,1.5)
            assert counts == scalar_counts(u,chunk,1.5)
            fractions = summarize(counts)
            assert 0 <= fractions['fully_firing_without_activity_signal_chunk_fraction'] <= 1
            checks += 1
    mixed = cases[-1]
    spikes=(mixed>=1).astype(np.float32)
    assert measure(mixed,spikes,4)['fully_firing_without_activity_signal_chunks']==3
    assert measure(mixed,spikes,8)['fully_firing_without_activity_signal_chunks']==0
    assert measure(cases[0],np.ones_like(cases[0]),4)['fully_firing_without_activity_signal_chunks']==6
    silent = measure(cases[3],np.zeros_like(cases[3]),4)
    assert silent['no_activity_signal_chunks']==6 and silent['fully_firing_without_activity_signal_chunks']==0
    assert silent['zero_surrogate_positions']==0 and silent['zero_activity_local_positions']==24
    bad = [
        (np.empty((0,2)),np.empty((0,2)),1,1.5),
        (np.zeros((2,2)),np.zeros((2,3)),1,1.5),
        (np.zeros((2,2)),np.zeros((2,2)),3,1.5),
        (np.zeros((2,2)),np.zeros((2,2)),0,1.5),
        (np.zeros((2,2)),np.zeros((2,2)),1,2),
        (np.zeros((2,2)),np.zeros((2,2)),1,float('nan')),
        (np.full((2,2),float('nan')),np.zeros((2,2)),1,1.5),
        (np.full((2,2),3),np.zeros((2,2)),1,1.5),
    ]
    rejected=0
    for args in bad:
        try: measure(*args)
        except ValueError: rejected+=1
        else: raise AssertionError('Malformed trace accepted')
    result=dict(passed=True,scalar_window_checks=checks,malformed_cases_rejected=rejected,
        silent_activity_gradient_distinguished_from_zero_spike_surrogate=True,
        backward_chunk_boundary_changes_reach=True,both_signed_firing_polarities_checked=True,
        implementation_sha256={str(p):file_hash(p) for p in (Path(__file__),Path('scripts/activity_penalty_reach.py'))},
        native_forwards=0,learning_updates=0)
    write(out,result)
    print(f'{checks} scalar window checks, {rejected} malformed cases; no model work.')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    check(parser.parse_args().out)
