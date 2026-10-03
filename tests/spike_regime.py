"""Distinguish dense constant firing from informative signed responses."""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from spike_regime import regime


def measure(u, segments=None):
    u = np.asarray(u, dtype=np.float32).reshape(-1, 1)
    spikes = (u >= 1).astype(np.float32) - (u <= -1).astype(np.float32)
    return regime(spikes, spikes, u, u, segments)


constant = measure([2.5] * 4)
assert constant['spike_event_fraction'] == constant['zero_local_spike_surrogate_fraction'] == 1
assert constant['constant_spike_neuron_fraction'] == 1
assert constant['mean_per_neuron_marginal_spike_entropy_bits'] == 0
assert constant['within_window_ternary_change_fraction'] == 0
assert constant['near_constant_emission_neuron_fraction'] == 1
alternating = measure([2.5, -2.5, 2.5, -2.5])
assert alternating['spike_event_fraction'] == alternating['zero_local_spike_surrogate_fraction'] == 1
assert alternating['constant_spike_neuron_fraction'] == 0
assert alternating['mean_per_neuron_marginal_spike_entropy_bits'] == 1
assert alternating['within_window_direct_sign_flip_fraction'] == 1
assert alternating['emission_per_neuron_temporal_std_mean'] == 1
split = measure([2.5, 2.5, -2.5, -2.5], [2, 2])
assert split['within_window_adjacent_pairs'] == 2
assert split['within_window_ternary_change_fraction'] == 0
assert split['constant_spike_neuron_fraction'] == 0
assert split['mean_per_neuron_marginal_spike_entropy_bits'] == 1
boundary = measure([-2, -1, 0, 1, 2])
assert abs(boundary['zero_local_spike_surrogate_fraction'] - .6) < 1e-12
assert abs(boundary['local_spike_surrogate_mean'] - .12) < 1e-7
assert measure([0])['within_window_ternary_change_fraction'] is None
assert measure([1e-9, -1e-9])['zero_local_spike_surrogate_fraction'] == 1
rejected = 0
for arguments in [([1, 2], [1]), ([np.nan], None), ([np.inf], None)]:
    try:
        measure(*arguments)
    except ValueError:
        rejected += 1
    else:
        raise AssertionError('Invalid trace accepted')
try:
    regime(np.ones((2, 1)), np.ones((2, 1)), np.zeros((2, 1)), np.ones((2, 1)))
except ValueError:
    rejected += 1
else:
    raise AssertionError('Inconsistent spike decisions accepted')
assert rejected == 4
print('Dense constant/alternating responses, independent windows, exact boundaries and four rejections passed.')
