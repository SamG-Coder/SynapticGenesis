"""Analytic fixtures for read-only activity statistics; no GPU or learning."""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from activity_metrics import activity


def check():
    spikes = np.array([[1., 0.], [0., -1.]])
    membrane = np.array([[1.2, .3], [.2, -1.5]])
    emission = np.array([[2., 1.], [3., -4.]])
    weights = np.array([[3., 0.], [4., 2.]])
    measured = activity(spikes, emission, membrane, weights)
    assert measured['spike_event_fraction'] == .5
    assert measured['silent_spike_with_nonzero_emission_event_fraction'] == .5
    assert abs(measured['silent_spike_emission_squared_fraction'] - 1/3) < 1e-15
    assert measured['per_neuron_mean_absolute_emission']['mean'] == 2.5
    assert measured['direct_output_proxy']['minimum'] == 5.
    assert measured['direct_output_proxy']['maximum'] == 12.5
    assert measured['centered_emission_participation_ratio'] == 1.
    assert measured['highest_direct_output_proxy'][0]['neuron_zero_based'] == 0
    assert measured['near_firing_threshold_fraction'] == 0.

    # Constant offsets and nonzero global scaling preserve centered diversity.
    for changed in (emission + [10., -5.], emission * 3):
        assert activity(spikes, changed, membrane, weights)['centered_emission_participation_ratio'] == 1.
    # Two orthogonal centered signals have effective dimension exactly two.
    independent = np.array([[1., 1.], [1., -1.], [-1., 1.], [-1., -1.]])
    assert activity(np.zeros_like(independent), independent, np.zeros_like(independent),
                    weights)['centered_emission_participation_ratio'] == 2.
    empty = activity(np.zeros((3, 2)), np.zeros((3, 2)), np.zeros((3, 2)), weights)
    assert empty['never_spiked_neurons'] == empty['never_emitted_above_epsilon_neurons'] == 2
    assert empty['silent_spike_emission_squared_fraction'] is None
    assert empty['centered_emission_participation_ratio'] is None
    single = activity(np.zeros((1, 2)), np.ones((1, 2)), np.zeros((1, 2)), weights)
    assert single['centered_emission_participation_ratio'] is None
    boundary = activity(np.array([[1., -1.]]), np.ones((1, 2)), np.array([[1., -1.]]), weights)
    assert boundary['near_firing_threshold_fraction'] == 1.

    invalid = [
        (spikes, emission[:, :1], membrane, weights),
        (spikes, emission, membrane, weights[:, :1]),
        (spikes + .1, emission, membrane, weights),
        (spikes, emission, np.zeros_like(membrane), weights),
        (spikes, emission * np.nan, membrane, weights),
        (spikes[:0], emission[:0], membrane[:0], weights),
    ]
    for case in invalid:
        try:
            activity(*case)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid activity fixture accepted')
    for epsilon in (-1, float('nan')):
        try:
            activity(spikes, emission, membrane, weights, epsilon)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid threshold accepted')
    print('Activity statistics: analytic energy/projection/rank fixtures and eight rejection cases pass.')


if __name__ == '__main__':
    check()
