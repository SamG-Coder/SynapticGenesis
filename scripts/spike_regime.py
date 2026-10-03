"""Finite-window response and local-surrogate statistics for native traces."""
import numpy as np


def absolute_distribution(values):
    absolute = np.abs(values)
    return dict(mean=float(absolute.mean(dtype=np.float64)), p50=float(np.quantile(absolute, .5)),
                p90=float(np.quantile(absolute, .9)), p99=float(np.quantile(absolute, .99)),
                maximum=float(absolute.max()))


def regime(spikes, emission, membrane, drive, segments=None):
    arrays = [np.asarray(a, dtype=np.float32) for a in (spikes, emission, membrane, drive)]
    spikes, emission, membrane, drive = arrays
    if spikes.ndim != 2 or not all(spikes.shape) or any(a.shape != spikes.shape for a in arrays):
        raise ValueError('Expected matching nonempty time-by-neuron native arrays')
    if any(not np.isfinite(a).all() for a in arrays):
        raise ValueError('Nonfinite diagnostic array')
    expected = (membrane >= 1).astype(np.float32) - (membrane <= -1).astype(np.float32)
    if not np.array_equal(spikes, expected):
        raise ValueError('Spikes disagree with native fixed-threshold decisions')
    steps, hidden = spikes.shape
    segments = [steps] if segments is None else list(segments)
    if not segments or any(n < 1 or int(n) != n for n in segments) or sum(segments) != steps:
        raise ValueError('Invalid independent window lengths')
    segments = [int(n) for n in segments]
    counts = np.stack([(spikes == value).sum(axis=0) for value in (-1, 0, 1)])
    probabilities = counts.astype(np.float64) / steps
    entropy_terms = np.zeros_like(probabilities)
    positive = probabilities > 0
    entropy_terms[positive] = -probabilities[positive] * np.log2(probabilities[positive])
    per_neuron_entropy = entropy_terms.sum(axis=0)
    constant = (counts == steps).any(axis=0)
    changes, sign_flips, pairs, at = 0, 0, 0, 0
    for length in segments:
        part = spikes[at:at + length]
        if length > 1:
            changes += int(np.count_nonzero(part[1:] != part[:-1]))
            sign_flips += int(np.count_nonzero(part[1:] * part[:-1] < 0))
            pairs += (length - 1) * hidden
        at += length
    # Evaluate in float32, including cancellation very close to zero.
    surrogate = np.float32(.3) * (np.maximum(0, 1 - np.abs(membrane - 1))
                                    + np.maximum(0, 1 - np.abs(membrane + 1)))
    std = emission.std(axis=0, dtype=np.float64)
    return dict(observed_steps=steps, hidden=hidden, independent_windows=len(segments),
        spike_negative_fraction=float(probabilities[0].mean()),
        spike_zero_fraction=float(probabilities[1].mean()),
        spike_positive_fraction=float(probabilities[2].mean()),
        spike_event_fraction=float(1 - probabilities[1].mean()),
        constant_spike_neuron_fraction=float(constant.mean()),
        mean_per_neuron_marginal_spike_entropy_bits=float(per_neuron_entropy.mean()),
        maximum_marginal_entropy_bits=float(np.log2(3)),
        within_window_adjacent_pairs=pairs,
        within_window_ternary_change_fraction=changes / pairs if pairs else None,
        within_window_direct_sign_flip_fraction=sign_flips / pairs if pairs else None,
        zero_local_spike_surrogate_fraction=float((surrogate == 0).mean()),
        local_spike_surrogate_mean=float(surrogate.mean(dtype=np.float64)),
        membrane_absolute=absolute_distribution(membrane), drive_absolute=absolute_distribution(drive),
        emission_absolute=absolute_distribution(emission),
        emission_per_neuron_temporal_std_mean=float(std.mean()),
        emission_per_neuron_temporal_std_p50=float(np.median(std)),
        near_constant_emission_neuron_fraction=float((std <= 1e-6).mean()),
        constant_emission_tolerance=1e-6)
