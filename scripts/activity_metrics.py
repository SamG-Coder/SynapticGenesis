"""Read-only statistics for native selective/associative neuron traces.

These measurements describe a finite observation window, not task importance,
learning capacity, eligibility traces, or a pruning decision.
"""
import numpy as np


def distribution(values):
    values = np.asarray(values, dtype=np.float64)
    return dict(mean=float(values.mean()), minimum=float(values.min()),
                p50=float(np.quantile(values, .5)), p90=float(np.quantile(values, .9)),
                maximum=float(values.max()))


def activity(spikes, emission, membrane, output_weights, emission_epsilon=1e-6):
    """Summarize T x H native arrays and the C x H direct output projection."""
    spikes, emission, membrane, weights = [np.asarray(x, dtype=np.float64)
                                            for x in (spikes, emission, membrane, output_weights)]
    if (spikes.ndim != 2 or not all(spikes.shape) or emission.shape != spikes.shape
            or membrane.shape != spikes.shape or weights.ndim != 2
            or weights.shape[1] != spikes.shape[1] or not weights.shape[0]):
        raise ValueError('Incompatible activity/projection shapes')
    if not np.isfinite(emission_epsilon) or emission_epsilon < 0:
        raise ValueError('Invalid emission threshold')
    if not all(np.isfinite(x).all() for x in (spikes, emission, membrane, weights)):
        raise ValueError('Nonfinite native activity or weights')
    if not np.isin(spikes, (-1, 0, 1)).all():
        raise ValueError('Expected signed ternary spikes')
    decisions = (membrane >= 1).astype(np.float64) - (membrane <= -1).astype(np.float64)
    if not np.array_equal(spikes, decisions):
        raise ValueError('Spikes disagree with the fixed unit firing threshold')

    steps, hidden = spikes.shape
    fired, signal = spikes != 0, np.abs(emission) > emission_epsilon
    spike_rate, mean_abs = fired.mean(axis=0), np.abs(emission).mean(axis=0)
    column_norm = np.sqrt(np.square(weights).sum(axis=0))
    proxy = mean_abs * column_norm
    total_squared = float(np.square(emission).sum())
    silent_squared = float(np.square(emission[~fired]).sum())
    centered = emission - emission.mean(axis=0, keepdims=True)
    # The smaller Gram matrix has the same nonzero squared singular values.
    gram = centered @ centered.T if steps <= hidden else centered.T @ centered
    trace, squared_trace = float(np.trace(gram)), float(np.square(gram).sum())
    participation = trace * trace / squared_trace if squared_trace > 0 else None
    margin = np.minimum(np.abs(membrane - 1), np.abs(membrane + 1))

    def feature(index):
        return dict(neuron_zero_based=int(index), spike_rate=float(spike_rate[index]),
                    mean_absolute_emission=float(mean_abs[index]),
                    direct_output_column_l2=float(column_norm[index]),
                    direct_output_proxy=float(proxy[index]))

    ordering = np.argsort(proxy, kind='stable')
    count = min(8, hidden)
    return dict(steps=steps, hidden=hidden, emission_epsilon=emission_epsilon,
                spike_event_fraction=float(fired.mean()),
                nonzero_emission_event_fraction=float(signal.mean()),
                silent_spike_with_nonzero_emission_event_fraction=float((~fired & signal).mean()),
                silent_spike_emission_squared_fraction=(silent_squared / total_squared
                                                       if total_squared > 0 else None),
                never_spiked_neurons=int((~fired.any(axis=0)).sum()),
                never_emitted_above_epsilon_neurons=int((~signal.any(axis=0)).sum()),
                per_neuron_spike_rate=distribution(spike_rate),
                per_neuron_mean_absolute_emission=distribution(mean_abs),
                direct_output_proxy=distribution(proxy),
                centered_emission_participation_ratio=participation,
                participation_ratio_maximum=min(hidden, max(0, steps - 1)),
                near_firing_threshold_fraction=float((margin <= 1e-5).mean()),
                minimum_firing_threshold_margin=float(margin.min()),
                lowest_direct_output_proxy=[feature(i) for i in ordering[:count]],
                highest_direct_output_proxy=[feature(i) for i in ordering[-count:][::-1]])
