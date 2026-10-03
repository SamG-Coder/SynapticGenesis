"""Analytical final-block objective gradients on frozen native traces, on CPU."""
import numpy as np


def components(normalized, membrane, spikes, leaks, initial, *, layers, band=1.5,
               objective='membrane'):
    """Return the final block's input weight/bias/leak gradient at unit strength.

    Earlier blocks' states and the incoming boundary are constants. For an
    objective summed over layer membranes/spikes, these final-block coordinates
    cannot receive a contribution from an earlier layer's own objective. Their
    combined norm is consequently a lower bound on the full objective norm.
    This is not the language gradient or the norm after objectives are combined.
    """
    norm, u, s, leak, boundary = [np.asarray(a, dtype=np.float64)
                                for a in (normalized, membrane, spikes, leaks, initial)]
    if (norm.ndim != 2 or u.ndim != 2 or not all(norm.shape) or not all(u.shape)
            or norm.shape[0] != u.shape[0] or s.shape != u.shape
            or leak.shape != (u.shape[1],) or boundary.shape != leak.shape):
        raise ValueError('Expected matching time/channel and time/neuron arrays')
    if any(not np.isfinite(a).all() for a in (norm, u, s, leak, boundary)):
        raise ValueError('Nonfinite gradient input')
    if (type(layers) is not int or layers < 1 or not np.isfinite(band) or not 1 < band < 2
            or objective not in ('membrane', 'activity')):
        raise ValueError('Invalid objective policy')
    if not np.array_equal(s, (u >= 1).astype(float) - (u <= -1).astype(float)):
        raise ValueError('Spikes differ from saved fixed-threshold decisions')
    beta = np.exp(-np.logaddexp(0, -leak))
    scale = 1 / (u.size * layers)
    excess = np.maximum(np.abs(u) - band, 0)
    if objective == 'membrane':
        local = np.sign(u) * excess * scale
        value = .5 * np.square(excess).sum() * scale
    else:
        surrogate = .3 * (np.maximum(1 - np.abs(u - 1), 0) + np.maximum(1 - np.abs(u + 1), 0))
        local = s * surrogate * scale
        value = np.abs(s).sum() * scale
    adjoint = np.empty_like(u)
    carry = np.zeros(u.shape[1])
    for t in range(len(u) - 1, -1, -1):
        carry = local[t] + beta * carry
        adjoint[t] = carry
    previous = np.vstack((boundary, (u - s)[:-1]))
    gradients = dict(input_weight=adjoint.T @ norm, input_bias=adjoint.sum(axis=0),
                     leak=(adjoint * previous).sum(axis=0) * beta * (1 - beta))
    if any(not np.isfinite(a).all() for a in gradients.values()):
        raise ValueError('Nonfinite calculated gradient')
    return gradients, float(value)


def norms(gradients):
    squared = {name: float(np.square(value).sum(dtype=np.float64)) for name, value in gradients.items()}
    return dict(parts={name: value ** .5 for name, value in squared.items()},
                lower_bound=sum(squared.values()) ** .5)
