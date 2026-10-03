"""Independent CPU autograd/finite-difference checks of frozen-trace gradients."""
import argparse
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from membrane_gradient import components, norms
from membrane_gradient_audit import final_leaks
from native_experiment import sha
from prose_founder import write
from probes_cli import Reference

torch.set_num_threads(2)


class Spike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        ctx.save_for_backward(value)
        return (value >= 1).double() - (value <= -1).double()

    @staticmethod
    def backward(ctx, gradient):
        value, = ctx.saved_tensors
        return gradient * .3 * ((1 - (value - 1).abs()).clamp_min(0) +
                                (1 - (value + 1).abs()).clamp_min(0))


def fixture(length, layers, seed):
    rng = np.random.default_rng(seed)
    normalized = rng.normal(size=(length, 3))
    weight = rng.normal(size=(5, 3)) * .8
    bias = rng.normal(size=5) * .3
    leak = np.linspace(-2, 3, 5)
    boundary = np.array([-3., -.4, 0., 2., 5.])
    return normalized, weight, bias, leak, boundary, layers


def forward(fix, objective, forced=None):
    norm, weight, bias, leak, boundary, layers = fix
    wi, bi, decay = [torch.tensor(a, dtype=torch.float64, requires_grad=True) for a in (weight, bias, leak)]
    drive = torch.tensor(norm) @ wi.T + bi
    reset = torch.tensor(boundary)
    potentials, events = [], []
    for i, sample in enumerate(drive):
        membrane = decay.sigmoid() * reset + sample
        spike = Spike.apply(membrane) if forced is None else torch.tensor(forced[i])
        reset = membrane - spike.detach()
        potentials.append(membrane)
        events.append(spike)
    potentials, events = torch.stack(potentials), torch.stack(events)
    loss = (.5 * (potentials.abs() - 1.5).clamp_min(0).square().mean() if objective == 'membrane'
            else events.abs().mean()) / layers
    return loss, (wi, bi, decay), potentials.detach().numpy(), events.detach().numpy()


def check(out):
    assert not out.exists(), 'Use a fresh audit report'
    comparisons, finite_differences = [], []
    last_inputs = None
    for length, layers, seed in [(1, 1, 17), (5, 4, 27), (128, 8, 37), (128, 4, 47)]:
        fix = fixture(length, layers, seed)
        for objective in ('activity', 'membrane'):
            loss, parameters, u, spikes = forward(fix, objective)
            expected = torch.autograd.grad(loss, parameters)
            inputs = (fix[0], u, spikes, fix[3], fix[4])
            actual, value = components(*inputs, layers=layers, objective=objective)
            assert abs(value - loss.item()) < 1e-12
            errors = {}
            for name, reference in zip(('input_weight', 'input_bias', 'leak'), expected):
                errors[name] = float(np.abs(actual[name] - reference.detach().numpy()).max())
                assert errors[name] < 2e-12, (length, objective, name, errors[name])
            assert norms(actual)['lower_bound'] > 0
            comparisons.append(dict(length=length, layers=layers, objective=objective, max_errors=errors))
            last_inputs = inputs
            if length == 5 and objective == 'membrane':
                # Hold the reset spikes fixed when perturbing the smooth surrogate path.
                for field, name, index in [(1, 'input_weight', (0, 0)), (1, 'input_weight', (4, 2)),
                                            (2, 'input_bias', (1,)), (3, 'leak', (0,)), (3, 'leak', (4,))]:
                    epsilon, losses = 1e-6, []
                    for sign in (-1, 1):
                        moved = list(fix)
                        moved[field] = fix[field].copy()
                        moved[field][index] += sign * epsilon
                        altered, _, _, _ = forward(moved, 'membrane', forced=spikes)
                        losses.append(altered.item())
                    numerical = (losses[1] - losses[0]) / (2 * epsilon)
                    error = abs(numerical - actual[name][index])
                    assert error < 2e-9, (name, error)
                    finite_differences.append(dict(parameter=name, index=list(index), absolute_error=error))
    # All-zero activity surrogates can coexist with a nonzero direct membrane gradient.
    norm = np.ones((8, 3))
    u = np.full((8, 5), 10.)
    spikes = np.ones_like(u)
    other = (np.zeros(5), np.zeros(5))
    old, _ = components(norm, u, spikes, *other, layers=4, objective='activity')
    new, _ = components(norm, u, spikes, *other, layers=4)
    assert norms(old)['lower_bound'] == 0 and norms(new)['lower_bound'] > 0
    # Full-coordinate norm is at least this measured subset; no such assertion
    # is valid for a separate language gradient plus a regularizer.
    negative = {k: -v for k, v in new.items()}
    combined = {k: new[k] + negative[k] for k in new}
    assert norms(combined)['lower_bound'] == 0

    rejected = []
    for name, changes, options in [
        ('shape', {0: np.zeros((2, 3))}, {}),
        ('nan', {1: np.full_like(last_inputs[1], np.nan)}, {}),
        ('spike', {2: np.zeros_like(last_inputs[2])}, {}),
        ('boundary', {4: np.zeros(4)}, {}),
        ('leak', {3: np.zeros(4)}, {}),
        ('layers', {}, {'layers': 0}),
        ('noninteger-layers', {}, {'layers': 1.5}),
        ('band', {}, {'band': 2.1}),
        ('unknown-objective', {}, {'objective': 'other'}),
    ]:
        args = list(last_inputs)
        for key, value in changes.items():
            args[key] = value
        try:
            components(*args, **(dict(layers=4) | options))
        except ValueError:
            rejected.append(name)
        else:
            raise AssertionError('Invalid input accepted: ' + name)
    checkpoint = Path('runs/prose-size-panel/founder-2m/initial.ckpt')
    original = sha(checkpoint)
    meta, hp, leaks = final_leaks(checkpoint)
    reference = Reference(checkpoint)
    assert (reference.c, reference.h, reference.l) == tuple(meta[2:5])
    assert np.array_equal(leaks, reference.blocks[-1][5].numpy())
    assert hp[2] == 1 and original == sha(checkpoint)
    result = dict(passed=True, autograd_cases=comparisons, frozen_reset_finite_differences=finite_differences,
        saturation_negative_control=True, combined_objective_cancellation_control=True,
        invalid_inputs_rejected=rejected, real_checkpoint_layout_matches_existing_reference=True,
        checkpoint_sha256=original, cpu_only=True, native_commands=0, new_model_updates=0,
        implementation_sha256={p.as_posix(): sha(p) for p in [Path(__file__),
            Path('scripts/membrane_gradient.py'), Path('scripts/membrane_gradient_audit.py')]})
    write(out, result)
    print('Eight CPU autograd cases, five finite differences, two controls and nine rejections passed.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    check(parser.parse_args().out)
