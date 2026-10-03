"""Independent CPU autograd oracle. This is verification only; the trainer is C++/CUDA.
Uses an existing CPU PyTorch installation, and only deterministic synthetic fixtures.
No model or tokenizer is downloaded. No oracle weights become production checkpoints.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from associative_reference import forward as associative_forward


class SignedSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u):
        ctx.save_for_backward(u)
        return (u >= 1).float() - (u <= -1).float()

    @staticmethod
    def backward(ctx, ds):
        (u,) = ctx.saved_tensors
        surrogate = 0.3 * (torch.clamp(1 - (u - 1).abs(), min=0)
                           + torch.clamp(1 - (u + 1).abs(), min=0))
        return ds * surrogate


class AdaptiveSignedSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, threshold):
        ctx.save_for_backward(u, threshold)
        return (u >= threshold).float() - (u <= -threshold).float()

    @staticmethod
    def backward(ctx, ds):
        u, threshold = ctx.saved_tensors
        positive = .3 * torch.clamp(1 - (u - threshold).abs(), min=0)
        negative = .3 * torch.clamp(1 - (u + threshold).abs(), min=0)
        return ds * (positive + negative), ds * (negative - positive)


def check(directory):
    directory = Path(directory)
    cfg = json.loads((directory / 'fixture.json').read_text())
    c, h, layers = cfg['channels'], cfg['hidden'], cfg['layers']
    batch, time = cfg['batch'], cfg['context']
    adaptive = cfg.get('cell', 1) == 2
    traced = cfg.get('cell', 1) in (3, 4, 5, 6)
    gated = cfg.get('cell', 1) in (4, 5, 6)
    selective = cfg.get('cell', 1) in (5, 6)
    associative = cfg.get('cell', 1) == 6
    secondary = adaptive or traced
    raw = np.fromfile(directory / 'weights.f32', dtype='<f4').copy()
    weights = torch.tensor(raw, requires_grad=True)
    offset = 0
    names = []

    def take(name, *shape):
        nonlocal offset
        size = int(np.prod(shape))
        names.append((name, offset, offset + size))
        result = weights[offset:offset + size].view(shape)
        offset += size
        return result

    embedding = take('embedding', 256, c)
    blocks = []
    for i in range(layers):
        block = (take(f'{i}.gain', c), take(f'{i}.input.weight', h, c),
                       take(f'{i}.input.bias', h), take(f'{i}.output.weight', c, h),
                       take(f'{i}.output.bias', c), take(f'{i}.leak', h))
        if secondary:
            label = 'trace' if traced else 'adapt'
            block += (take(f'{i}.{label}_leak', h), take(f'{i}.{label}_scale', h))
        if gated:
            label = 'retention_gate' if selective else 'read_gate'
            block += (take(f'{i}.{label}.weight', h, c), take(f'{i}.{label}.bias', h))
        if associative:
            block += (take(f'{i}.association.projection', 98, h), take(f'{i}.association.bias', 98),
                      take(f'{i}.association.readout', c, 32), take(f'{i}.association.readout_bias', c))
        blocks.append(block)
    final_gain = take('final.gain', c)
    head = take('head.weight', 256, c)
    bias = take('head.bias', 256)
    assert offset == cfg['parameters'] == len(raw)
    tokens = torch.from_numpy(np.fromfile(directory / 'inputs.i32', '<i4').copy()).long().view(batch, time)
    targets = torch.from_numpy(np.fromfile(directory / 'targets.i32', '<i4').copy()).long()
    x = F.embedding(tokens, embedding)

    def norm(v, gain):
        return v * torch.rsqrt((v * v).mean(-1, keepdim=True) + 1e-5) * gain

    activities = []
    initial_path = directory / 'initial_state.f32'
    recurrent = (2 if secondary else 1) * batch * h + (batch * 1024 if associative else 0)
    initial_states = (torch.from_numpy(np.fromfile(initial_path, '<f4').copy()).view(layers, recurrent)
                      if initial_path.exists() else torch.zeros(layers, recurrent))
    final_states = []
    for i, block in enumerate(blocks):
        gain, wi, bi, wo, bo, leak = block[:6]
        normalized = norm(x, gain)
        z = F.linear(normalized, wi, bi)
        gate_logits = F.linear(normalized, block[8], block[9]) if gated else None
        beta = torch.sigmoid(leak)
        # A saved boundary state is a constant for truncated BPTT. Its effect on
        # the first timestep's leak derivative must still be included.
        reset = initial_states[i, :batch*h].view(batch, h)
        if secondary:
            rho = torch.sigmoid(block[6])
            gamma = F.softplus(block[7])
            adaptation = initial_states[i, batch*h:2*batch*h].view(batch, h)
        spikes = []
        emissions = []
        for t in range(time):
            membrane = beta * reset + z[:, t, :]
            threshold = 1 + gamma * adaptation if adaptive else 1
            spike = AdaptiveSignedSpike.apply(membrane, threshold) if adaptive else SignedSpike.apply(membrane)
            spikes.append(spike)
            # Deliberate detached reset, matching the documented training rule.
            reset = membrane - (threshold * spike).detach()
            if adaptive:
                adaptation = rho * adaptation + (1 - rho) * spike.abs()
            if traced:
                retention = torch.sigmoid(block[6] + gate_logits[:, t, :]) if selective else rho
                adaptation = retention * adaptation + (1 - retention) * spike
                read = 2 * torch.sigmoid(gate_logits[:, t, :]) if gated and not selective else 1
                emissions.append(spike + gamma * read * adaptation)
        final_states.append(reset.detach().reshape(-1))
        if secondary:
            final_states.append(adaptation.detach().reshape(-1))
        all_spikes = torch.stack(spikes, dim=1)
        activities.append(all_spikes.abs().mean())
        emission = torch.stack(emissions, dim=1) if traced else all_spikes
        x = x + F.linear(emission, wo, bo)
        if associative:
            memory = initial_states[i, 2*batch*h:].view(batch, 32, 32)
            output, memory = associative_forward(emission, *block[10:14], initial=memory)
            x = x + output
            final_states.append(memory.detach().reshape(-1))
    logits = F.linear(norm(x, final_gain), head, bias)
    target_weights_path = directory / 'target_weights.f32'
    if target_weights_path.exists():
        target_weights = torch.from_numpy(np.fromfile(target_weights_path, '<f4').copy())
        token_loss = F.cross_entropy(logits.reshape(-1, 256), targets, reduction='none')
        loss = (token_loss * target_weights).sum() / target_weights.sum()
    else:
        loss = F.cross_entropy(logits.reshape(-1, 256), targets)
    if initial_path.exists():
        native_state = np.fromfile(directory / 'final_state.f32', '<f4')
        assert np.allclose(native_state, torch.cat(final_states).numpy(), atol=2e-5, rtol=1e-5)
    loss.backward(retain_graph=True)
    actual_logits = np.fromfile(directory / 'logits.f32', '<f4')
    expected_logits = logits.detach().numpy().reshape(-1)
    assert np.max(np.abs(actual_logits - expected_logits)) < 2e-5
    assert abs(loss.item() - cfg['loss']) < 2e-6
    actual_grad = np.fromfile(directory / 'gradients.f32', '<f4')
    expected_grad = weights.grad.numpy()
    tensor_results = []
    for name, start, end in names:
        a, b = actual_grad[start:end], expected_grad[start:end]
        err = float(np.max(np.abs(a - b)))
        tolerance = 1e-8 if ('.adapt_' in name or '.trace_' in name or '_gate.' in name or '.association.' in name) else 2e-6
        assert np.allclose(a, b, atol=tolerance, rtol=5e-4), (name, err)
        if '.adapt_' in name or '.trace_' in name or '_gate.' in name or '.association.' in name:
            assert float(np.linalg.norm(b)) > 1e-9, (name, 'fixture must exercise adaptation gradients')
        tensor_results.append({'tensor': name, 'max_abs_gradient_error': err})
    # The native test performs its first unclipped, zero-decay Adam update.
    g = weights.grad.detach()
    expected_update = weights.detach() - 0.001 * g / (g.abs() + 1e-8)
    actual_update = np.fromfile(directory / 'updated.f32', '<f4')
    update_error = float(np.max(np.abs(actual_update - expected_update.numpy())))
    assert update_error < 5e-6, update_error
    weights.grad = None
    (loss + torch.stack(activities).mean()).backward()
    regularized = np.fromfile(directory / 'gradients_regularized.f32', '<f4')
    regularized_error = float(np.max(np.abs(regularized - weights.grad.numpy())))
    assert np.allclose(regularized, weights.grad.numpy(), atol=3e-6, rtol=5e-4), regularized_error
    result = {'passed': True, 'reference': 'independent CPU PyTorch autograd',
              'nonzero_boundary_state': initial_path.exists(), 'weighted_targets': target_weights_path.exists(),
              'cell': cfg.get('cell', 1),
              'loss': loss.item(), 'logits_max_abs_error': float(np.max(np.abs(actual_logits - expected_logits))),
              'gradients_max_abs_error': float(np.max(np.abs(actual_grad - expected_grad))),
              'adam_update_max_abs_error': update_error, 'regularized_gradient_max_abs_error': regularized_error,
              'tensors': tensor_results}
    (directory / 'oracle.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'tensors'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    check(parser.parse_args().directory)
