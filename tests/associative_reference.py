"""Independent differentiable CPU matrix equations, not a runtime backend."""
import torch
import torch.nn.functional as F


def forward(emission, projection, bias, readout, readout_bias, initial=None, mode='normal'):
    if mode not in ('normal', 'discard_history', 'zero_read'):
        raise ValueError('Unknown associative diagnostic mode')
    # [batch, time, hidden] -> normalized keys/queries and scalar write/decay.
    packed = F.linear(emission, projection, bias)
    q, k, v = packed[..., :32], packed[..., 32:64], packed[..., 64:96].tanh()
    q = q / torch.sqrt(q.square().sum(-1, keepdim=True) + 1e-5)
    k = k / torch.sqrt(k.square().sum(-1, keepdim=True) + 1e-5)
    a, b = packed[..., 96].sigmoid(), packed[..., 97].sigmoid()
    memory = (torch.zeros(emission.shape[0], 32, 32, dtype=emission.dtype)
              if initial is None else initial)
    reads = []
    for t in range(emission.shape[1]):
        prior = a[:, t, None, None] * (torch.zeros_like(memory) if mode == 'discard_history' else memory)
        residual = v[:, t] - (prior @ k[:, t, :, None]).squeeze(-1)
        memory = prior + b[:, t, None, None] * residual[:, :, None] * k[:, t, None, :]
        read = (memory @ q[:, t, :, None]).squeeze(-1)
        reads.append(torch.zeros_like(read) if mode == 'zero_read' else read)
    return F.linear(torch.stack(reads, dim=1), readout, readout_bias), memory
