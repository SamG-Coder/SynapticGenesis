"""Compare native neuron traces with a read-only CPU spike intervention."""
import numpy as np
import torch


def compare(reference, raw, directory, answer_bytes=4):
    traces = []
    logits = reference.logits(raw[:-1], traces=traces)
    native, first = [], None
    for layer, trace in enumerate(traces):
        arrays = {name: np.fromfile(directory / f'layer-{layer}-{name}.f32', dtype='<f4')
                  .reshape(len(raw) - 1, reference.h) for name in ('u', 'spikes')}
        assert all(np.isfinite(value).all() for value in arrays.values())
        assert np.array_equal(arrays['spikes'],
                              (arrays['u'] >= 1).astype('float32') - (arrays['u'] <= -1).astype('float32'))
        native.append(arrays)
        differences = np.argwhere(trace['spikes'].numpy() != arrays['spikes'])
        if len(differences) and first is None:
            t, h = map(int, differences[0])
            first = dict(layer_zero_based=layer, byte_zero_based=t, neuron_zero_based=h,
                         cpu_membrane=float(trace['u'][t, h]), native_membrane=float(arrays['u'][t, h]),
                         cpu_spike=float(trace['spikes'][t, h]), native_spike=float(arrays['spikes'][t, h]))
    targets = torch.tensor(list(raw[1:]), dtype=torch.long)

    def answer_nll(values):
        return -values.log_softmax(-1).gather(1, targets[:, None]).flatten()[-answer_bytes:].double().sum().item()

    dumped_nll = float(np.fromfile(directory / 'losses.f32', dtype='<f4')[-answer_bytes:].astype('float64').sum())
    native_logits = torch.from_numpy(np.fromfile(directory / 'logits.f32', dtype='<f4').reshape(len(raw) - 1, 256))
    result = dict(first_spike_divergence=first, native_nll=dumped_nll, cpu_nll=answer_nll(logits))
    if first is None:
        return result
    layer, t, h = (first[k] for k in ('layer_zero_based', 'byte_zero_based', 'neuron_zero_based'))
    result['first_difference_within_1e_minus_5_of_threshold'] = (
        abs(first['cpu_membrane'] - first['native_membrane']) < 1e-5 and
        max(abs(abs(first[k]) - 1) for k in ('cpu_membrane', 'native_membrane')) < 1e-5)
    override = [torch.full((len(raw) - 1, reference.h), float('nan')) for _ in reference.blocks]
    override[layer][t, h] = first['native_spike']
    controlled_traces = []
    controlled = reference.logits(raw[:-1], traces=controlled_traces, forced_spikes=override)
    result.update(single_spike_override_nll=answer_nll(controlled),
                  single_spike_override_nll_error=abs(answer_nll(controlled) - dumped_nll),
                  single_spike_override_max_logit_error=float((controlled - native_logits).abs().max()),
                  single_spike_override_matches_all_native_spikes=all(
                      np.array_equal(trace['spikes'].numpy(), n['spikes'])
                      for trace, n in zip(controlled_traces, native)))
    return result
