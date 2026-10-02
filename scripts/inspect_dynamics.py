"""Read-only research diagnostics for signed LIF and adaptive LIF checkpoints.

Reports passive leak time constants, not measured memory or task performance.
Does not execute or train a model. Uses only the Python standard library.
"""
import argparse
from array import array
import hashlib
import json
import math
from pathlib import Path
import struct
import sys


def summary(values):
    values = sorted(values)

    def percentile(q):
        position = (len(values) - 1) * q
        lo = int(position)
        hi = min(lo + 1, len(values) - 1)
        return values[lo] + (values[hi] - values[lo]) * (position - lo)

    return dict(zip(('min', 'p50', 'p90', 'p99', 'max'),
                    [percentile(q) for q in (0, .5, .9, .99, 1)]))


def inspect(path):
    path = Path(path)
    with path.open('rb') as f:
        meta = struct.unpack('<32Q', f.read(256))
        f.read(32)
        if meta[0] != 0x314D4C53434E5042 or meta[1] not in (1, 2):
            raise ValueError('Unsupported checkpoint format')
        adaptive = meta[1] == 2
        c, h, layers, batch = meta[2:6]
        if not (8 <= c <= 2048 and 8 <= h <= 8192 and 1 <= layers <= 32 and 1 <= batch <= 256):
            raise ValueError('Invalid model dimensions')
        expected = 256*c + layers*(c + 2*c*h + 2*h + c + (2*h if adaptive else 0)) + c + 256*c + 256
        if meta[17] > 3 or (meta[17] < 2 and meta[31]) or meta[31] > 16 + 3*65536:
            raise ValueError('Unsupported live state extension')
        synaptic_bytes = 12*expected if meta[17] == 3 else 0
        if meta[14] != expected or path.stat().st_size != 288 + 12*expected + 4*meta[18] + 8*meta[31] + synaptic_bytes:
            raise ValueError('Checkpoint layout/length mismatch')
        weights = array('f')
        weights.fromfile(f, expected)
        if sys.byteorder != 'little':
            weights.byteswap()
        offset = 256*c
        blocks = []
        all_taus = []
        for layer in range(layers):
            offset += c + h*c + h + c*h + c
            leaks = weights[offset:offset+h]
            offset += h
            # log(sigmoid(leak)) = -softplus(-leak), evaluated stably.
            log_decay = [max(-v, 0) + math.log1p(math.exp(-abs(v))) for v in leaks]
            taus = [1/v for v in log_decay]
            all_taus.extend(taus)
            blocks.append({'layer': layer, 'beta': summary([math.exp(-v) for v in log_decay]),
                           'passive_e_folding_steps': summary(taus)})
            if adaptive:
                adapt_leaks = weights[offset:offset+h]
                scales = weights[offset+h:offset+2*h]
                offset += 2*h
                adapt_log_decay = [max(-v, 0) + math.log1p(math.exp(-abs(v))) for v in adapt_leaks]
                blocks[-1]['adaptation_e_folding_steps'] = summary([1/v for v in adapt_log_decay])
                blocks[-1]['adaptation_threshold_strength'] = summary([max(v, 0)+math.log1p(math.exp(-abs(v))) for v in scales])
        result = {'checkpoint': path.as_posix(), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                  'cell_version': meta[1], 'step': meta[7], 'parameters': expected, 'blocks': blocks,
                  'passive_e_folding_steps': summary(all_taus),
                  'fraction_with_tau_at_least_128_steps': sum(v >= 128 for v in all_taus)/len(all_taus),
                  'recurrent_state_bytes': batch*layers*h*4*(2 if adaptive else 1),
                  'one_f32_input_weight_eligibility_trace_bytes': layers*c*h*4}
        if meta[17]:
            f.seek(288 + 12*expected)
            states = array('f')
            states.fromfile(f, meta[18])
            if sys.byteorder != 'little':
                states.byteswap()
            if adaptive:
                stride = batch*h
                membranes = [v for layer in range(layers) for v in states[layer*2*stride:layer*2*stride+stride]]
                adaptation = [v for layer in range(layers) for v in states[layer*2*stride+stride:(layer+1)*2*stride]]
                result['saved_adaptation_state'] = summary(adaptation)
            else:
                membranes = states
            result['saved_reset_membrane_absolute_value'] = summary([abs(v) for v in membranes])
            result['saved_reset_membrane_abs_above_one_fraction'] = sum(abs(v) > 1 for v in membranes)/len(membranes)
        if meta[17] == 3:
            extra = struct.unpack(f'<{meta[31]}Q', f.read(8*meta[31]))
            synapses = array('f')
            synapses.fromfile(f, 3*expected)
            if sys.byteorder != 'little':
                synapses.byteswap()
            result['synaptic_memory'] = {
                'boundaries': extra[12], 'observed_optimizer_updates': extra[13],
                'strength': struct.unpack('<f', struct.pack('<I', extra[10]))[0],
                'damping': struct.unpack('<f', struct.pack('<I', extra[11]))[0],
                'durable_gpu_bytes': synaptic_bytes, 'scratch_gpu_bytes': 4*expected,
                'importance': summary(synapses[:expected]),
                'positive_importance_fraction': sum(v > 0 for v in synapses[:expected])/expected,
                'current_trajectory_estimate': summary(synapses[expected:2*expected]),
            }
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoints', nargs='+')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = {'scope': 'Read-only parameter diagnostics; passive decay ignores input, spikes, reset and layer interactions. '
                       'One step is one byte, not a biological millisecond. A single saved state is not a firing-rate sample.',
              'checkpoints': [inspect(path) for path in args.checkpoints]}
    Path(args.output).write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    for entry in result['checkpoints']:
        print(entry['checkpoint'], json.dumps(entry['passive_e_folding_steps']))
