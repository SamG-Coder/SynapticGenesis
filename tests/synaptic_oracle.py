"""Independent scalar reference for consolidation, clipping and an AdamW update."""
import argparse
from array import array
import json
import math
from pathlib import Path


def check(directory):
    def read(name):
        result = array('f')
        result.frombytes((directory / (name + '.f32')).read_bytes())
        return list(result)

    cfg = json.loads((directory / 'fixture.json').read_text())
    weights, gradient = read('weights'), read('task-gradient')
    moment, variance, mask = read('moment'), read('variance'), read('decay-mask')
    state = read('synapses-before')
    n = len(weights)
    importance, path, reference = state[:n], state[n:2*n], state[2*n:]
    total = [g + 2*cfg['strength']*o*(w-r)
             for g, o, w, r in zip(gradient, importance, weights, reference)]
    norm = math.sqrt(sum(g*g for g in total))
    scale = min(1, cfg['clip']/norm)
    clipped = [g*scale for g in total]
    new_m = [.9*m + .1*g for m, g in zip(moment, clipped)]
    new_v = [.95*v + .05*g*g for v, g in zip(variance, clipped)]
    b1, b2 = 1-.9**cfg['step'], 1-.95**cfg['step']
    expected_w = []
    for i, (w, m, v, decay) in enumerate(zip(weights, new_m, new_v, mask)):
        rate = cfg['lr'] * (cfg['core_scale'] if i < cfg['core_end'] else 1)
        expected_w.append(w-rate*(m/b1/(math.sqrt(v/b2)+1e-8)+cfg['wd']*decay*w))
    actual_w = read('updated')
    # Utility follows the actual representable float weight displacement.
    expected_path = [p-g*(a-w) for p, g, a, w in zip(path, gradient, actual_w, weights)]
    consolidated = [o+max(0, p)/((w-r)**2+cfg['damping'])
                    for o, p, w, r in zip(importance, expected_path, actual_w, reference)]
    expectations = {
        'updated': expected_w, 'moment-after': new_m, 'variance-after': new_v,
        'clipped-gradient': clipped,
        'synapses-after-update': importance + expected_path + reference,
        'synapses-after-boundary': consolidated + [0.]*n + actual_w,
    }
    errors = {}
    for name, expected in expectations.items():
        actual = read(name)
        assert len(actual) == len(expected)
        error = max(abs(a-b) for a, b in zip(actual, expected))
        tolerance = 2e-7 if name == 'synapses-after-boundary' else 5e-7
        assert error < tolerance, (name, error)
        errors[name] = error
    assert abs(norm-cfg['gradient_norm']) < 2e-6
    assert any(v > 0 for v in path) and any(v < 0 for v in path)
    assert scale < 1, 'Fixture must exercise global gradient clipping'
    result = {'passed': True, 'reference': 'independent scalar SI and AdamW equations',
              'gradient_norm_error': abs(norm-cfg['gradient_norm']), 'max_abs_errors': errors}
    (directory / 'oracle.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    check(parser.parse_args().directory)
