"""Reproduce the two-coordinate gradient/Adam geometry counterexample.

This is float64 arithmetic on linear losses, not native execution or a learner.
"""
import argparse
import math
from pathlib import Path

from native_experiment import sha, write


def example():
    reference, gradient = [.2, 1.], [2., -1.]
    dot = lambda a, b: sum(x*y for x, y in zip(a, b))
    conflict = dot(gradient, reference)
    projected = [g - conflict*r/dot(reference, reference)
                 for g, r in zip(gradient, reference)]
    norm = math.sqrt(dot(projected, projected))
    clipped = [g / max(1., norm) for g in projected]
    lr, beta1, beta2, epsilon = .001, .9, .95, 1e-8
    sgd = [-lr*g for g in clipped]
    moments = [(1-beta1)*g for g in clipped]
    variances = [(1-beta2)*g*g for g in clipped]
    adam = [-lr*(m/(1-beta1))/(math.sqrt(v/(1-beta2))+epsilon)
            for m, v in zip(moments, variances)]
    assert conflict < 0 and abs(dot(reference, projected)) < 1e-15
    assert abs(dot(reference, clipped)) < 1e-15
    assert abs(dot(reference, sgd)) < 1e-15
    assert dot(reference, adam) > .00079
    return dict(kind='Illustrative float64 optimizer-geometry counterexample; not a model experiment',
        reference_gradient=reference, new_gradient=gradient, projected_gradient=projected,
        clipped_projected_gradient=clipped, projection_dot=dot(reference, projected),
        clipped_dot=dot(reference, clipped), initial_moments='zero', learning_rate=lr,
        beta1=beta1, beta2=beta2, epsilon=epsilon, global_gradient_norm_clip=1.,
        weight_decay=0., core_scale=1., optimizer_step=1, sgd_delta=sgd, adam_delta=adam,
        old_linear_loss_change_sgd=dot(reference, sgd),
        old_linear_loss_change_adam=dot(reference, adam),
        source_sha256={str(Path(__file__).relative_to(Path.cwd())): sha(__file__),
                       'src/spike_lm.cu': sha('src/spike_lm.cu')},
        claim='A raw-gradient projection constraint need not survive Adam coordinate normalization.',
        limits=['Analytic linear loss with two parameters, not the production spiking model.',
                'This example does not reproduce the cited A-GEM implementation.',
                'It does not change the status of existing independent numerical failures.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = example()
    write(args.out, result)
    print('Old linear loss change after projected gradient:',
          'SGD', result['old_linear_loss_change_sgd'],
          'Adam', result['old_linear_loss_change_adam'])
