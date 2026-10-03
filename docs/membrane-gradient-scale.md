# Membrane correction strength changes with learning history

The proposed direct membrane cost can supply a gradient where the local
spike-activity penalty has none. That does not make its strength automatically
appropriate. A read-only CPU analysis of the existing native traces now
measures part of each objective's gradient at initialization, after the first
prose stage, and after the complete curriculum.

For the earlier 105M founder, the median measured membrane-gradient norm grows
from **0.236 to 44.99**. At coefficient `0.02`, the final-layer coordinates
alone exceed the saved global clipping threshold in 10 of the 64 final-stage
intervals. This is a result for the proposed objective by itself, not evidence
that a combined language/regularizer update clips or that a particular
coefficient improves learning.

## What is measured

The [declared policy](../data/membrane-gradient-audit-v1.json) reuses all nine
cases and all eight 1,024-byte windows from the preserved
[native trace panel](prose-spike-results.md). Each window is divided into
eight 128-byte backward intervals, giving 576 analyzed intervals. There is
no selection of favorable windows or new source material.

Only the final block's input matrix, input bias and learned membrane-leak
logit are differentiated. Earlier layers' own membrane costs cannot depend
on these later parameters. For an objective summed across layers, these
coordinates are consequently an exact subset of its mathematical gradient
at the frozen values. The Euclidean norm of that subset is a **lower bound
on the full regularizer gradient norm**. It is not an estimate of all the
other coordinates.

For a single native batch item, the direct-cost derivative is computed as:

```text
local[t] = sign(u[t]) * max(abs(u[t]) - 1.5, 0) / (T * H * L)
d[t]     = local[t] + beta * d[t+1]
dW       = sum_t outer(d[t], normalized_input[t])
db       = sum_t d[t]
dleak    = beta * (1-beta) * sum_t d[t] * previous_reset[t]
```

`T=128`; `H` and `L` are the actual hidden width and layer count. Backward
carry ends at each interval boundary. Its incoming reset state is retained
as a constant, including its effect on the first timestep's leak derivative.
Spike reset decisions are detached, matching the declared native training
rule. The spike-activity comparison uses `spike * surrogate(u)` for its local
term and the same normalization and recurrence.

The calculation uses double precision on saved native FP32 values. It does
not reproduce CUDA reduction order bit-for-bit. The traces came from a
reset-state, 1,024-byte diagnostic forward; partitioning them does not make
them the actual live training chunks. Every saved checkpoint has activity
cost zero and was trained without the new membrane objective. Both reported
regularizer gradients are therefore counterfactual at those states.

## Observed scale

These are medians over the 64 intervals per size/stage at unit coefficient,
including only the final-block coordinates described above:

| Founder | Initial membrane bound | Stage-one membrane bound | Final membrane bound | Final spike-activity bound |
| --- | ---: | ---: | ---: | ---: |
| 2M | 0.6174 | 1.6613 | 21.5212 | 0.002727 |
| 27M | 0.4965 | 7.8244 | 28.3366 | 0.000771 |
| 105M | 0.2358 | 24.1155 | 44.9920 | 0.000519 |

All saved clipping thresholds are 1. At membrane coefficient `0.02`, these
final-stage intervals already have a subset norm greater than that threshold:

| Founder | Intervals above 1 | Maximum subset norm at `0.02` |
| --- | ---: | ---: |
| 2M | 6 / 64 | 1.7667 |
| 27M | 3 / 64 | 1.2316 |
| 105M | 10 / 64 | 1.6182 |

The [complete result](../reports/membrane-gradient-audit.json) preserves every
interval, separate input-matrix/bias/leak norms, min/median/p90/max summaries,
and six illustrative strengths from `0.00001` through `0.1`. A small subset
norm cannot establish that the complete gradient is small. The combined
language and regularizer gradient can also reinforce or cancel, so these
counts are not predicted clipping counts for actual learning.

These earlier founders used the smaller replay reservoir. The analysis is
not a measurement of the active 411M learner, its newer replay policy, AdamW
displacements, language quality, or biological development. A model trained
with the penalty from initialization may develop quite different states.

## Verification and use in the next experiment

`scripts/membrane_gradient.py` owns the analytical calculation.
`scripts/membrane_gradient_audit.py` authenticates the published panel, all
nine checkpoints, 216 final-layer trace arrays and eight source windows. It
reads only the required checkpoint leak vectors and hashes large files by
streaming. It executes no native model command and writes no checkpoint.

The [CPU verification](../reports/membrane-gradient-checks.json) passes eight
independent PyTorch autograd cases, including 128-step recurrence and nonzero
incoming state, with maximum absolute gradient error `3.33e-16`. Five
finite-difference checks with reset spikes held fixed have maximum error
`9.25e-12`. Separate controls cover fully saturated spikes, cancellation by
an opposing gradient, nine invalid inputs, and the actual leak-vector layout
against the existing independent checkpoint reference.

```powershell
$env:OPENBLAS_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
python tests/membrane_gradient.py --out runs/membrane-gradient-checks-new.json
python scripts/membrane_gradient_audit.py --out runs/membrane-gradient-audit-new.json
```

Use fresh output files. These diagnostics ran on CPU while the separate 411M
study continued; they are not an isolated throughput benchmark. The queued
candidate CUDA acceptance checks remain the gate for a native learning trial.

The result supports a controlled coefficient comparison with fresh learners
and an unchanged control, followed by explicit continuation/retention checks.
Record the full combined gradient, clipping frequency, loss and raw output;
reduced firing alone is insufficient. The illustrative strengths in this
audit are not selected settings, and no model is admitted for teaching or
reproduction by these measurements.
