# Projection strength across learned ages

CPU inspection of eight immutable checkpoints found that the larger founder's
effective input projections grew more than the smaller founder's under the
common learning setting. This is a parameter measurement after observing the
[105M regression](prose-105m-complete.md), not a diagnosis of its cause or an
intervention. The [full record](../reports/prose-projection-snapshots.json)
preserves every layer, checkpoint identity and original validation mean.

For each neuron, the diagnostic measures the Euclidean norm of its input-weight
row after multiplication by the learned channel gains. In the native forward,
the residual is divided by its RMS, multiplied by those gains, then projected
into the spiking neurons. Combining weights and gains avoids attributing a
change to either tensor in isolation.

Let `y` be the residual after RMS division but before the learned gain `g`.
The projection is `z = W diag(g) y + b`. The row norm measures sensitivity of
one component of `z` to changes in `y`: a perturbation with Euclidean length
one can change that component by at most that row norm. This does not include
the derivative through RMS division and is not a complete network Jacobian.
It does not predict the observed distribution of `z` without information about
the direction and covariance of actual inputs.

| Model | Source observations | Range of per-layer median effective input row norms | Four-book validation mean, nats/byte |
| --- | ---: | ---: | ---: |
| 2M | 0 | 0.997–1.002 | Not assessed |
| 2M | 8,176 | 0.892–1.061 | 2.200162 |
| 2M | 95,207 | 1.344–1.732 | 1.769407 |
| 2M | 216,289 | 1.980–2.278 | 1.800033 |
| 105M | 0 | 0.999–1.000 | Not assessed |
| 105M | 8,176 | 1.307–1.546 | 2.192909 |
| 105M | 95,207 | 2.298–3.061 | 1.804779 |
| 105M | 216,289 | 2.429–3.517 | 1.967449 |

These ranges span **layer medians**, not every neuron and not confidence
intervals. The 2M model has four layers; the 105M model has eight. Their widths
and hidden/channel ratios also differ, so the comparison does not isolate
width alone. Source, replay and scheduled-generation counts match between the
completed founders at the assessed endpoints.

The final 105M model's gain RMS ranges from 0.731 to 1.031 across layers. Its
larger effective input norms therefore cannot be described simply as all
normalization gains becoming large. Its gate-row median norms range from
2.183 to 3.524 and its spiking output-projection column medians from 2.004 to
4.115. Those are separate sensitivity measurements; neither establishes how
much the corresponding path contributes to predictions.

Learned leak parameters also change differently across the models. Final
per-layer median leak coefficients range from 0.456 to 0.598 in the small model
and 0.667 to 0.750 in the large model, compared with 0.745 at initialization.
These coefficients are not measured memory duration: inputs, resets, gates,
continuous traces and the associative path also affect behavior.

Growth in projection norms is not automatically harmful. Both models improve
their validation mean substantially between initialization-era learning and
stage three while these norms grow. Both then worsen on the validation mean
during stage four. These observations cannot distinguish useful learned
features from unstable scaling, distribution shift or interference.

## Checks and next consequence

The [CPU verification](../reports/prose-projection-validation.json) compared
all 112 layer-tensor views from the initial and final 2M checkpoints with the
existing independent learned-model oracle. Every view matched exactly. Scalar
dot-product calculations independently reproduced input-row norms, and seven
malformed shape/value cases were rejected. This checks the reader and
statistics using existing native artifacts; no new CUDA forward was run.

The inspection memory-maps only the parameter section, processes one layer at
a time, and verifies checkpoint hashes before and after reading. The report
contains 48 layer records across eight checkpoints. It changes no weights,
moments, recurrence or training data. The 27M learner was running concurrently;
this stage makes no runtime-speed claim. Its CPU reader does not replace the
native loader's full checksum and layout checks.

The next measurements remain the declared [matched native traces](prose-spike-panel.md)
and the [late-stage learning-rate control](https://github.com/SamG-Coder/SynapticGenesis/blob/research/learning-rate-retention/docs/prose-retention-lr.md).
Matched traces can show whether learned projections are accompanied by larger
actual drive, constant signed firing or narrower local surrogate support on
the same text. The learning-rate comparison can test a change in retention
without changing neuron equations. Neither result exists yet in this report.

If a subsequent scaling intervention is warranted, it should preserve the
original checkpoints and declare its effect on input projections, output
projections, scalar dynamics and optimizer state. The existing
[width review](width-and-learning-scale.md) explains why changing one global
rate is not a complete width-dependent parametrization. This snapshot does
not justify clipping or rescaling trained weights in place.
