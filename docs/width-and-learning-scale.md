# Width, initialization and live learning

The larger founder's membrane snapshots motivate checking update scaling as
well as activity regulation. This review does not identify the cause of those
large values and does not change the active size comparison. Its source and
inspection boundaries are recorded in the
[research ledger](../reports/width-and-learning-scale.json).

Yang and colleagues show that conventional initialization alone does not make
optimal learning settings stable across widths in their MLP and transformer
experiments. Their maximal update parametrization changes initialization,
parameter multipliers and parameter-specific learning rates together. Rescaling
one global learning rate is not the complete method. They also report limits
to transfer across depth and regularization settings. The relevant reading here
covered the introduction, sections 2-4, Table 3 and the caveats in section 6.1;
the full mathematical derivation was not audited.
[Tensor Programs V, arXiv v2](https://arxiv.org/pdf/2203.03466v2).

The authors' implementation documentation suggests checking activation
magnitudes at initialization and after a few updates across widths. It also
distinguishes tensor dimensions that grow from fixed dimensions and warns
against rescaling an already trained checkpoint as though it were a fresh
initialization. We can adopt that diagnostic question in native CUDA without
importing their PyTorch package. Passing a few magnitude checks would still
not establish useful language learning.
[Authors' documentation](https://github.com/microsoft/mup#checking-correctness-of-parametrization).

Shi and Yu's SpikeInit work separately studies initialization of spiking
weights and surrogate-gradient shape to stabilize forward firing and backward
signal magnitudes. For this review, only the publisher abstract and authors'
README were accessible; the linked publisher PDF failed to load and OpenReview
returned a browser-verification page. No detailed theorem or experimental
replication was verified. It remains a research lead, not an adopted
initialization or a justification for removing our normalization.
[Publisher](https://proceedings.mlr.press/v306/shi26ae.html),
[authors' README](https://github.com/xyshi2000/SpikeInit).

## What the current native code actually does

Inspection of `initialize` in [spike_lm.cu](../src/spike_lm.cu) gives these
initial standard deviations, with channel width `C`, spiking width `H` and
layer count `L`:

| Parameter family | Current initialization |
| --- | --- |
| Byte embeddings | `0.1` |
| Input projection into spiking neurons | `1 / sqrt(C)` |
| Spike-output projection into residual channels | `0.1 / sqrt(H * L)` |
| Vocabulary head | `0.1 / sqrt(C)` |
| Associative query/key/value rows | `1 / sqrt(H)` |
| Retention-gate weights and associative output projection | Zero |

The Adam kernel applies one rate to all parameters before `core_end`, multiplied
by `core_scale`, and the unscaled rate to the final normalization gain and
vocabulary output group. The
current founders use `core_scale=1`. There is no tensor-role-specific width
multiplier in this optimizer. These facts do not establish that the existing
optimizer is incorrect, but the code is not an implementation of the reviewed
maximal-update prescription.

RMS normalization occurs before each input projection. The membrane and
emitted signal are not subsequently normalized to a common scale before the
trace dynamics and output projection. The signed threshold remains one, while
leak, retention gates and continuous trace gain are learned. The auxiliary
associative matrix stays 32 by 32 as the other widths grow. A derivation for
ordinary feed-forward activations cannot be assumed to cover this signed
recurrence and its surrogate backward rule.

The current 2M-to-27M comparison changes both depth and the `H/C` ratio:
`C256/H512/L4` versus `C512/H2048/L8`. The 27M-to-105M pair keeps depth eight
and ratio four, doubling both widths. Thus the full three-size experiment is
a capacity comparison, not a pure width-scaling experiment. Its common learning
rate is one declared control, not proof that every size is equally well tuned.

## Consequence for the next experiment

Finish the declared quality and neuron-response panels first. If scale-dependent
drive growth remains a plausible issue, measure a fixed-depth, fixed-ratio
family such as `C256/H1024/L8`, `C512/H2048/L8` and `C1024/H4096/L8` on identical
selected windows. Record per-layer drive, membrane and emission magnitudes at
initialization and early updates, together with actual parameter displacement,
loss, gradient clipping and elapsed time. Separate random seeds from a change
in width. This is a proposed diagnostic, not an additional queued run.

Any subsequent update-scaling candidate needs an explicit mapping for input,
hidden and output projections, scalar neuron parameters, normalization gains,
and the fixed-width associative path. Its native shared forward, checkpoint
resume and child-growth semantics must remain explicit. Test it against the
unchanged implementation and a simple learning-rate control. An early scale
check can screen obvious instability; longer new-source learning and retained
reading performance must decide whether a candidate helps.

Neither a small local surrogate coefficient nor a large weight norm by itself
measures irreversible ageing. The [matched trace panel](prose-spike-panel.md)
will separate constant firing from changing signed responses, and the
[quality comparison](prose-evaluation.md) will supply the separate outcome
measurements. No paper, external code or model weights from this review were
added to the training selection.
