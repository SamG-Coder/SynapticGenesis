# Neuron responses after the complete prose curriculum

The native matched-window diagnostic completed all 73 commands: one preserved
trace compatibility check and 72 new forwards over nine checkpoints. The
larger models develop much denser firing and many more zero local spike
derivatives than the small control under their shared learning policy. These
measurements identify a learning-regime problem to investigate; they do not
establish the cause of the language-quality regressions or a successful repair.

The [full report](../reports/prose-spike-panel.json) preserves per-window and
pooled measurements. The [publication audit](../reports/prose-spike-panel-audit.json)
verified all 3,050 raw trace files, their lengths and hashes, the exact native
command arguments and successful output logs, and the nine checkpoint hashes.
The [predeclared protocol](prose-spike-panel.md) fixes the eight source windows,
three model sizes and three saved ages.

## Measured response regime

Each checkpoint receives the same eight independent 1,024-target-byte windows
from selected training books, using the production forward computation in
strict FP32 with zero initial recurrence. Each layer is summarized over 8,192
byte positions. The ranges below run from the lowest to highest layer value
within a model, not confidence intervals or ranges across seeds.

An event is a nonzero signed spike. A zero local derivative means the native
triangular surrogate coefficient is zero at that neuron and byte position.
It does not mean that fraction of neurons is permanently dead, or that the
full gradient through the network is zero.

| Model | Saved age | Nonzero spikes, layer range | Zero local spike derivative, layer range | Mean source-window loss, nats/byte |
| --- | --- | ---: | ---: | ---: |
| 2M | Initial | 38.70-41.85% | 8.43-11.03% | 5.542130 |
| 2M | Stage 1 | 32.70-52.93% | 14.06-25.00% | 1.989002 |
| 2M | Stage 4 | 81.27-87.93% | 64.04-76.00% | 1.511372 |
| 27M | Initial | 38.05-42.23% | 8.12-11.73% | 5.535918 |
| 27M | Stage 1 | 55.44-85.39% | 43.44-72.38% | 1.938328 |
| 27M | Stage 4 | 89.21-94.60% | 79.06-89.21% | 1.613328 |
| 105M | Initial | 38.03-41.83% | 8.22-11.23% | 5.553249 |
| 105M | Stage 1 | 89.22-97.51% | 79.29-95.03% | 1.920790 |
| 105M | Stage 4 | 94.40-98.78% | 88.85-97.57% | 1.538728 |

The initial response ranges are similar across sizes. By the first saved
learning checkpoint, the 105M model already has nearly continuous firing in
some layers. The final 2M model also develops denser responses, but its local
surrogate remains nonzero more often than in either larger model.

Source-window loss improves alongside this increased saturation. Thus the
diagnostic does not show that learning has stopped, and firing density alone
does not rank language quality. The independent [validation-book comparison](prose-size-results.md)
shows a final-stage regression, with 2M finishing better than both larger
models. Its 128-byte validation windows and these 1,024-byte training-source
windows are different measurements and their losses must not be compared as
if they were the same benchmark.

## Example: the fifth layer of the final 105M model

The human-numbered fifth layer is `layer_zero_based: 4` in the report. It has
4,096 neurons and 33,554,432 measured neuron-position responses.

| Measurement | Value |
| --- | ---: |
| Nonzero spike fraction | 98.78% |
| Zero local spike derivative fraction | 97.57% |
| Adjacent signed-spike change fraction within windows | 4.52% |
| Direct positive/negative sign flip fraction within windows | 2.39% |
| Mean per-neuron marginal spike entropy | 0.393857 bits |
| Median absolute pre-reset membrane | 46.662964 |
| Median absolute input drive | 12.654481 |
| Fixed positive/negative firing threshold magnitude | 1 |

Nearly every response fires, but most adjacent responses retain their sign.
The membrane and drive are large relative to the fixed threshold. This agrees
with the earlier [saved-state observation](prose-spike-state.md), now measured
on the same text across model sizes and ages rather than different saved
stream positions.

This layer has no neuron whose signed spike is constant over all eight windows.
The final 105M model does have constant signed spikes in 1.17% of layer seven's
neurons and 0.24% of layer eight's neurons over the pooled observation. It has
no near-constant pooled emitted-signal neuron under the declared standard
deviation threshold of 1e-6. Pooling includes variation between reset windows;
these counts do not prove that every neuron contributes useful information
or that every neuron varies within every individual window.

## Verification and boundaries

The rebuilt diagnostic reproduced all 26 files of the preserved 256-byte trace
exactly. All nine selected checkpoints remained unchanged. The independent
stable-softmax reconstruction of the 72 windows' native target losses has
maximum absolute error 1.949443e-6 nats, below the predeclared 3e-5 threshold.
Every command's log confirms the expected byte and layer count. No model
update, source admission, reserved-test evaluation or reproduction occurred.

The publication audit also checks that pooled response fractions and local
surrogate means agree with the eight window measurements. It authenticates
the saved arrays and published result; it does not independently recompute
every percentile, entropy or emission statistic. Earlier CPU tests cover
constant versus alternating responses, independent-window boundaries, exact
thresholds, float32 cancellation near zero and malformed inputs.

The observations cover one seed, eight selected training windows and short
reset trajectories. They are not a census of all live-stream states. Marginal
spike entropy is not joint information, predictive usefulness or a pruning
criterion. The current CUDA projections remain dense, so fewer events would
not itself demonstrate faster inference.

The queued learning-rate and replay-capacity studies keep the neuron equations
unchanged and test separate learning-policy changes. A smaller final-stage
learning rate starts from a model that already shows strong saturation; it
cannot establish what a different rate from random initialization would do.
Likewise, more replay slots test retention coverage, not a direct correction
of membrane scale. No default policy or growth rule is changed by this report.

To reauthenticate publication without new CUDA work, choose fresh paths:

```powershell
python scripts/publish_prose_spikes.py --report runs/prose-spike-recheck.json --audit runs/prose-spike-recheck-audit.json
```
