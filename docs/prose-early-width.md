# Early learning at fixed depth and width ratio

The [completed neuron panel](prose-spike-results.md) shows dense firing in the
105M model by its first saved learning age. Lowering the rate only after stage
three cannot tell us whether different early updates would avoid that regime.
This study measures actual early source updates and tests a simple core-rate
control from random initialization.

The [specification](../data/prose-early-width-v1.json) fixes depth at eight and
spiking width at four times channel width. This removes the depth and width-ratio
changes in the original 2M-to-27M comparison. Each case runs seeds 1337, 2027 and
4099, for fifteen fresh models. All receive the original first curriculum stage:
four selected reading books, four passes, 8,176 source observations and 1,045,096
source next-byte targets per model. Original replay and scheduled speech remain
enabled. The remaining source stages and astronomy extension are not learned.

| Case | Channels | Neurons per layer | Total neurons | Parameters | Core rate | Output-group rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| C256 control | 256 | 1,024 | 8,192 | 7,339,280 | 0.0003 | 0.0003 |
| C512 control | 512 | 2,048 | 16,384 | 27,260,432 | 0.0003 | 0.0003 |
| C1024 control | 1,024 | 4,096 | 32,768 | 104,851,472 | 0.0003 | 0.0003 |
| C512 half core rate | 512 | 2,048 | 16,384 | 27,260,432 | 0.00015 | 0.0003 |
| C1024 quarter core rate | 1,024 | 4,096 | 32,768 | 104,851,472 | 0.000075 | 0.0003 |

The existing `--core-scale` option applies to embeddings and all spiking blocks.
Final normalization and the vocabulary head retain the original rate. The
multiplier also affects their respective AdamW decay steps. Initialization,
neuron equations, loss, clipping and moment accumulation are unchanged. This
is a two-group rate control, not a complete maximal-update parametrization or
a biological homeostatic controller.

## Native observations

`experiments/early_learning_probe.cu` initializes the same native model, source
cursor, curriculum and replay state as production and calls the existing
`live_tick`. The separate `experiments/learning_scale_observer.cuh` uses callbacks
immediately before and after the source update. It does not replace the forward,
backward, optimizer, replay or generation implementations.

At source updates 1, 32, 128, 512, 2,048 and 8,176, it records:

- Sixty-four deterministic, evenly spaced parameter coordinates per tensor,
  covering 116 tensors and 7,424 coordinates in each study model. Values before
  and after the source update, displacement from initialization and post-clipping
  gradients are preserved as raw float32 files. Sampled RMS values describe
  those coordinates, not complete parameter tensors.
- Full layer-cache statistics for that source chunk: normalized input, projected
  drive, pre-reset membrane, emitted signal, signed firing fraction and zero
  local spike-surrogate fraction. These caches came from the forward before
  the update, not a second forward with the updated weights.
- Source document/offset/length, optimizer steps after source learning and after
  any replay, source loss, pre-clipping gradient norm, clipping factor and
  speech/replay counts.

The run also counts clipping over every source update. This is not a gradient
census of replay updates. The observer copies the source gradient after clipping
and before replay can replace the shared gradient buffer. Forward caches include
the live recurrent stream and scheduled speech; these are not reset windows.

At the final first-stage checkpoint, the shared assessment module runs the
original four validation books, two training-book monitors and four raw
generations. At this age, training-book scores measure fit to the reading stage;
they do not establish retention through later novel content. Reserved test
books remain unscored.

## Guarded execution and evidence

Build and CPU preflight passed, as recorded in the
[preflight report](../reports/prose-early-width-preflight.json). Native host-only
tests cover role coverage, scalar moments, exact firing boundaries, float32
cancellation and eleven malformed inputs. CPU orchestration checks preserve
archived 27M/105M learning arguments and reject thirteen altered exposure,
coordinate, clipping, rate or initialization cases. The artifact fixture is
synthetic; it is not evidence of native model learning.

The original attempt stopped during the tiny native guard's CPU speech audit,
before any study case. The [speech audit correction](early-width-speech-audit.md)
preserves that failure and verifies its native checkpoints, coordinates and
raw speech. Full study execution remains pending. Before cases are accepted:

1. A fresh C8/H32/L2 guard runs 511 source observations through both the production
   CLI and the observer. This includes replay and scheduled speech. Initial and
   final checkpoints must be byte-identical. The last observation has no replay,
   allowing direct comparison of saved weights with after-update coordinates.
2. Seed-1337 controls must reproduce the original 27M and 105M initial and
   first-stage checkpoint bytes, six book scores and four generated outputs.
   Failure preserves artifacts and stops later candidates.
3. CPU verification independently reconstructs sampled statistics, source windows
   and optimizer boundaries from raw coordinates and source text.
4. Cases for a seed require matched source/replay/speech counts, cursor/RNG and
   replay payload. Same-width rate pairs start with identical weight, optimizer
   and recurrent arrays; only their declared core-rate field differs.

The runner waits on the actual replay-capacity process handle, then authenticates
its completed result before CUDA work. That predecessor waits behind the
learning-rate study. Planned work comprises two native guard calls, fifteen
instrumented learners and 150 read-only assessment calls. No model is
automatically promoted for teaching, reproduction or a default role.

The diagnostic has a separate executable and build directory. Instrumented
timing includes coordinate copies, CPU statistics and artifact writes.
Subtracting measured callback time does not yield a controlled production
speed benchmark.

```powershell
# In the study source checkout:
.\build.ps1 -EarlyLearningProbe

# With the main repository as the working directory, using fresh output paths:
python D:\SynapticGenesis\runs\early-width-worktree\scripts\prose_early_width.py declare --out runs/prose-early-width
python D:\SynapticGenesis\runs\early-width-worktree\scripts\prose_early_width.py run --out runs/prose-early-width
```

For a live predecessor, pass its actual `--wait-pid` and `--wait-executable`.
Keep the checkout, binaries and declared inputs fixed until completion. The
protocol records their hashes and the source commit.

## Research connection and limits

Turrigiano and colleagues reported activity-dependent changes in overall
synaptic input strength in cortical cultures, including a return toward control
firing after initially increased activity. The publisher abstract supports
activity regulation as an inspiration; it supplies no learning-rate schedule,
target firing rate or developmental clock for this model. Only the accessible
abstract and figure descriptions were reviewed here.
[Original study, Nature 1998](https://www.nature.com/articles/36103).

Zenke and Vogels studied surrogate shape and scale. The authors' summary reports
robustness to shape with stronger sensitivity to scale, including sparse-activity
regularization experiments. Direct full-paper retrieval was blocked here; this
record uses the accessible author summary and does not claim a replication.
[Authors' account](https://zenkelab.org/2020/06/preprint-the-remarkable-robustness-of-surrogate-gradient-learning-for-instilling-complex-function-in-spiking-neural-networks/).

The engineering decision is to measure update scale before adding an activity
controller. A byte timestep is not a biological millisecond, and lower firing
is not itself a language-quality objective. The 105M model has denser final
responses than 27M yet better final validation loss, so saturation cannot alone
explain the size ranking. A promising early-rate result still needs later-source
retention and separate production speed measurements. These papers are not
training data. The [research ledger](../reports/early-learning-research.json)
records inspection boundaries.
