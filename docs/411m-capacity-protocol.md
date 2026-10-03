# Matched 105M and 411M live capacity check

Status: executed successfully after all declared CUDA gates passed. The
[results](411m-capacity-results.md) record actual allocation, learning and
generation at both sizes. The [borrowing constructor](live-view-construction.md)
and [shared evaluation views](shared-evaluation-views.md) passed their native
checks first. The [preflight record](../reports/capacity-411m-preflight.json)
preserves the earlier preparation stage. This document does not queue GPU work.

The candidate capacity executable is compiled from the same C++/CUDA source
as the candidate runtime. It calls the production `live_tick` and shared
speaker. The experiment driver now accepts explicit executable paths and
repeated `--shape CHANNELS HIDDEN LAYERS` arguments. Omitting these arguments
retains the original four-size panel and default executable locations.

| Shape | Parameters | Spiking neurons |
| --- | ---: | ---: |
| 1,024 channels, 4,096 neurons per layer, 8 layers | 104,851,472 | 32,768 |
| 2,048 channels, 8,192 neurons per layer, 8 layers | 411,028,496 | 65,536 |

Both disposable founders start from random seed 1337 and read the existing
admitted `general-foundations-v1` training edition. The smaller, fixed text
fixture keeps this capacity check comparable to the earlier benchmark. It is
not the full prose curriculum or a new source admission.

The matched settings are a 128-byte maximum observation, learning rate 0.0003,
uniform reservoir replay every four observations with 1,024 slots, TF32 and
graph generation. The speaker generates 64 bytes from `The bird ` every 128
observations. After 128 warm-up observations, three consecutive rounds each
measure 512 further observations. This gives 1,664 source observations per
shape, including warm-up. Source/replay pairs and generated-byte counts must
match between shapes; they are not assumed to be exactly 128 bytes per source
observation because document tails can be shorter.

Live timing includes source learning, replay, scheduled speech, synchronization
and small speech writes. It excludes initialization, the first 128 observations
and checkpoint saving. The separate generation measurement uses the same live
speaker after learning: one untimed 1,024-byte run, then three timed runs with
fixed weights, recurrent state and sampling RNG. Prompt processing and CPU
sampling remain inside those timings. Every repetition must produce identical
bytes. Medians summarize the three measured rounds.

All tokens in this runtime are UTF-8 bytes. These rates cannot be directly
compared with another model's subword tokens per second. Finite updates and
successful generation establish execution at this size, not useful answers,
long-term retention or better language quality. No benchmark checkpoint is
saved or admitted for teaching or reproduction.

Before this panel, the capacity observer must reproduce complete production
checkpoints in both FP32 and TF32 on its existing tiny synthetic fixture.
The fixture uses four native calls; its checkpoints are disposable diagnostics.
The source driver authenticates the admitted text and records hashes of the
executable, included CUDA sources and orchestration before execution. It checks
those identities before each shape and at the end, validates finite timings
and losses, and compares actual exposure counts between shapes.

Allocation-only checks exercise the production root, speaker and both bounded
view caches at each size before actual learning. Explicit float-buffer peaks
exclude integer buffers, CUDA/cuBLAS internals, graphs and other applications.
Free VRAM values at named boundaries are snapshots, not a continuous process
peak. Successful allocation alone does not prove forward/backward execution,
graph capture or arbitrary prompt lengths fit.

From the main repository, after the candidate's earlier gates pass, use its
explicit paths and fresh output directories:

```powershell
python runs/live-view-allocation-worktree/tests/capacity_probe.py --out runs/capacity-shared-view-compat-new --probe-exe runs/live-view-allocation-worktree/build/capacity-probe/synaptic-capacity-probe.exe --native-exe runs/live-view-allocation-worktree/build/synapticgenesis.exe
python runs/live-view-allocation-worktree/scripts/capacity_experiment.py --out runs/capacity-411m-new --probe-exe runs/live-view-allocation-worktree/build/capacity-probe/synaptic-capacity-probe.exe --shape 1024 4096 8 --shape 2048 8192 8
```

The current preserved main executable is not replaced by preparing or running
this panel. A failed gate retains its outputs for inspection before any later
gate is attempted.
