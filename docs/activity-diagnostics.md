# Native activity diagnostics

The [plasticity research review](research-live-plasticity-2026-10.md) identifies a measurement gap: parameter magnitudes and passive time constants do not describe which features emit signals on observed text. These diagnostics use the existing native forward trace to examine that activity. They perform no learning or structural change.

`tests/learned_trace_dump.cu` calls the production forward implementation in strict FP32, starting with reset recurrent state. It writes membrane values, signed spikes, continuous emissions and intermediate arrays. `scripts/inspect_native_activity.py` invokes that executable, authenticates its inputs and summarizes the emitted arrays. Its CPU checkpoint reader only decodes parameters; it does not run another model forward pass. An independent log-sum-exp calculation checks loss/input/logit consistency without claiming CPU-model parity.

`scripts/activity_metrics.py` contains the array statistics. Its analytic tests cover a known silent-emission energy fraction, weighted output-column magnitudes, rank-one and rank-two activity, invariance to constant offsets and global scaling, zero variance, exact firing boundaries, and invalid shapes/values.

| Measurement | Definition and interpretation |
| --- | --- |
| Spike event fraction | Fraction of sampled byte/neuron positions with a nonzero signed spike |
| Nonzero emission fraction | Fraction with absolute emission greater than `1e-6` |
| Silent-spike squared emission fraction | Sum of squared emissions where the spike is zero, divided by all sampled squared emissions; null when the denominator is zero. This is a mathematical signal magnitude, not measured electrical energy. |
| Never-spiked / never-emitted counts | Features with no such event in these sampled windows; no claim of permanent dormancy |
| Direct-output contribution proxy | Each feature's mean absolute emission multiplied by the L2 norm of its direct output-projection column |
| Centered emission participation ratio | `(trace C)^2 / trace(C^2)` for the centered emission Gram/covariance matrix; null for zero variance. It measures representation diversity in these observations. |
| Near-threshold fraction | Membrane values within `1e-5` of either fixed signed firing threshold |

The output proxy excludes cancellation, other residual paths and the associative cue/readout path. It cannot identify expendable features by itself. Participation ratio does not establish task capacity, learning capacity or plasticity. Spike inactivity does not imply zero signal in cells with a continuous trace.

## Fixed observation panel

`scripts/activity_panel.py` selects two documents each from the new reading, binding and narrative training groups of the authenticated retention-study curriculum. Document indexes are `floor(N/4)` and `floor(3N/4)` within each group. The centered window contains at most 257 bytes and supplies at most 256 next-byte observations. Selection does not use model outputs or evaluation answers. The prerequisite stage is not sampled.

The prepared windows contain 256, 256, 76, 72, 256 and 256 observations: **1,172 positions per model**. Every source window resets recurrent state. All nine 130,000-observation parents and all eighteen final ordinary/teacher models receive the same six windows, requiring **162 native diagnostic commands**. The model comparison is exploratory and includes every seed and architecture. Intermediate 160,000-observation checkpoints are not included.

Per-window records retain source/checkpoint/executable hashes and loss consistency. Pooled layer statistics concatenate the six independent native observation windows; each byte receives one weight, so longer windows contribute more. Centering occurs across the pooled observations and includes between-window variation. Neither these tiny samples nor a correlation with retained binding establishes a cause of forgetting.

```powershell
# Build the existing diagnostic separately; wait for other GPU studies to end.
.\build.ps1 -TraceDiagnostic
python tests/activity_metrics.py

python scripts/inspect_native_activity.py --checkpoint runs/example/latest.ckpt --input runs/example/selected-window.txt --out runs/example/activity
python scripts/activity_panel.py --study runs/teacher-retention-panel --out runs/native-activity-panel
```

Use fresh output directories. Python analysis requires NumPy and the existing CPU PyTorch checkpoint reader. Model computation remains native C++/CUDA. The panel requires the completed full retention study and its execution audit; it must run after the queued native compatibility and memory checks finish. It does not score reserved evaluation books or supply new training targets.

## Completed census

All **27 models and 162 native windows** completed. The [full panel](../reports/native-activity-panel.json) retains each model's pooled layer measurements, source selection and window-result identities. The [summary and artifact audit](../reports/native-activity-summary.json) authenticate every checkpoint, selected input, source file, diagnostic executable, command journal and all **4,212 trace files**. The maximum independent loss/input/logit consistency error is **1.184e-6**, below `3e-5`; this is not an independent CPU model-parity result.

Every neuron fired at least once in each model's pooled six-window sample. No neuron had zero emitted signal throughout that sample. Across the 108 pooled layers, spike event fractions range from **52.6% to 70.7%**. The sample therefore provides no never-firing units to use as a replacement criterion. This does not establish that every feature contributes useful information.

The table gives equal means over three seeds and four layers per architecture/role. Each layer pools 1,172 observed positions. Layers within a model and descendants of the same parent are related observations, not independent experimental replicates.

| Architecture | Role | Spike events | Silent-spike squared emission | Centered participation ratio |
| --- | --- | ---: | ---: | ---: |
| Selective H512 | Parent | 61.90% | 4.71% | 23.56 |
| Selective H512 | Ordinary replay | 66.17% | 4.17% | 22.93 |
| Selective H512 | Frozen-self teaching | 66.33% | 4.23% | 22.91 |
| Selective H588 | Parent | 61.16% | 4.85% | 23.95 |
| Selective H588 | Ordinary replay | 65.32% | 4.32% | 23.64 |
| Selective H588 | Frozen-self teaching | 65.63% | 4.34% | 23.71 |
| Associative H512 | Parent | 59.75% | 5.28% | 27.97 |
| Associative H512 | Ordinary replay | 63.65% | 4.74% | 25.78 |
| Associative H512 | Frozen-self teaching | 63.46% | 4.85% | 26.65 |

Silent spike positions still carry part of the continuous emitted signal. Skipping their weights based only on predicted spikes would change this model's computation. Dense projections and associative cues also remain in the forward path. These observations do not benchmark predictive paging or establish its performance under memory pressure.

Participation ratios range from 14.71 to 30.22 across the sampled layers. These are centered signal-diversity statistics, not counts of useful neurons: one cannot subtract them from the hidden width and call the difference spare capacity. The census does not determine whether the experienced models have lost plasticity.

Analytic statistics and deterministic source-selection checks pass. The census makes no model updates, and all checkpoints remain unchanged. A separate experiment comparing full new-task learning curves for experienced and fresh learners is still required. The [gradient-conflict research](gradient-conflict-research.md) also motivates measuring actual optimizer displacements and per-source losses before introducing a retention intervention.

```powershell
# Audit saved evidence and aggregate it without launching the model again.
python scripts/summarize_activity.py --out reports/native-activity-summary.json
```
