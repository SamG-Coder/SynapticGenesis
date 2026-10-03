# Captured replay scoring

CUDA graph capture reduces the cost of the existing replay interference
diagnostic without changing its scores or learning trajectory. In the fixed
three-seed comparison, captured scoring made the measured live loop 12–16%
faster than ordinary scoring. Its remaining overhead over disabled scoring was
33–38%. This is a runtime improvement; replay selection is still unchanged.

`experiments/replay_score_graph.cuh` captures the same production
`Model::forward_device(false)` and byte classifier, with pinned input/target and
loss buffers. Each graph belongs to a scoring view with independent recurrence
and a strict-FP32 cuBLAS handle. It reads the current shared weights on every
launch. The caller synchronizes the live writer once before each scoring pass,
and every graph launch completes before learning can continue. Graphs are
destroyed before their model buffers. No new neuron or classifier equations are
introduced.

The optional diagnostic flag is `--graph-scoring 1`. The default remains `0`,
preserving the original diagnostic and its ordinary execution control.
Construction warms kernels/cuBLAS and captures each actual window length once.
The warm-up runs on the scoring view, not on the live recurrent state.

## Measured comparison

The experiment repeats the [declared interference diagnostic](replay-priority-diagnostic.md):
three admitted 160,000-observation ancestors, 512 source updates each, 16
candidate pools of 32 windows, 1,024 scoring forwards per measured continuation,
and two fresh-process timing rounds per condition. The order rotates between
disabled scoring, ordinary scoring and captured scoring. GPU commands run
sequentially on the RTX 5080 with other desktop contexts active.

| Seed | Disabled, seconds | Ordinary scoring, seconds | Captured scoring, seconds | Ordinary / captured speed ratio | Captured overhead vs disabled |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1337 | 1.903 | 2.931 | 2.530 | 1.158 | 33.0% |
| 2026 | 1.818 | 2.803 | 2.477 | 1.131 | 36.2% |
| 31415 | 1.810 | 2.789 | 2.495 | 1.118 | 37.8% |

These are medians of two loops; the larger early timings for seed 1337 are
retained. Timing includes graph construction, source learning, replay,
diagnostic work and scheduled speech. Initial loading and final checkpoint
saving are excluded. Graph construction adds startup work, so these results do
not promise the same benefit for very short streams. The experiment is too small
to estimate a production latency distribution.

All 18 learned runs produced the previously recorded checkpoint and speech for
their seed. Every candidate's before/after loss matched the earlier ordinary
diagnostic exactly, including its known threshold-sensitive window. The graph
did not change the 111.18 MiB of scorer-owned device arrays; observed free-memory
drops during view construction remained 438 MiB. This does not measure graph
metadata or whole-process peak memory separately.

The expanded fixture suite passed 28 native invocations, checking graph versus
ordinary scores, before/after snapshots, complete checkpoints, speech and
restart under FP32 and TF32 learning. It includes short document tails, answer
weighting and curriculum transitions. A targeted CUDA memcheck completed with
zero errors. The original learned CPU scoring failure remains a failure: graph
equality does not establish independent CPU parity or improve language quality.

The [full recorded comparison](../reports/replay-score-graph.json) retains the
protocol, source identities and all timing repeats. The
[validation report](../reports/replay-score-graph-validation.json) retains the
fixture results and sanitizer identity.

## Next experiment

The remaining scoring cost motivates testing a smaller candidate pool, rather
than enabling the 32-candidate diagnostic as the live default. A two-candidate
comparison within a uniformly selected replay stage can preserve stage coverage
while reducing forward work. That selector needs its own causal execution and
restart checks, a scoring-cost-matched uniform control, and retention results
under both equal exposure and equal time budgets before promotion.

```powershell
.\build.ps1 -ReplayPriorityProbe
python tests/replay_priority_probe.py --out runs/my-graph-fixture
python scripts/replay_score_graph_experiment.py --out runs/my-graph-panel
```

The local admitted ancestors and the prior diagnostic's raw speech records are
required for the learned comparison. It authenticates them before using the
same selected training sources. It does not add new content or use generated
text as targets.
