# Two-candidate live replay experiment

The tested selector did **not** improve retention reliably. It remains an
optional experiment; ordinary live learning keeps its existing uniform replay.
The next scaling stage should increase model capacity and varied source text,
using the established replay policy as the control.

Before a scheduled replay, the experiment previews the existing uniform stage
and window choice on a copy of host state. It samples one different stored
window of the same length from that stage. Both candidates are scored before
and after the real source update through the shared production forward path.
The alternative replaces the uniform choice only when its positive increase
in weighted loss exceeds the uniform candidate's increase. Scoring uses the
[captured forward implementation](replay-score-graphs.md).

The real replay RNG still advances at its original point in the live loop.
Overrides must belong to the same stage and already exist in the stored
reservoir. The common scorer lives in `experiments/replay_candidate_scoring.cuh`;
selection, its policy identity, and experiment orchestration are separate
modules. The default observer returns the original replay choice.

## Declared comparison

Three admitted 160,000-observation ancestors each ran all three modes:
ordinary uniform replay, uniform replay that pays the candidate scoring cost,
and interference selection. Each mode ran for 4,096 source observations and,
separately, six seconds of live execution. Arm order rotated across seeds and
budgets. The protocol was saved before execution. The same original selected
curriculum supplied all learning; no generated text became learning targets.

| Budget | Mode | Mean binding accuracy | Mean source bytes/s | Mean observations |
|---|---|---:|---:|---:|
| 4,096 observations | Ordinary uniform | 47.69% | 36,795 | 4,096 |
| 4,096 observations | Scored uniform | 47.69% | 31,444 | 4,096 |
| 4,096 observations | Interference | 40.28% | 31,185 | 4,096 |
| Six seconds | Ordinary uniform | 56.71% | 36,890 | 1,730.3 |
| Six seconds | Scored uniform | 51.62% | 30,706 | 1,440.7 |
| Six seconds | Interference | 51.39% | 30,881 | 1,448.7 |

Means include every seed. Binding accuracy requires all four questions in a
development group to be correct. The declared gate required an improvement
over both controls for every seed and budget, no book-loss regression above
0.02 nats/byte, and no binding loss from the ancestor. All six seed/budget
combinations failed that gate. Under equal exposure, the selector changed
1,135 of 3,072 replay choices; its mean binding score fell 7.41 percentage
points versus uniform. Under equal time it fell 5.32 points versus ordinary
uniform. Book losses stayed within the declared margin.

Live timing includes learning, replay, scoring, graph construction, and
scheduled speech, but excludes loading, final checkpoint save and assessments.
The time limit stops after a complete tick. This is one run per arm on an
RTX 5080 with other desktop GPU contexts active, not an isolated throughput
benchmark or a general language evaluation. Tokens here are UTF-8 bytes.

## Verification and limits

All 18 final arms passed independent checks of source traversal, replay RNG,
stored windows, candidate membership and speech counts. Equal-exposure
ordinary and scored-uniform checkpoints were byte-identical. Their paired
candidate workloads also matched the priority arms. Fixed final development
scores agreed with the CPU reference within the unchanged 3e-5 tolerance;
short greedy completions matched. This does not remove the earlier
[learned-threshold discrepancy](replay-priority-diagnostic.md).

The focused fixture exercised FP32 and TF32, an actual replay override,
uninterrupted versus resumed checkpoints/speech/selection journals, time-limit
stopping and restart, and five policy identity rejections. It used 20 selector
calls and four native setup calls. The existing scoring fixture passed all
28 calls after extraction of the common scorer; all 18 native CTests passed.
A targeted selector CUDA memory-sanitizer run reported zero errors.

Experiment resumes require an additional checksummed `.sgpriority` sidecar
binding the checkpoint to its base, curriculum, mode, seed and scoring version.
The native checkpoint format is unchanged. These descendants have not been
admitted as population parents or teachers. The main CLI does not enforce the
experimental sidecar, so it is the experiment runner's responsibility to keep
the checkpoint and sidecar together.

Machine-readable [comparison](../reports/replay-selection-comparison.json),
[execution audit](../reports/replay-selection-execution.json),
[gate and aggregates](../reports/replay-selection-summary.json), and
[validation record](../reports/replay-selection-validation.json) accompany the
implementation. Full native journals and checkpoints remain under the local
`runs/replay-selection-panel` capture.

```powershell
.\build.ps1 -ReplaySelectionProbe
python tests/replay_selection_probe.py --out runs/replay-selection-fixture-new
python scripts/replay_selection_experiment.py --out runs/replay-selection-new
python tests/replay_selection_experiment.py --root runs/replay-selection-new --out runs/replay-selection-new/execution.json
python scripts/summarize_replay_selection.py --root runs/replay-selection-new --out runs/replay-selection-new/summary.json
```
