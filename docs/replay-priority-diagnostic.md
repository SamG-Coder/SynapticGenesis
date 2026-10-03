# Measuring interference during live learning

The native diagnostic can measure which stored examples a source update damages
without changing the learning trajectory. In this experiment, selecting the
highest-loss example would usually select a different example from the one with
the largest immediate loss increase. Measuring that difference is expensive:
the declared diagnostic added about 54% to the live loop. Replay selection
remains the original uniform policy within uniformly selected nonempty stages.

This is the first measurement stage proposed in the
[replay research note](replay-priority-research.md). It does not establish that
using the measured scores for replay improves retention.

## Implementation and isolation

`src/live.cuh` has an optional `LiveSourceObserver` around the actual source
update: after the current window is identified and before its forward pass,
then after AdamW updates the shared weights and before scheduled replay.
The production CLI passes no observer. Neuron equations, gradients, optimizer,
speech and replay selection remain in the existing implementation.

`experiments/replay_priority_scoring.cuh` owns the diagnostic pool, execution
views and score journal. It samples without replacement from each stage's
stored descriptors using its own deterministic RNG. Sampling never consumes
the learning/replay or speech RNG. The candidate pool only contains already
observed training windows. An example may recur in later pools.

Each scoring view shares weights and Adam moments with the live root but owns
its recurrent state, activations and cuBLAS handle. It runs the production
forward computation in strict FP32 with reset recurrence. It never runs
backward or an optimizer update. The learner keeps its inherited TF32 setting.
Scoring does not borrow learning replay views, whose math setting and state
must remain intact. Views are cached by actual window length.

Scores are mean next-byte loss, loss weighted by the existing answer emphasis,
and mean loss on emphasized answer targets alone. The answer metric is null
when a window contains no emphasized targets. All use natural logarithms.
The weighted mean uses double-precision host accumulation of the native
per-byte float losses and the source's existing weights. There is no new loss
function in the learner. Both sides of a comparison use identical recurrence
initialization; the live source update itself retains its ordinary streaming
state and inherited Adam history.

The diagnostic executable calls the same `live_tick` and curriculum transition
functions as production. Its optional first-update snapshots are disposable
ordinary checkpoints for CPU inspection, with no live stream state. The timed
branches do not save snapshots or perform state-hash audits inside the loop.

## Declared comparison

The three 1,951,624-parameter associative models are the recorded
160,000-observation ordinary checkpoints from the admitted earlier lineage.
Their hashes and original four-stage schedule are authenticated against the
published teacher-retention study and `data/training-selection.json`. These
earlier diagnostic bases still have learning time remaining in that unchanged
schedule. No new book edition, teacher, held-out answer or generated text enters
learning. The main continuation bases remain at 190,000 observations.

For seeds 1337, 2026 and 31415:

- Continue 512 source observations, from 160,000 to 160,512, with the inherited
  replay interval of four. This gives 640 optimizer updates and 96 generated
  bytes per continuation.
- Measure observations 160,001, 160,033, and so on through 160,481: 16 pools per
  model, each with eight candidates from each of four source stages.
- Score each pool before and after the actual source update, before uniform
  replay. This adds 1,024 forwards and 97,424 scored byte targets per model.
- Run each timed condition twice in alternating order. Use separate old and
  rebuilt production executable controls as well as the disabled-observer arm.
- Independently score all 32 candidates on each side of the first update using
  the CPU reference. This checks 192 candidate/phase cases across the seeds.

The [recorded protocol and raw timings](../reports/replay-priority-comparison.json)
were written before execution. The hardware was an RTX 5080 with 16,303 MiB,
driver 616.64, with other desktop GPU contexts active. Timing is the native
probe loop, including learning, uniform replay, scheduled speech and diagnostic
work. It excludes process startup, initial loading and final checkpoint save.
This is a small diagnostic, with two timing repeats per seed; it is not a
time-matched comparison of two learning policies.

## Cost and observed signal

| Seed | Ordinary loop, seconds | Measured loop, seconds | Extra time | Ordinary new byte targets/s | Measured new byte targets/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1337 | 1.819 | 2.801 | 54.0% | 35,980 | 23,364 |
| 2026 | 1.819 | 2.798 | 53.8% | 35,973 | 23,386 |
| 31415 | 1.821 | 2.808 | 54.3% | 35,940 | 23,300 |

Times are medians of two complete loops. These throughput figures count new
source byte targets, not generated text or subword tokens. They include the
cost of the inherited replay and scheduled speech. All measured loops reached
the same complete checkpoint and speech as their corresponding control.

Five scorer lengths required 116,581,224 bytes (111.18 MiB) of additional owned
device arrays. The observed drop in free GPU memory across their construction
was 459,276,288 bytes (438 MiB), including library/allocator overhead. Shared
weights/moments are excluded from the owned-array count. The free-memory
measurement is not a process-wide peak or a portable allocation guarantee.
Other corpora can require more distinct lengths and larger view caches.

Scoring took about 1.01 seconds per loop; view setup took 14–15 ms. Measured
scoring ticks had approximate p95 values of 123–129 ms, while ordinary tick p50
was about 2.8 ms. Histogram percentiles round upward to the existing logarithmic
bin boundary. In this fixed phase the scheduled speech tick did not coincide
with a scoring tick, so these measurements do not test that worst-case
combination. The single live loop waits for scoring to finish.

| Seed | Median rank correlation: current weighted loss vs loss increase | Same highest-ranked candidate | Fraction of candidates with increased weighted loss |
| --- | ---: | ---: | ---: |
| 1337 | 0.060 | 0 / 16 pools | 50.6% |
| 2026 | -0.135 | 1 / 16 pools | 47.7% |
| 31415 | -0.034 | 0 / 16 pools | 49.6% |

These are descriptive native results. The correlation is calculated within each
pool and summarized by its median. Current loss is a freshly computed difficulty
proxy; this experiment does not test stale cached scores. Approximately half
the candidates worsened immediately while mean weighted loss across the pools
fell slightly. A source update can improve some predictions and damage others.
Neither a loss increase nor a rank proves semantic forgetting or the benefit of
replaying that window.

The [summary](../reports/replay-priority-summary.json) also includes unweighted
and answer-only results, per-stage coverage, full timing repeats and top-four
ranking overlap. Candidate journals are published for
[1337](../reports/replay-priority-scores-1337.jsonl),
[2026](../reports/replay-priority-scores-2026.jsonl) and
[31415](../reports/replay-priority-scores-31415.jsonl).

## Verification and numerical limitation

All 18 native CTest suites passed. The synthetic diagnostic suite passed 21
native invocations, including exact old-executable checkpoint and speech
controls, FP32 and TF32 learning, reordered scoring, process restart, a
curriculum boundary, answer weighting, one-byte tails and seven rejected
invocations. Its CPU score errors stayed below 3.98e-7. A targeted CUDA memcheck
reported zero errors. These fixtures establish isolation and bounded execution,
not language quality.

The learned execution audit independently reconstructed all 48 candidate pools,
source windows, uniform replay state, RNGs and counters. Old executable,
rebuilt executable, disabled diagnostic and both measured runs produced
identical final checkpoint bytes for every seed. Generated speech was identical
as well. Inputs remained unchanged.

**The learned CPU scoring check did not fully pass.** Of 192 candidate/phase
checks, 191 met the unchanged 3e-5 absolute score tolerance. Seed 1337's
pre-update window at document 39,564, offset 0, length 76 had maximum error
0.001270609. Seeds 2026 and 31415 had maximum errors 1.17e-6 and 2.05e-5.
Independent scoring covers the first measured pool only, not every later score.

The [post-hoc trace diagnosis](../reports/replay-priority-diagnosis.json)
reproduced the failing native score exactly. The first differing spike was at
layer 0, input position 52, neuron 390: CPU membrane 1.000000119 and CUDA
membrane 0.999998868 lay on opposite sides of the threshold. Forcing only that
CPU spike to the CUDA decision matched all subsequent native spike decisions
and reduced the five-answer-byte NLL discrepancy to 1.81e-6. This intervention
localizes the discrepancy; the original independent check remains failed and
the production model is unchanged.

The [execution audit](../reports/replay-priority-execution.json) and
[fixture validation](../reports/replay-priority-validation.json) retain these
separate verification boundaries.

Published JSON/JSONL uses LF line endings, and cross-report hashes identify those
published bytes. The [publication manifest](../reports/replay-priority-publication.json)
also retains hashes of the original, unmodified Windows captures. Measurement
values, source/executable identities and failed checks are unchanged.

## Next decision

Do not switch the default policy to highest-loss replay. In this bounded sample
that proxy rarely chose the most immediately interfered candidate. Before
promoting any interference selector, reduce and measure its scoring cost,
preserve stage coverage, and compare retention against uniform replay with
both equal source exposure and equal wall-clock budgets. A control that pays
the same scoring cost but keeps uniform selection is needed to isolate the
selection effect. Near-threshold score discrepancies must stay visible.

This stage adds measurement. It does not establish better language, durable
memory, biological development, model growth benefits or generational progress.

## Reproduction

Prepared admitted sources and local ancestor checkpoints are required; the
public repository records their hashes and protocol rather than embedding the
checkpoint payloads. Preserve the prior executable before rebuilding production.
Run GPU comparisons sequentially.

```powershell
.\build.ps1
.\build.ps1 -ReplayPriorityProbe
python tests/replay_priority_probe.py --out runs/my-replay-fixture
python scripts/replay_priority_experiment.py --out runs/my-replay-panel
python tests/replay_priority_experiment.py --root runs/my-replay-panel --out runs/my-replay-panel/execution.json
python scripts/summarize_replay_priority.py --root runs/my-replay-panel --execution runs/my-replay-panel/execution.json --out runs/my-replay-panel/summary.json
```

The fixture and experiment accept `--native`, `--original` (experiment only)
and `--probe` paths as shown in their `--help`. Defaults use the preserved
`build/pre-replay-priority/synapticgenesis.exe` and the separate diagnostic build.
All diagnostic output directories must be fresh. Boolean diagnostic options
such as `--snapshots` and `--audit-state` take explicit `0` or `1` values.
