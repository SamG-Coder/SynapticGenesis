# Bounded reading breadth: declared native comparison

This comparison follows the failed durable-retention gate in the
[live-rehearsal study](live-reading-results.md) and the audited
[reading expansion](reading-breadth.md). It tests a broader selection with less
repetition under two existing replay rates. It changes no neuron equations,
optimizer, model size, population policy or native forward implementation.

## Conditions fixed before training

Start with the ordinary-replay associative models at 190,000 source
observations from the teacher-retention study, seeds 1337, 2026 and 31415.
Each has 1,951,624 parameters. Continue its complete native checkpoint,
including weights, Adam history, recurrent state, speech RNG and grouped replay
reservoirs. Do not start from the later overfit reading checkpoints.

| Condition | Newly admitted training material | Replay cadence |
| --- | --- | --- |
| `starter-4` | Ten starter stories | Every fourth observation |
| `broader-4` | The same ten stories, then 34 expansion stories | Every fourth observation |
| `starter-1` | Ten starter stories | Every observation |
| `broader-1` | The same ten stories, then 34 expansion stories | Every observation |

The new stage uses learning rate 0.000075, inherited TF32 training and ordered
gradient reductions. A source observation is up to 128 next-byte targets, with
shorter tails at whole-document boundaries. Documents repeat sequentially in
their selected order. All arms retain 1,024 replay descriptors across five
source groups. Existing native admission preserves earlier source editions and
replay history. Generated text never supplies learning targets.

Assess after **116, 546 and 1,092** new observations. The primary comparison is
546, one complete broader pass. The last endpoint is a fixed continuation to
two broader passes, not a checkpoint selected after seeing development scores.

| Additional observations | Starter targets | Broader targets | Interpretation |
| --- | ---: | ---: | --- |
| 116 | 14,182 | 14,182 | Identical source prefix within each replay cadence |
| 546 | 66,704 | 67,188 | About 4.7 starter passes versus one complete broader pass |
| 1,092 | 133,317 | 134,376 | About 9.4 starter passes versus two complete broader passes |

Within a replay cadence, optimizer counts and scheduled speech are equal.
Target bytes are not exactly equal: the primary difference is 484 targets,
about 0.73% of the starter count. Short tails and changed replay selections also
affect work. Content identity, breadth, order after the shared prefix and
repetition change together. This tests the practical source selection; it does
not isolate a universal causal effect of corpus size.

Speech remains 96 bytes every 500 source observations with the inherited prompt
`The bird ` and shared mutable weights/state. Native runs are sequential; arm
order reverses on alternating seeds. Executables, source files, selected
editions, ancestors and schedules are authenticated before and after execution.

## Common measurements and candidate rule

At each ancestor and endpoint, the existing native diagnostic makes a disposable
zero-update view. It scores every within-document target of the same 44
training-monitor stories and seven development stories, with strict FP32,
128-target reset windows and exact short tails. Report four separate weighted
losses: starter training, expansion training, starter development and expansion
development. Expansion training texts are an unlearned monitor in starter arms;
assessment supplies no updates or feedback to any arm. Every original live
checkpoint stays unchanged.

Also assess the existing 576-question binding development set and the three
earlier development books, using the unchanged 32 batches of 16 sequences of
128 targets per book. At final endpoints record the two fixed 384-byte story
samples and seven rounds of 512-byte decoding. Apply all six previous question
prompts to all twelve final models, generating 192 bytes each at temperature
0.8, top-k 40, seed 42, fresh context, with no retries. These continuations are
illustrative output, not a new instruction-following benchmark.

At the primary endpoint, evaluate broader versus starter separately for each
replay cadence. The broader arm qualifies as a candidate only if **every seed**
meets all six requirements:

1. Both development-set losses improve from its own ancestor.
2. Both development-set losses are no worse than the matching starter arm.
3. Complete binding accuracy drops by no more than five percentage points from its ancestor.
4. Binding is no more than 2.5 percentage points worse than the matching starter arm.
5. Every earlier book loss is at most 0.05 nats/byte above its ancestor.
6. Every earlier book loss is at most 0.02 nats/byte above the matching starter arm.

Report every endpoint, seed, failure and continuation result. Passing permits
further research; it is not automatic population promotion, biological age,
general conversation or lifelong retention. The source selection and
development sets share some authors and constructions, and the three ancestors
are related experimental runs. No reserved tests are scored.

## Execution checks and timing

Reuse the independent replay/cursor reference and common assessment audit.
Check all replay descriptors, source counters, RNG state, document boundaries,
per-group counts and speech counters against complete checkpoints. Assessment
copies must keep weights and Adam moments byte-identical. The first 116 shared
observations must produce identical learned payloads, including recurrence,
between starter and broader arms at each cadence. Reproduce the earlier
starter-only baseline assessments from the same ancestors.

The seed-1337 smoke run uses endpoints 16, 116 and 546, shortened output samples
and two book-evaluation batches. Four uninterrupted native controls must match
the segmented smoke checkpoints and speech exactly across the scheduled
500-observation speech boundary. Smoke measurements do not evaluate the gate.

The unchanged fixed-group CPU reference additionally checks final native binding
scores and greedy answers. It is a limited numerical check, not a full independent
quality evaluation. Preserve any strict-tolerance failure; do not relax it or
reinterpret a diagnostic intervention as a pass.

Live times include learning, replay, periodic generation, logging and saves,
excluding process setup and assessment. Decode includes sampling, transfers
and synchronization, excluding loading, graph capture and prompt processing.
Other desktop GPU contexts remain active. Report actual source/replay bytes,
optimizer updates and speech counts with speed.

```powershell
python scripts/reading_breadth_experiment.py --smoke --out runs/reading-breadth-smoke
python tests/reading_breadth_experiment.py --root runs/reading-breadth-smoke --continuation-control
python scripts/summarize_reading_breadth.py --root runs/reading-breadth-smoke

python scripts/reading_breadth_experiment.py --out runs/reading-breadth-panel
python tests/reading_breadth_experiment.py --root runs/reading-breadth-panel
python tests/binding_learned_oracle.py --root runs/reading-breadth-panel
python scripts/summarize_reading_breadth.py --root runs/reading-breadth-panel
python scripts/sample_live_reading_questions.py --root runs/reading-breadth-panel
```

Use fresh output directories and the exact already-audited selected editions
and ancestors. Python orchestrates and checks data/state; the existing C++/CUDA
path performs every model update, assessment and generation.
