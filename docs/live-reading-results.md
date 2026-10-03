# Live rehearsal: partial retention benefit and continued overfitting

The [declared comparison](live-reading-replay.md) is complete: three learned
ancestors, two replay cadences and 27 continuation endpoints. Rehearsing after
every observation improves final binding retention in all three seeds relative
to rehearsal every fourth observation, but both conditions lose earlier skills
and overfit the ten repeated training stories. The candidate rule fails; no
runtime default or population policy is promoted.

All model computation uses the existing C++/CUDA learner. These continuations
preserve weights, Adam history, recurrent state and replay history. They do not
use frozen teachers, imported weights or generated text as training targets.
The [complete report](../reports/live-reading-replay.json) and
[summary](../reports/live-reading-summary.json) retain every endpoint and seed.

## Equal new-source exposure

These are means across seeds 1337, 2026 and 31415. Both conditions observe the
same 500,571 new-source byte targets in 4,096 chunks and generate 768 bytes of
speech within their shared live state. Lower loss is better; binding accuracy
requires all four context/query reversals in a development group to be correct.

| Measurement | Ancestors | Replay every 4 observations | Replay every observation |
| --- | ---: | ---: | ---: |
| New-story training loss, nats/byte | 1.9257 | 0.4978 | 0.3943 |
| Unseen-story development loss, nats/byte | 1.9537 | 2.5943 | 2.3625 |
| Complete binding accuracy | 53.24% | 36.81% | 43.06% |
| Earlier reader loss | 2.2267 | 3.2841 | 2.8234 |
| Earlier geography loss | 1.8366 | 2.8130 | 2.3377 |
| Earlier narrative loss | 1.5792 | 2.5014 | 2.0555 |
| Additional optimizer updates | — | 5,120 | 8,192 |
| Replay target bytes | — | 103,947 | 413,311 |
| Cumulative native live-loop time | — | 14.16 s | 21.61 s |

More replay improves mean final binding by **6.25 percentage points** and reduces
the regression on every earlier book in every seed. Its measured loop cost is
about **52.6% higher**, with 60% more optimizer updates. These wall times include
replay, scheduled speech, logging and saves, but exclude process setup and
assessment. Other desktop GPU contexts were active. They do not establish an
intrinsic hardware speedup or a universal cost ratio.

| Seed | Ancestor binding | Every-4 final | Every-1 final | Every-1 improvement over every-4 |
| --- | ---: | ---: | ---: | ---: |
| 1337 | 70.83% | 61.81% | 66.67% | +4.86 pp |
| 2026 | 17.36% | 6.25% | 13.89% | +7.64 pp |
| 31415 | 71.53% | 42.36% | 48.61% | +6.25 pp |

Seed 1337 misses the declared five-percentage-point improvement requirement.
All three every-1 models finish with worse unseen-story loss than their own
ancestors, failing another requirement. Relative book/development cost margins
pass, but **none of the three seeds passes all four candidate conditions**.
The thresholds remain unchanged after observing the results.

## Equal optimizer-update counts

At 5,120 extra optimizer updates, compare every-4 after 4,096 source observations
with every-1 after 2,560. The latter has mean binding accuracy **49.77%**, versus
36.81%, and mean new development loss **2.1100**, versus 2.5943. All three seeds
have better binding and lower development/earlier-book losses in this comparison.
Every-1 still has worse new development loss than its ancestors.

This comparison is not equal compute or equal source exposure: every-1 observes
312,861 new-source targets, replays 258,552 targets and generates 480 bytes,
versus 500,571, 103,947 and 768 for every-4. Its measured loop time is 13.58 s,
versus 14.16 s. Different target counts, window lengths, speech and process
boundaries prevent interpreting this as a pure timing or hardware advantage.

## Early adaptation and actual language

Both mean development curves improve quickly. At 256 observations, new-story
development loss is **1.6781** with every-4 and **1.6908** with every-1, compared
with 1.9537 initially. Mean binding at that point is 43.98% and **55.09%**,
respectively. Continuing to 4,096 observations drives training loss much lower
while development loss rises. The ten stories provide only 14,182 distinct
within-document next-byte targets; the final condition presents approximately
35.3 passes' worth of those targets. The observed early endpoint is not an
independently validated stopping policy.

All twelve predeclared final story samples remain incoherent. A separate
[follow-up display](../reports/live-reading-questions.json) applies all six
questions/story prompts from the earlier speed demonstration to every final
model, using unchanged sampling settings and no retries. All six models fail
the arithmetic, sky-color and melting questions in these samples. Only the two
seed-1337 models begin both simple location reversals correctly; their following
prose still breaks down. This is illustrative output, not a comprehensive
question-answering evaluation. The [review](../reports/live-reading-sample-review.json)
records all 48 inspected outputs and their identities.

For example, seed 1337 with every-1 answers the two location questions with
`bag.` and `box.`, but its answer to `What is 2 + 2?` begins:

```text
box in the rod.

Wite do not me! xick clear, cleamps the priced parting
into the with holisheds they untitenig wateticles, and the snikle of the bananas.
```

Resident graph decoding averages approximately 6,672 byte tokens/s for every-4
and 6,637 for every-1 at the tested checkpoint shapes. These are averages of the
per-model seven-round measurements, excluding loading, capture and prompt
processing. Their similar speed does not establish equivalent language quality.

## Execution and numerical boundaries

The [execution audit](../reports/live-reading-execution.json) passes all 30
checkpoint assessments, covering 165 native commands and 30 zero-update
diagnostic commands. An independent structured reference reproduces all source
cursors, replay RNG, reservoir descriptors, per-stage counts and speech
RNG/counts exactly. Every assessment copy retains the live weights and Adam
moments byte for byte. Original ancestors, selected editions and executable
bytes remain unchanged; reserved tests are unused.

Both [smoke restart controls](../reports/live-reading-smoke-execution.json)
match uninterrupted runs across complete checkpoints and speech, including a
scheduled generation boundary. Additional [consistency checks](../reports/live-reading-consistency.json)
reproduce all three prior baseline reading assessments exactly, and the
seed-1337 every-4 final checkpoint matches the previous 4,096-observation
one-shot speed demonstration byte for byte.

The [independent CPU check](../reports/live-reading-learned-oracle.json) covers
four fixed development questions per final model. All **24 generated answers**
agree. Five models pass the unchanged score tolerance. Seed 2026 every-4 fails
at **0.0091638057**, versus the required `3e-5`. The test returns an unsuccessful
exit and the report preserves `passed: false`; this is not a full development
CPU audit.

A [read-only diagnosis](../reports/live-reading-score-diagnosis.json) reproduces
the native scores and localizes the largest discrepancy to zero-based layer 3,
byte 17, neuron 403. CPU membrane value **-0.9999998808** stays above the negative
threshold; native value **-1.0000005960** crosses it. Forcing that one CPU spike
to the native decision makes all subsequent traced spikes agree and reduces
answer-score error to **1.62e-6**. That intervention explains this case; it does
not convert the independent check into a pass or establish general CPU/CUDA
probability equivalence. The earlier independent cold-Adam tolerance failure
also remains unchanged; this study changes no native optimizer code.

Published JSON preserves parsed runtime data. The
[publication manifest](../reports/live-reading-publication.json) maps runtime
hashes to the LF-normalized public files, including the failed gate and failed
CPU score check. The protocol, execution checks and diagnostic are distinct
from evidence of useful language.

## Next research decision

At these settings, more rehearsal is insufficient for durable general learning.
The immediate next comparison should test selected-content breadth and bounded
repetition while retaining the replay controls and earlier-skill measurements.
This is an inference from the learning curves, not evidence that any larger
corpus or stopping rule will automatically fix the model. Adding neurons solely
because a model has accumulated more observations is not supported by this
study. General conversation and demonstrated generational improvement remain
open requirements.
