# Broader reading improves text loss; skill retention still fails

The [declared native comparison](reading-breadth-experiment.md) is complete:
three learned ancestors, four conditions and 36 continuation endpoints. The
44-story selection improves unseen-story prediction and preserves earlier
reading better than repeating ten stories. It does not reliably retain the
earlier question-binding skill. **Neither replay cadence passes the candidate
rule, and no seed passes all requirements at the primary endpoint.** Generated
prose remains incoherent. No runtime, growth or population default changes.

All model computation uses the existing C++/CUDA path. Each model has 1,951,624
parameters and continues its full live checkpoint, including weights, Adam
history, recurrence and replay memory. The new content is the audited
[reading expansion](reading-breadth.md); no external trained weights or generated
text supply training targets. The [complete results](../reports/reading-breadth-comparison.json)
and [summary](../reports/reading-breadth-summary.json) preserve all seeds and endpoints.

## Primary result after one broader pass

These are three-seed means after 546 additional observations. The starter arms
see 66,704 new-source byte targets, about 4.7 repetitions of their ten stories.
The broader arms see 67,188 targets, one pass over 44 stories. The 484-target
difference is about 0.73% of the starter exposure. Within each replay cadence,
optimizer update counts and scheduled speech are equal. Content, breadth,
repetition and order after the first ten stories change together.

Loss is in nats per byte; lower is better. Binding requires all four answers
across a development group's changed facts and queries to be correct.

| Measurement | Ancestors | 10 stories, replay every 4 | 44 stories, replay every 4 | 10 stories, replay every 1 | 44 stories, replay every 1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Starter development loss | 1.9537 | 1.7337 | 1.7147 | 1.7461 | 1.7008 |
| Expansion development loss | 2.0050 | 2.0062 | 1.8006 | 1.9902 | 1.7991 |
| Complete binding accuracy | 53.24% | 37.50% | 34.95% | 56.71% | 48.84% |
| Earlier reader loss | 2.2267 | 2.3009 | 2.1734 | 2.2684 | 2.1439 |
| Earlier geography loss | 1.8366 | 1.9349 | 1.8399 | 1.9048 | 1.8139 |
| Earlier narrative loss | 1.5792 | 1.6915 | 1.6078 | 1.6515 | 1.5788 |
| Native live-loop seconds | — | 2.045 | 2.111 | 3.061 | 3.136 |

All six broader conditions improve both development-story losses relative to
their ancestors. They also improve all 18 paired earlier-book comparisons over
their matching starter controls. Five of six improve both development sets over
the starter control; seed 31415 with replay every four has a small starter-set
regression of 0.00161 nats/byte. These benefits coexist with failures to retain
binding. The table below lists every failed part of the predeclared gate; all
unlisted parts pass.

| Seed | Replay every | Failed requirements at 546 observations |
| --- | ---: | --- |
| 1337 | 4 | Binding falls more than 5 percentage points from its ancestor and more than 2.5 points below starter |
| 2026 | 4 | Binding falls more than 5 points from its ancestor |
| 31415 | 4 | Starter development loss exceeds starter control; binding falls more than 5 points from its ancestor |
| 1337 | 1 | Binding falls more than 2.5 points below starter |
| 2026 | 1 | Binding falls more than 5 points from its ancestor |
| 31415 | 1 | Binding falls more than 5 points from its ancestor and more than 2.5 points below starter |

The every-one broader mean is only 4.40 points below its ancestral binding
score, but a mean cannot satisfy a rule that requires every seed to pass.
It also trails the matching starter mean by 7.87 points. More content is a
measured reading benefit at this setting, not evidence that skill retention is
solved.

## Fixed continuation to two broader passes

The final endpoint is 1,092 additional observations, fixed before training.
It presents 134,376 new targets in the broader arms and 133,317 in the starter
arms. This is two broader passes versus about 9.4 starter passes, rather than
equal numbers of passes.

| Final measurement | 10 stories, every 4 | 44 stories, every 4 | 10 stories, every 1 | 44 stories, every 1 |
| --- | ---: | ---: | ---: | ---: |
| Starter development loss | 1.8218 | 1.6903 | 1.8031 | 1.6993 |
| Expansion development loss | 2.1457 | 1.7868 | 2.0797 | 1.8048 |
| Complete binding accuracy | 43.52% | 45.37% | 44.68% | 41.20% |
| Earlier reader loss | 2.4337 | 2.1849 | 2.3890 | 2.1680 |
| Earlier geography loss | 2.0513 | 1.8484 | 1.9895 | 1.8288 |
| Earlier narrative loss | 1.8138 | 1.6131 | 1.7304 | 1.5859 |

Broader reading keeps the mean development losses below their starting values
while the starter arms' expansion-set loss worsens. Final binding remains below
the ancestral mean in every arm. Its response to broader content varies with
seed and replay cadence. This endpoint does not replace the failed primary gate
or establish a validated stopping policy.

## Speed on the RTX 5080

The [performance report](../reports/reading-breadth-performance.json) preserves
all per-model decode rates and exact live target counts. Tokens here are
**bytes**, using a fixed 256-entry vocabulary. They are not word-piece tokens.

| Final condition | Graph generation, byte tokens/s | Regular generation, byte tokens/s | Live new-source targets/s | Live source + replay targets/s | Mean live seconds for 1,092 observations |
| --- | ---: | ---: | ---: | ---: | ---: |
| 10 stories, replay every 4 | 6,575 | 1,982 | 33,181 | 40,225 | 4.018 |
| 44 stories, replay every 4 | 6,507 | 1,990 | 32,129 | 38,926 | 4.182 |
| 10 stories, replay every 1 | 6,559 | 2,007 | 22,169 | 40,567 | 6.014 |
| 44 stories, replay every 1 | 6,613 | 2,008 | 21,694 | 39,695 | 6.194 |

For each model, decoding uses the median duration of seven 512-byte rounds;
the table averages the three resulting rates. Per-model graph rates span
6,424–6,682 byte tokens/s. Graph and regular output bytes match in all twelve
comparisons, with zero recorded state/logit error. Decode includes sampling,
transfers and synchronization, but excludes model loading, roughly 40 ms graph
capture, prompt processing and output I/O. These are resident generation rates.

Live throughput pools target counts over measured loop seconds. It includes
learning, replay, two 96-byte speech events, logging and saves, while excluding
startup and assessment. It covers three resumed segments per model, not the
whole experiment's elapsed time. Every-four arms make 1,365 optimizer updates;
every-one arms make 2,184. More rehearsal therefore reduces the rate of learning
from new text while processing more old targets. Other desktop GPU contexts
remain active; these figures do not establish an energy advantage or a causal
quality benefit from small speed differences.

## Actual question and story output

All 24 fixed final story samples and all 72 follow-up samples are retained.
The [question report](../reports/reading-breadth-questions.json) uses the same six
prompts from the earlier demonstration on every final model, with no retries:
192 generated bytes, temperature 0.8, top-k 40, seed 42 and fresh context.
These models have not been instruction tuned; the outputs are continuations,
not a comprehensive question-answering benchmark.

The following are exact beginnings from seed 1337, broader reading with replay
every observation. This is the first declared seed for that condition, not a
selection by answer quality. The complete continuations remain in the report.

| Prompt | Output beginning |
| --- | --- |
| `The toy is in the bag. The tag is in the box.` / `Where is the toy?` / `Answer:` | `bag.` then `And is box.` |
| `The toy is in the box. The tag is in the bag.` / `Where is the toy?` / `Answer:` | `box.` then `Answer: box.` |
| `Question: What is 2 + 2?` / `Answer:` | `bed.` then `It was did no the room, but onele stil is of` |
| `Question: What color is the sky?` / `Answer:` | `bed.` then `Now, I last bles, still will, I'll sadp,` |
| `Question: Why does ice melt?` / `Answer:` | `bed, I canner the Rabbing bannansl sure watter wistaring,` |
| `Once upon a time ` | `all sure with the Pigenly brack the of smelalon` |

Six of twelve models begin both location reversals with the expected word.
Later text is still incoherent and can contradict that beginning. All twelve
fail the arithmetic, sky-color and melting prompts in these samples, and all
24 fixed story samples remain incoherent. The [manual review](../reports/reading-breadth-sample-review.json)
records all 96 inspected outputs and their hashes. Faster generation and lower
text loss have not yet produced useful conversation.

## Execution and numerical verification

The [execution audit](../reports/reading-breadth-execution.json) passes all 39
checkpoint assessments, covering 228 production-native commands and 39
zero-update diagnostic commands. An independent reference reproduces source
cursors, replay RNG and descriptors, per-stage counts and speech history.
All six shared-prefix pairs have identical learned payloads after the first
116 observations. All three earlier starter baseline assessments reproduce
exactly. Ancestors and selected editions remain unchanged; reserved tests
remain unused.

All four [smoke controls](../reports/reading-breadth-smoke-execution.json)
match uninterrupted native runs across the full checkpoint and speech,
including the scheduled 500-observation generation boundary. Those controls
verify continuation behavior, not the learning gate.

The [fixed-group independent CPU check](../reports/reading-breadth-learned-oracle.json)
reproduces all 48 greedy answers. Eleven of twelve models pass the unchanged
`3e-5` score tolerance. Seed 2026, broader reading, every-one replay fails at
**0.0373044906**; its test exits unsuccessfully and retains `passed: false`.
This is four questions per model, not a full independent development audit.

The [read-only diagnosis](../reports/reading-breadth-score-diagnosis.json)
reproduces the failed native score and locates the first different spike at
zero-based layer 2, byte 9, neuron 20. CPU membrane 1.0000003576 crosses the
positive threshold; native 0.9999994636 does not. Forcing that one CPU decision
to the native value makes subsequent traced spikes match and reduces this
answer-score discrepancy to 1.06e-6. This explains the selected discrepancy;
it does not convert the independent test into a pass. The older cold-Adam
tolerance failure also remains unchanged; this study changes no native code.

The [publication manifest](../reports/reading-breadth-publication.json) maps
retained runtime hashes to the LF-normalized public reports and verifies parsed
JSON equality. Failed retention gates, failed numerical checks and raw outputs
are published together with the reading improvements and speed measurements.

The next research question is how to allocate the existing rehearsal budget
without losing earlier skills. A [primary-source review](replay-priority-research.md)
separates weak learning, actual update-induced interference and future usefulness,
and proposes a bounded diagnostic before any selector changes. It is a research
direction, not a demonstrated improvement or reason to promote reproduction.
