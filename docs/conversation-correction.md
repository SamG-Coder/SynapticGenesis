# Direct conversation and correction

The completed prose model has learned book continuation but has not demonstrated
useful question answering. This experiment asks it short questions, records its
actual replies, gives six reviewed corrections, continues learning through the
native live loop, and asks again with fresh recurrent context.

## Fixed teaching session

The starting model is the complete 105M parameter prose checkpoint, seed 1337,
after 27,680,129 source byte pairs. Its SHA-256 is
`c45f5d962112f73fe3ad2688f61f6689f6e9f10aa775ff6dfc38c62c241fe2a5`.
Its selected book ancestry is preserved. The correction edition and this
diagnostic continuation are explicitly admitted in `data/training-selection.json`.

The six corrections cover a greeting, the name SynapticGenesis, two plus three,
five minus two, hydrogen and oxygen in water, and acknowledging that the contents
of a closed cupboard are unknown. These are original short examples, reviewed
before any new model response. Model-generated mistakes are retained as evidence;
the training stream contains only the selected questions and correct answers.

| Checkpoint | Direct passes through the six corrections | Added source updates | Added source byte pairs |
|---|---:|---:|---:|
| Before correction | 0 | 0 | 0 |
| First exposure | 1 | 6 | 416 |
| Practice | 16 | 96 | 6,656 |
| Further practice | 256 | 1,536 | 106,496 |

These are cumulative direct exposures. Existing replay runs every four source
updates and may revisit a correction; its exposures are reported separately.
The first exposure includes one replay update, selected from the inherited
memory or corrections already observed. Later repetition is deliberate and
must not be described as learning immediately from a single correction.

The new stage uses one quarter of the inherited base learning rate, approximately
0.000075, and eightfold relative answer-byte emphasis. The native objective
normalizes the weighted loss. Each correction fits one 128-byte source window.
`--resume` and `--extend-curriculum` retain weights, optimizer history, replay,
RNG and learning counters. A normal document boundary resets source recurrence.
The ancestor checkpoint is never overwritten.

## What is measured

Every endpoint generates 96 byte tokens for each of 18 questions:

- Six questions that receive corrections.
- Six rewordings excluded from the correction targets.
- Four arithmetic questions using new operands.
- Two facts not practised in this correction session.

All replies are retained. Greedy generation uses a fixed seed, top-k 1 and the
existing graph sampler. A narrow automated check compares the first nonblank
answer line with a declared list after case and terminal punctuation
normalization. The rest of the continuation still needs inspection: a correct
first line followed by nonsense is not evidence of good dialogue. An exact
match to a practised answer does not establish arithmetic or generalization.

A separate six-question control puts a correct answer in the prompt without any
weight updates. Post-learning checks reload the checkpoint and start with fresh
context, so prompt copying and durable learning are measured separately. This
first experiment is a scripted teaching conversation with checkpoint reloads
between measurement and learning rounds. It does not yet add a continuously
resident interactive terminal.

Four existing prose books are scored before and after correction: two validation
books and two earlier training books, with 65,536 target bytes per book, fixed
sampling and strict FP32 evaluation. Loss changes measure a limited retention
cost. No reserved test books are scored. The authored quantitative development
suite's full contexts and unique trial statements are checked against the actual
new learning stream. These exact-text checks do not establish semantic isolation.

Question generation timing includes loading, GPU setup and prompt processing.
It must be reported as process-level generation cost, not isolated decode speed.
Native learning sessions separately report observed byte pairs per second.

## Execution and evidence

`scripts/conversation_material.py` owns selection, target isolation and the
append-only curriculum. `scripts/conversation_measure.py` owns raw answers,
first-line scoring and book retention measurements.
`scripts/conversation_experiment.py` owns the serialized ask/correct/retest loop
and inherited-state checks. All model computation uses the existing native
C++/CUDA executable and shared model implementation.

The driver first waits on the held process handle for the quantitative baseline
assessment. That assessment itself waits for the full membrane study. A failed
or incomplete predecessor blocks CUDA launch. Each declaration authenticates
its inputs; evidence, logs and failures remain in the run directory.

The [host checks](../reports/conversation-correction-host-checks.json) passed 16
rejection controls, checked the real base/source provenance, and exercised raw
reply preservation with 22 synthetic native calls. They launched zero native
commands and do not establish whether the model learns these corrections.
Actual before/after replies and retention results will be published after the
serialized GPU experiment completes. No diagnostic descendant is admitted as
a teacher or parent.
