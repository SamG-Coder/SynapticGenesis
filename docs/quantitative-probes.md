# Quantitative development probes

This preparation adds original arithmetic and science questions for the
existing native `language-probes` evaluator. There are 128 items in 32 groups
covering addition, subtraction, multiplication, exact division, average
speed, net force, aligned-force work and constant-specific-heat heating.
These are development fixtures, not training text or a reserved test set.
No model has been scored on them.

Each numerical case contains two independent trials, two assignments of
facts to those trials, and a question about either trial. All four answers
must be correct to pass the group. Both candidate answers have equal byte
lengths and the same unit. Neither answer integer occurs among the visible
operands. The existing scorer records strict-FP32 candidate likelihoods,
context-erased scores, and unconstrained greedy output for the common
answer-length budget, while checking weights and optimizer state stay exact.

The [first-draft audit](../reports/quantitative-probes-prototype-extended-checks.json)
solves every answer independently with SymPy equations parsed from the
rendered question. It checks physical assumptions and units, verifies
crossed contexts/queries, and rejects 12 altered questions or contaminated
learning streams. The source guard finds no exact authored contexts or
trial statements in the three declared prose/arithmetic/Physics training
streams. It does not establish absence of conceptual overlap or paraphrases.

## First-draft limitation

Context-only, query-only and constant-choice controls pass zero full groups,
but this does not establish that the calculation itself is necessary.
Ranking the two trials by their largest operand, then associating that order
with the candidate answer order, passes **19/32 groups**. Other ranking
heuristics also exploit the selected numbers. Per-skill results expose
several complete failures of this numerical contrast design:

| Ranking heuristic | Correct groups | Joint accuracy |
| --- | ---: | ---: |
| First operand | 17/32 | 53.125% |
| Last operand | 13/32 | 40.625% |
| Largest operand | 19/32 | 59.375% |
| Smallest operand | 16/32 | 50% |

This first draft is retained as preparation evidence. Its numerical cases
need revision before native scoring; a successful choice between its two
candidates would not by itself establish arithmetic skill. Exact generated
numbers and units are a separate measure and still do not test explanations,
fractional arithmetic, general science or dialogue.

`scripts/quantitative_probes.py` owns the small fixture renderer and exact
text-overlap guard. `prepare_quantitative_probes.py` authenticates the selected
teaching editions and emits `SGPROBE2`, readable questions, answer witnesses
and a manifest. The implementation changes no native model computation.
`tests/quantitative_probes.py` independently solves displayed questions and
computes the shortcut baselines.

All work here is host-side preparation. The active GPU objective study is
unchanged and no extra CUDA work has been started. The first prepared copy
is `data/prepared/quantitative-development-v1` in the main working directory;
its source and code are from `research/quantitative-probes`.
