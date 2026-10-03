# Quantitative development probes

This preparation adds original arithmetic and science questions for the
existing native `language-probes` evaluator. There are 128 items in 32 groups
covering addition, subtraction, multiplication, exact division, average
speed, net force, aligned-force work and constant-specific-heat heating.
The current edition is `quantitative-development-v2`. These are development
fixtures, not training text or a reserved test set. No model has been scored
on them; native evaluation remains pending while the objective study uses
the GPU.

Each numerical case contains two independent trials, two assignments of
facts to those trials, and a question about either trial. All four answers
must be correct to pass the group. Both candidate answers have equal byte
lengths and the same unit. Neither answer integer occurs among the visible
operands. The existing scorer records strict-FP32 candidate likelihoods,
context-erased scores, and unconstrained greedy output for the common
answer-length budget, while checking weights and optimizer state stay exact.

The [v2 audit](../reports/quantitative-probes-v2-checks.json)
solves every answer independently with SymPy equations parsed from the
rendered question. It checks physical assumptions and units, verifies
crossed contexts/queries, and rejects 12 altered questions or contaminated
learning streams. The source guard finds no exact authored contexts or
trial statements in the three declared prose/arithmetic/Physics training
streams. It does not establish absence of conceptual overlap or paraphrases.
All prompts are ASCII and no longer than 345 bytes. The final native probe
SHA-256 is
`5c4c4203f20c166322f10cd0e4bb6f359d8058f430893f8f46d88ea6c62f18d4`.

For example, one net-force group gives trial A a mass of 11 kg and an
acceleration magnitude of 7 m/s^2, and trial B 13 kg and 5 m/s^2. The expected
answers are 77 N and 65 N. Swapping those complete facts must reverse the
answers to both queries. This example is a declared answer key, not output
from a trained model. The context explicitly fixes mass and uses an inertial
frame. Work questions specify a constant force parallel to displacement in
the same direction; heat questions specify constant specific heat, no phase
change and negligible work/heat loss.

## Numerical contrast revision

The first draft exposed an operand-ranking shortcut before any model was
scored. A deterministic revision selects new numbers against four declared
heuristics. The [selection evidence](../reports/quantitative-case-selection.json)
records seed 20261004, the original specification, 20,000 candidate draws per
skill and each selected ordering pattern. Exact equations determine every
label; no model response participates in selection.

| Control | First draft: correct groups | V2: correct groups |
| --- | ---: | ---: |
| Context only | 0/32 | 0/32 |
| Query only | 0/32 | 0/32 |
| Constant choice | 0/32 | 0/32 |
| Rank by first operand | 17/32 | 16/32 |
| Rank by last operand | 13/32 | 16/32 |
| Rank by largest operand | 19/32 | 16/32 |
| Rank by smallest operand | 16/32 | 16/32 |

In v2 each ranking heuristic passes exactly two of the four groups in every
skill; none passes all four as several did in the original draft. The
independent audit derives those results from the displayed facts. Ranking
controls can bind the correct named trial without carrying out its requested
operation, so context dependence alone is insufficient evidence of numerical
reasoning. Stronger strategies and undiscovered shortcuts remain possible.
The chosen 32 groups are a small development diagnostic with fixed wording,
not a broad arithmetic/science benchmark or independent confirmation set.

Report forced-choice accuracy, all-four-in-group accuracy, context-erased
controls and exact greedy numeric/unit output separately, retaining every
item and per-skill result. A 50% group score is attainable by the declared
ranking heuristics. A nominal independent-coin probability is not an adequate
baseline for this structured task. Unit correctness in the fixtures also
does not mean forced-choice scoring tests unit selection: both candidates
share the requested unit.

## Preserved prototype and implementation

This first draft and its
[audit](../reports/quantitative-probes-prototype-extended-checks.json) are
retained as preparation evidence; v2 supersedes it for future native scoring.
A successful choice between its two candidates would not by itself establish
arithmetic skill. Exact generated numbers and units are a separate measure
and still do not test explanations,
fractional arithmetic, general science or dialogue.

`scripts/quantitative_probes.py` owns the small fixture renderer and exact
text-overlap guard. `prepare_quantitative_probes.py` authenticates the selected
teaching editions and emits `SGPROBE2`, readable questions, answer witnesses
and a manifest. The implementation changes no native model computation.
`tests/quantitative_probes.py` independently solves displayed questions and
computes the shortcut baselines.

`scripts/select_quantitative_cases.py` owns the deterministic numerical
contrast selection. The displayed-question audit is independent of that
selector and the renderer's rational arithmetic. All work here is host-side
preparation. The active GPU objective study is unchanged and no extra CUDA
work has been started.

The final prepared copy is `data/prepared/quantitative-development-v2` in
the main working directory; its source and code are from
`research/quantitative-probes`, which includes the corrected Physics branch.
The original v1 copy remains separately preserved. Neither suite is admitted
as training, a teacher, a parent or a reproduction gate.

## Next assessment and reproduction

When the GPU study finishes, score the frozen prose checkpoints before
lesson continuation, then use exactly the same suite after a declared
continuation. Record checkpoint, executable and suite hashes and retain
every raw result. No checkpoint has been selected here and no scoring job
has been launched or queued. Do not infer a model-quality result from the
host-side checks above.

Future learning inputs must pass `quantitative_probes.protect` for these
full contexts and trial statements before launch. The manifest's
`protected_input` also supplies a native probe record compatible with the
existing full-context reservation mechanism. These checks currently cover
the three declared training streams; they do not automatically modify
every earlier source manifest or admit a new training corpus.

With the pinned prose, arithmetic and corrected Physics preparations
available, use fresh output/report paths from this branch:

```powershell
python -X utf8 scripts/prepare_quantitative_probes.py --out data/prepared/quantitative-development-v2-rebuilt
python -X utf8 tests/quantitative_probes.py --prepared data/prepared/quantitative-development-v2-rebuilt --report reports/quantitative-probes-local-checks.json
```

The existing native command is `language-probes --checkpoint <checkpoint>
--probes <prepared>/development.sgprobe --output <fresh-result.json>`.
It performs model computation in C++/CUDA using the common forward path;
the Python fixtures and independent answer solver do not answer on behalf
of the model.
