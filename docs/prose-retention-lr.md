# Late-stage learning-rate comparison

The completed [105M run](prose-105m-complete.md) improved its four-book validation
mean through stage three, then worsened on every validation book and both early
training readers during stage four. Its mean rose from 1.80477925 to 1.9674495
nats per byte. A larger parameter count has not, by itself, solved continual
learning. This experiment tests one change: a lower learning rate for the final
source stage.

The [specification](../data/prose-retention-lr-v1.json) is exploratory and was
chosen after seeing that regression. It is not a blinded confirmation or a
biological ageing model. The [completed native comparison](prose-retention-rates.md)
now shows worse results for the lower rate on all eight book monitors at both
measured endpoints. The default rate is unchanged.

Both arms resume the same admitted 104,851,472-parameter, 32,768-neuron associative
checkpoint at 95,207 source observations. They use the original complete-book
schedule, the same remaining 15,497,256 source targets, and the inherited native
state. The original 55 training books contain 4,817,047 word occurrences. Word
occurrences, byte targets and parameter counts are different quantities.

| Arm | Learning rate | Halfway endpoint | Final endpoint |
| --- | ---: | ---: | ---: |
| Resumed control | 0.0003 | 155,748 observations | 216,289 observations |
| Quarter rate | 0.000075 | 155,748 observations | 216,289 observations |

Each arm runs in two native sessions. Adam moments, recurrent state, replay
descriptors, random-number state, source cursor and scheduled speech survive
the restart. `live --resume --lr` changes the base/effective rate without an
optimizer reset. In this AdamW implementation, that also changes the effective
decoupled weight-decay step. It does not isolate gradient plasticity from decay.

Before the quarter-rate arm may begin, the completed resumed control must match
the original final checkpoint's SHA-256 exactly:
`f68c58fde2e073d33fb155530517a8c44f79280d9d54adcf21751b55eabc4501`.
A mismatch stops the experiment and preserves the failure. This verifies the
actual 105M restart through the changed process boundaries; smaller fixture
tests alone would not establish that property.

At the shared parent, halfway point and final point, the study records:

- All four original validation losses and their equal-book mean.
- Both original early-reader training losses.
- Two additional training losses: book 43 and book 1257, selected mechanically
  as the first and last books in the admitted fourth stage. These measure fit
  to new training material and are not held-out generalization scores.
- All four original raw generations with unchanged prompts, sampling settings,
  output bytes and SHA-256 digests.
- Source/replay/speech counters, replay payload identity, source cursor and RNG,
  native session time and complete process time.

The parent assessment's original six book results and four generated samples
must also match the published assessment exactly. Between arms, each endpoint
must have identical exposure and replay state. Improving early-reader scores
while failing to learn the new books will be reported as a tradeoff. No model
is automatically selected for teaching or reproduction. Reserved tests remain
unscored, and the prepared astronomy curriculum is not used here.

The driver waits behind the existing matched neuron panel using a verified
Windows process handle. After that process succeeds, it requires the completed
three-size comparison and all 73 neuron-panel native calls, including the old
trace compatibility check. It authenticates inputs again after waiting and
after each learning/assessment segment. This avoids concurrent GPU experiments
and detects changes to declared code, sources and checkpoints.

## Implementation and verification

`scripts/checkpoint_assessment.py` now owns the shared native book and raw-sample
measurements. The original size assessment calls it without extra books. The
new study supplies its two training monitors. Native C++/CUDA still owns all
model computation and the shared training/inference forward path.

`scripts/prose_retention_inputs.py` owns scoped source/parent admission, bounded
checkpoint inspection and exposure checks. `scripts/prose_retention_lr.py`
owns declaration, serialization, execution and paired differences. The
diagnostic checkpoint reader seeks past the large model arrays instead of
loading the entire checkpoint into Python; it does not replace native checksum
and layout validation.

The [assessment extraction audit](../reports/checkpoint-assessment-extraction.json)
replayed 20 existing native command outputs on CPU, preserving every model/input
and sampling argument and every resulting score/sample field. It rejected
duplicate book roles and malformed generation output. These were archived
commands, not 20 fresh CUDA calls.

The [continuation preflight](../reports/prose-retention-lr-preflight.json)
authenticated the real parent and original completion, independently walked
all final-stage source windows, matched native completion counters and rejected
13 altered/incomplete cases. It verified that an incomplete predecessor fails
before constructing a native command runner. It performed no CUDA work. The
full checkpoint control and lower-rate learning have now completed. The
[publication audit](../reports/prose-retention-rates-audit.json) verifies the
exact control, all 64 native commands and matched exposure; the candidate's
quality regression is retained in the complete result.

Development uses an isolated checkout while the older assessment and trace
runners still depend on their declared code. Launch its script by absolute
path with `D:\SynapticGenesis` as the working directory, then keep both its code
and the main source/checkpoint assets unchanged until the study finishes. The
protocol records the source checkout, Git commit and file hashes. A completed
study contains 60 read-only assessment calls and four native learning calls.

Native session timing includes source learning, replay, speech and saves;
complete process timing additionally includes loading and startup. Assessments
are timed outside the learning sessions. Sequential rates on one seed do not
provide a controlled speed benchmark. Short, reset evaluation windows also do
not establish long-context memory or useful question answering.
