# Live reading and rehearsal: declared comparison

This protocol follows the completed [reading-adaptation study](reading-adaptation.md).
That diagnostic established quick initial adaptation followed by overfitting and
forgetting on the small selected early-reader edition. It intentionally omitted
live recurrent history and replay. This study continues complete native live
histories and compares two existing rehearsal budgets. It does not introduce a
new forward implementation or claim to reproduce biological replay.

## Conditions fixed before training

Use the ordinary-replay associative checkpoints at 190,000 observations from the
[teacher-retention comparison](teacher-retention.md), seeds 1337, 2026 and 31415.
The architectures, weights, Adam history, recurrent state, speech RNG, source
history and grouped replay reservoirs remain intact. Each model has 1,951,624
parameters. No frozen teacher or synaptic-importance penalty is active.

Append the ten selected early-reader training stories as a new curriculum stage.
The stage repeats only that new source suffix, in complete document order with
128-target chunks and shorter document tails. Training uses the inherited TF32
setting, fixed-order gradient reductions and learning rate 0.000075. Sources,
original checkpoints, executable bytes and relevant scripts are hashed before
execution. No reserved tests are evaluated.

| Condition | Additional reading observations | Replay |
| --- | --- | --- |
| `every-4` | 64, 256, 1,024, 4,096 | One previous window every fourth observation |
| `every-1` | 64, 256, 1,024, 2,560, 4,096 | One previous window after every observation |

Both retain 1,024 reservoir descriptors distributed across five source groups.
The new group fills from observations; earlier groups shrink uniformly to their
new quotas while keeping lifetime history. Replay selection precedes insertion
of the current source window. Generated text supplies no training targets.
The existing live speech policy generates 96 bytes every 500 observations with
the fixed prompt `The bird `, sharing weights and recurrent state with reading.

The primary comparison uses equal new-source exposure: 4,096 observations.
The secondary comparison uses equal optimizer counts: `every-4` at 4,096 versus
`every-1` at 2,560, both 5,120 additional updates. Equal optimizer count does not
equal target bytes, computation, speech exposure or wall time. Cadence changes
the quantity and identities of replayed windows, so this does not isolate timing
alone. Actual counters and cost accompany both comparisons.

## Measurements and candidate rule

At the ancestor and every endpoint, score all within-document train/development
byte pairs from the early-reader edition in the existing strict-FP32 diagnostic,
resetting recurrence for each 128-target window and exact shorter tail. Use a
zero-update copy that preserves weights and Adam moments without replacing live
state. Also score the existing binding development questions and the three
earlier books with 32 batches of 16 sequences of 128 targets. Assessments leave
the learning checkpoint unchanged. Final checkpoints receive the two fixed
384-byte samples and a seven-round, 512-byte native decode measurement.

At equal final reading exposure, `every-1` must meet all four requirements in
**each of the three seeds** to qualify as a candidate:

1. Improve complete binding development accuracy by at least five percentage points.
2. Keep new-story development loss within +0.03 nats/byte of `every-4`.
3. Keep each earlier book loss within +0.02 nats/byte of `every-4`.
4. Improve new-story development loss from its own ancestor.

Also report the entire equal-optimizer-count comparison. A passing candidate is
not automatic population promotion or proof of general conversation, lifelong
learning or a benefit of biological age. The reused development probes, small
edition and three related ancestors constrain the conclusion.

Native sessions are sequential, alternating condition order by seed. Timings
include the learning loop, replay, speech, logging and saving; held-out
assessments and process startup are outside native loop time. More endpoints
in `every-1` add extra starts and saves. Other desktop GPU contexts may be active.

## Verification and execution

An independent structured Python reference reproduces source cursors, byte
counts, replay RNG, reservoir contents, per-stage counters and speech RNG/counts.
It performs no neural computation. The smoke run uses the real seed-1337 parent
with endpoints 16, 64 and 512, plus 320 for the matched-count condition. Both
conditions must match uninterrupted 512-observation runs byte for byte across
the complete checkpoint and generated speech, including the 500-observation
generation boundary. Evaluation copies must retain exact weights and moments.

```powershell
python scripts/live_reading_experiment.py --smoke --out runs/live-reading-smoke
python tests/live_reading_experiment.py --root runs/live-reading-smoke --continuation-control
python scripts/summarize_live_reading.py --root runs/live-reading-smoke

python scripts/live_reading_experiment.py --out runs/live-reading-panel
python tests/live_reading_experiment.py --root runs/live-reading-panel
python tests/binding_learned_oracle.py --root runs/live-reading-panel
python scripts/summarize_live_reading.py --root runs/live-reading-panel
```

The existing fixed-group CPU oracle is an additional numerical check, not a new
quality benchmark. Any spike-threshold tolerance failure remains visible.
Implementation and execution verification must be completed before interpreting
the learning comparison. The production C++/CUDA implementation is unchanged.
