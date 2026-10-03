# Exact restart control for the 105M learner

The resumed original-rate control reproduced the original 105M final checkpoint
byte for byte. The [published record](../reports/prose-retention-control.json)
verifies 38 completed native commands, all three sets of raw assessments and
the two resumed learning sessions. This establishes a reproducible control for
the ongoing lower-rate comparison. It does not establish improved learning.

Both final checkpoint files have SHA-256:

```text
f68c58fde2e073d33fb155530517a8c44f79280d9d54adcf21751b55eabc4501
```

The native C++/CUDA model has 104,851,472 parameters and 32,768 spiking neurons.
The control resumed the same stage-three checkpoint at 95,207 source
observations, learned to 155,748, restarted again, and finished at 216,289.
It used the original learning rate of 0.0003 and the unchanged source,
replay and scheduled-generation policies. The final file includes weights,
Adam moments, recurrent state, counters, source cursor, replay descriptors
and random-number state. Their exact final identity shows that these changed
process boundaries preserved the saved result on this run and hardware.

## Prediction and retention

Each assessment uses the same 65,536 sampled target bytes per book through
independent 128-byte windows, in strict FP32 with reset recurrence. Loss is
in nats per target byte; lower is better. The validation mean gives the four
validation books equal weight. The other rows measure fit to training books.

| Measurement | Parent: 95,207 observations | Halfway: 155,748 | Final: 216,289 |
| --- | ---: | ---: | ---: |
| Four-book validation mean | 1.804779 | 1.842210 | 1.967449 |
| McGuffey's Eclectic Primer, early training monitor | 1.729837 | 1.835966 | 1.932688 |
| McGuffey's First Eclectic Reader, early training monitor | 1.681960 | 1.767095 | 1.906738 |
| Book 43, first book in the final source stage | 1.655413 | 1.575266 | 1.657562 |
| Book 1257, last book in the final source stage | 1.743082 | 1.709722 | 1.325197 |

The final-stage validation regression is reproduced. The early readers worsen
at both measurements, while the last new-stage training book improves. The
first new-stage training book improves halfway and ends slightly worse than
its parent score. These are prediction measurements on fixed sampled text,
not counts of forgotten facts or proof of a particular biological mechanism.

The fresh parent and final assessments exactly match their previously
published six book scores and four raw continuations. The two added training
monitors are retained separately. All four generated outputs at each of the
three points are included, with their raw bytes and file hashes; the
[original completion report](prose-105m-complete.md) shows the final question
examples and their failure to answer correctly.

## Speed and verification

The two resumed sessions took 1,028.65 and 1,033.98 seconds, reporting
7,532.85 and 7,493.92 source byte targets per second. These times include
source learning, replay, scheduled speech and checkpoint saves, but exclude
startup/loading and the intervening assessments. They are sequential timings
of one run, not a controlled comparison of model sizes or learning rates.

`scripts/publish_retention_control.py` authenticates the declared inputs,
recomputes checkpoint and replay-payload identities, reconstructs the exact
38-command prefix, and checks the recorded native sessions, raw book scores,
sample bytes and logs. The prefix contains two learning commands, 24 book
evaluations and 12 generation commands. Publication itself makes no native
model calls. To reproduce the audit, preserve the experiment assets and use
a fresh destination:

```powershell
python scripts/publish_retention_control.py --out runs/retention-control-audit.json
```

The paired arm starts from the same parent with a rate of 0.000075. It was
allowed to start only after the exact-control check passed. Its outcome is
not included in this snapshot; the subsequent
[complete two-rate report](prose-retention-rates.md) records its worse outcome.
Lowering the rate also lowers the effective
AdamW decay step, so that comparison will not isolate gradient plasticity
from weight decay. No reserved test books are scored and no model is promoted
for teaching or reproduction by this report.
