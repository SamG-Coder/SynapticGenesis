# Adding physical science to the live curriculum

The reviewed [astronomy edition](astronomy-corpus.md) now has an explicit
append-only exposure schedule after the four complete prose stages. This
preparation does not select a checkpoint or start learning. The active model
size comparison and read-only neuron panel keep their original inputs.

The addition contains 174 training modules and 466,756 word occurrences. One
complete pass of every module introduces 2,832,810 next-byte targets across
22,227 source observations of at most 128 targets each. Targets never cross a
document separator. Validation and test modules are excluded from the schedule.

| New stage | Topic | Modules | Source observations | Cumulative endpoint |
| ---: | --- | ---: | ---: | ---: |
| 5 | Observing, motion, light and instruments | 44 | 4,607 | 220,896 |
| 6 | Planets, moons and the solar system | 45 | 5,769 | 226,665 |
| 7 | The Sun, stars and stellar evolution | 49 | 6,474 | 233,139 |
| 8 | Galaxies and the universe | 36 | 5,377 | 238,516 |

Stages one through four retain their source bytes, 216,289 final observation
count, learning-rate scales, traversal scopes and answer weights. Each new
stage also uses rate scale one, `new` source scope and ordinary byte-loss
weight one. This is a first exposure schedule, not a demonstrated optimum or
an assignment of biological ages to textbook topics.

The preparer authenticates both admitted editions, the published astronomy
preparation audit and the previous prose schedule. It rechecks each astronomy
document, split and topic file against their recorded identities. It then uses
the existing [curriculum extension helper](../scripts/extend_curriculum.py)
to produce four successive self-contained editions. The final schedule is
`runs/prose-astronomy-curriculum/after-astronomy-4/curriculum.sg`.
The source license, attribution and original preface are copied beside it.

Before writing an edition, the preparer checks the entire proposed learning
text against both corpora's validation and test documents. Exact documents
and normalized held-out paragraphs of at least 120 characters are rejected.
This does not exclude paraphrases, shorter overlap or shared subject matter.
Reading hashes and checking separation does not score reserved test content.

The [CPU audit](../reports/astronomy-curriculum.json) passed:

- Byte-for-byte preservation of all four existing prose editions and policies.
- Reconstruction of all four cumulative additions from the 174 selected
  training modules, each introduced exactly once.
- Independent enumeration of every target window, including each document's
  final partial window, matching the declared observation and target counts.
- Zero exact held-out document or substantial-paragraph matches in the
  final combined learning corpus; unchanged source attribution files.
- Five rejections: altered prepared document, altered topic file, exact
  held-out document, embedded held-out paragraph, and that paragraph with
  changed whitespace. Altered preparations produce no output edition.

```powershell
python scripts/astronomy_schedule.py --out runs/prose-astronomy-curriculum
python tests/astronomy_schedule.py --out runs/astronomy-schedule-check --report reports/astronomy-curriculum.json
```

These commands require fresh output directories. Native admission of this
particular eight-stage edition is still pending; no model has learned its
astronomy stages. A subsequent learning experiment should resume a preserved
completed prose checkpoint with the original and extended schedules, keeping
the existing optimizer, live state and replay policy. Native extension already
supports this state contract, but that feature's earlier tests do not prove
new science learning or retained prose skill here.

Before that continuation, declare the selected checkpoint, replay settings and
assessment schedule. Compare new astronomy validation loss before and after
learning, and retain the original reading/prose checks and raw generations.
The astronomy validation chapter is a topic holdout, so its loss measures
transfer to separate material rather than coverage of every trained fact.
Keep the reserved astronomy test chapter unscored during these decisions.
Nothing in this preparation admits a model for teaching or reproduction.
