# Selected lesson diversity

The current selective-trace model can fit the original binding lessons while its accuracy on new object combinations varies widely across seeds. This experiment asks whether a more varied selection of simple lessons helps it reuse the fact/query rule. It retains the same native C++/CUDA live learner and earlier-reading sources.

## Research motivation

[Zhou, Jiang and Bansal (2023)](https://aclanthology.org/2023.emnlp-main.898/) found that training-data diversity, complexity and repetition affect compositional generalization in Transformer experiments. Their results motivate varying these factors; they do not establish the effect for this spiking model. [Andreas (2020)](https://aclanthology.org/2020.acl-main.676/) studied recombining compatible fragments for compositional data augmentation. Our literal selected-noun expansion is a separate intervention, not an implementation of that paper's GECA procedure.

[Ruis and Lake (2022)](https://arxiv.org/abs/2202.10745) found that augmentation alone could fail in a gSCAN baseline, while a modular architecture made better use of it. More examples are therefore a hypothesis to test, not a presumed fix for the model's memory or generalization limitations. These are machine-learning studies, not evidence that model update counts represent human developmental ages.

## Selected material and unchanged assessment

[The selection file](../data/lessons-binding-diversity-v1.json) adds eighteen concrete three-letter object nouns to the original six. The four locations, two fact forms, two query forms and answer format stay unchanged. All labels follow the stated facts. There are no imported weights, teacher predictions or downloaded synthetic examples.

| Property | Original control | Expanded lessons |
| --- | ---: | ---: |
| Object names | 6 | 24 |
| Training object pairs | 9 | 270 |
| Single-fact prerequisite documents | 96 | 384 |
| Two-fact training documents | 1,728 | 51,840 |
| Complete four-question training groups | 432 | 12,960 |
| Original development groups | 144 | The same 144 |
| Original reserved-test groups | 144 | The same 144 |

Every question asks for one object's location after two facts. Each four-question group reverses both the queried object and the fact-to-location assignment. Always copying the first or last location scores half the individual questions and zero complete groups. All orders, assignments and templates for the original six development/test object pairs remain excluded from training. Exact held-out contexts are also checked against the selected reading and every generated training document.

Original `train.sgprobe`, `development.sgprobe` and `test.sgprobe` are copied byte for byte. Here `train.sgprobe` continues to mean the **original six-object training set**. A separate `expanded-train-monitor.sgprobe` samples 432 complete groups from expanded training with the declared seed 917203. It is fixed before learning, fits the native parser's item limit, and is **not full expanded-training accuracy**. For the control it includes unfamiliar object combinations; for the expanded arm it is a training sample. The primary development questions are identical in both arms.

The change increases vocabulary diversity, corpus size and the number of combinations together, reducing repetition at the same update budget. It does not isolate vocabulary size alone. The small, fixed grammar is not a broad dialogue corpus or an age-calibrated teaching curriculum.

## Declared paired experiment

Three seeds, 1337, 2026 and 31415, each train a founder from random weights for 6,000 reading observations. Both arms resume that seed's **same complete learned checkpoint**. Native append-only curriculum extension adds either the original or expanded lessons while preserving earlier weights, optimizer history, recurrent state and replay. Both branches receive 4,000 prerequisite observations, then 120,000 two-fact observations, ending at 130,000 total.

Both arms use selective-trace cells, C256/H512/L4, chunk 128, TF32 learning, fixed-order gradient reductions, learning rate 0.0003 with a 0.25 lesson-stage multiplier, answer emphasis 64, and 1,024 stage-balanced replay slots. Replay updates occur every four observations. Every 500 observations, the same shared model generates 96 bytes; generated text is not a training target. Both arms retain 342 reading, 341 prerequisite and 341 binding replay windows once the three stages have filled their slots.

The primary endpoint is original development **joint group accuracy at 130,000** observations. Original training accuracy, the expanded-training monitor, unconstrained exact answer generation, earlier-reader loss, target-byte exposure and live timing accompany it. Predeclared observations at 34,000 and 67,000 show the trajectory; they are not candidates for selecting a favorable checkpoint. No trained model is evaluated on the reserved test partition.

Equal observation and replay-update counts do not guarantee equal target-byte counts. Source order, window length and replay selection can change that exposure. Reports preserve those counts. Live timing sums the continuation segments after the common ancestor and includes replay, speech, logging and checkpoint I/O; setup and held-out evaluation are outside it. Run order alternates across seeds and measurement points. There is one timing realization per arm/seed, not a dedicated throughput benchmark.

## Preparation and validation

After preparing the existing `binding-v2` selected edition:

```powershell
python scripts/prepare_binding_diversity.py
python tests/binding_diversity.py --out runs/diversity-validation
python tests/binding_cli.py --out runs/original-binding-validation
python scripts/diversity_experiment.py --smoke --out runs/diversity-smoke
python scripts/diversity_experiment.py --out runs/diversity-comparison
```

The preparation refuses changed base hashes and existing output directories. The experiment authenticates prepared inputs, records source/executable/checkpoint identities, and writes its protocol before native training. The smoke mode checks extension, resume, grouped scoring and matched counters with only 128 observations; it cannot establish learning quality.

Independent tests parse all 52,992 generated question/answer rows, check all 13,248 four-question groups and prerequisite labels, verify pair exclusion and monitor membership, and reproduce both old and new prepared bytes. Negative controls reject malformed selections and held-out contexts hidden in reading or templates. A disposable native test checkpoint reads the complete monitor; eight rows also receive independent CPU scoring and greedy-generation checks. The original five-cell binding tests cover the shared renderer extraction.

Model computation remains native. `binding_lessons.py` owns shared literal rendering, preparation scripts own their selected editions, and `experiment_checkpoint.py` provides read-only checkpoint evidence for experiment drivers. The native loader remains responsible for full checkpoint validation.

The [validation report](../reports/lesson-diversity-validation.json) records successful preparation/negative controls, the original ten-command five-cell binding check, and a 25-command native smoke run through extension, scoring and two resumes per branch. Eight expanded-monitor rows match independent CPU scores within 1.44e-6, and their greedy answers match exactly. Eighteen historical checkpoint views still match their published hashes, dimensions and replay distributions after helper extraction. The native executable is unchanged from the previously validated fourteen-suite ordered-reduction stage; this data stage did not rebuild it.

Learning-quality results will be recorded after the declared paired runs complete. Passing preparation or smoke checks is not evidence of improved generalization.
