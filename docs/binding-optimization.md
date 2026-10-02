# Binding optimization controls

The earlier full-size LIF and trace models reached about 50% question accuracy and zero complete four-answer groups on the selected binding curriculum. Before changing their representation, two controls tested whether existing cells could fit a complete group and whether the later-stage learning rate was simply too low.

## One complete group

The tiny diagnostic selects the first complete **training** group from the unchanged binding-v2 source specification. It contains four prompts: two swapped location assignments crossed with two object questions. It trains a fresh full-size model for 6,000 observations at answer emphasis 64, without replay, and with live graph generation every 500 observations. The learning rates are 0.000075, 0.0003 and 0.001.

Both LIF and trace reach 100% candidate accuracy and 100% unrestricted greedy exact answers on all four prompts at all three rates. This is memorization of four training examples, not generalization. It shows that the cells and supervision pipeline can fit the required two-context/two-query relation in a small case.

```powershell
python scripts/diagnose_binding.py --out runs/binding-tiny-diagnostic
```

## Rate comparison from identical live history

Within each cell, every arm resumes the exact same checkpoint saved after the 6,000-reading and 4,000-prerequisite stages. All arms preserve optimizer history, replay reservoir, source cursor, speech state and the original curriculum identity. Only the explicitly supported base learning-rate override changes. Each then completes 24,000 binding observations and 6,000 replay updates. No reserved test questions are evaluated.

The rate shown below is the actual binding-stage rate. The CLI base rate is four times larger because the unchanged curriculum applies its 0.25 multiplier.

| Cell | Binding rate | Complete training groups | Complete development groups | Final earlier-reader loss |
| --- | ---: | ---: | ---: | ---: |
| LIF | 0.000075 | 0.69% | 0% | 2.69042 |
| LIF | 0.0003 | 0% | 0% | 2.72581 |
| LIF | 0.001 | 0.46% | 0% | 2.81140 |
| Trace | 0.000075 | 0.23% | 0% | 2.70066 |
| Trace | 0.0003 | 0% | 0% | 2.83882 |
| Trace | 0.001 | 0% | 0% | 2.83184 |

Increasing the rate did not solve the full collection and worsened earlier-reader loss compared with the slowest arm for each cell. These are exploratory one-seed comparisons, not evidence that no training schedule could work. Small differences from earlier repeated runs are expected from non-bitwise-deterministic GPU training. The source checkpoint hashes and identical exposure counts are recorded in the [full report](../reports/binding-optimization.json).

```powershell
python scripts/compare_binding_rates.py --out runs/binding-rate-panel
python scripts/summarize_binding_diagnostics.py
```

## Response sensitivity

For the earlier LIF baseline, changing the queried object while preserving facts changed its candidate prediction in only 3.13% of comparisons. Swapping the facts with the query fixed changed it in 7.99%. The trace baseline changed predictions in 11.81% and 23.61%, respectively. These are sensitivity measures, not correctness: an answer can change in the wrong direction.

The report separately measures predictions matching the first or last mentioned location. Neither explains every error. The broader failure is weak use of both fact content and query identity. This motivates an input-dependent read gate over stored traces, while retaining the full task and its controls. It does not prove that architecture alone caused the failure.

```powershell
python scripts/audit_binding.py runs/binding-trace-1337/development.json
```
