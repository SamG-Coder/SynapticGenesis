# Arithmetic and Physics after the complete prose curriculum

The reviewed [arithmetic examples](prealgebra-worked-selection.md) and
[Physics modules](physics-corpus.md) now have a prepared continuation of the
four-stage prose schedule. It adds **168 complete worked lessons and 90
Physics modules**, containing 284,838 alphabetic word occurrences. No model
has learned these additions yet. This is preparation for live learning, not
a checkpoint selection or an automatically queued training run.

The first four stages retain their exact source bytes, cumulative endpoints,
learning-rate scales, traversal scopes and answer weights. They still end at
216,289 online observations. The additions introduce every selected training
document once. Arithmetic retains its four reviewed topics in source order
within one native stage, followed by four Physics stages:

| Stage | Content | Documents | New observations | Cumulative endpoint |
| ---: | --- | ---: | ---: | ---: |
| 5 | Whole numbers, algebraic language, integers, fractions, decimals, percents and selected geometry/graphing | 168 | 809 | 217,098 |
| 6 | Units, motion and forces | 23 | 3,385 | 220,483 |
| 7 | Mechanics, heat and thermodynamics | 23 | 2,705 | 223,188 |
| 8 | Waves, optics, electricity and magnetism | 29 | 4,483 | 227,671 |
| 9 | Quantum, atomic and particle physics with reference tables | 15 | 2,675 | 230,346 |

At chunk length 128, the addition contains **14,057 source observations and
1,783,551 next-byte targets**. Each document's final partial window is included;
targets do not cross document separators. These counts exclude any replay
updates, generated speech and the prior prose history. Arithmetic supplies
809 observations, Physics 13,248. One pass is a declared starting exposure,
not evidence of sufficient practice or an optimal training duration.

The final cumulative corpus contains 313 documents and 28,680,483 UTF-8 bytes,
including separators. The newly added text is a subject supplement to the
existing 4,817,047-word prose corpus. Its small arithmetic component and four
geometry/graphing examples should not be described as a comprehensive
mathematics curriculum. Topic order is not a biological age, a verified
prerequisite relationship, or a measured mastery gate.

## Source integrity and evaluation separation

The reusable `corpus/reviewed_edition.py` module authenticates each admitted
source specification and the published preparation audit. It checks every
prepared document, split and topic file, the source commit, reviewed
corrections, attribution, licence, original preface and collection metadata.
It does not extract new passages or change admission decisions.

Before creating output, the schedule builder checks the combined old and new
training text against 20 protected corpus/probe files. The existing whole-pair
reservation logic preserves mathematical signs, decimals and variable case.
It rejects normalized whole texts, repeated questions from reserved arithmetic
pairs, held-out paragraphs of at least 120 characters, and embedded authored
probe contexts. A conflict rejects the proposal; it does not delete part of a
question or solution. These exact checks do not rule out paraphrases,
shorter overlap or changed-number templates. Reading protected text for
separation checks does not score it with a model.

Both complete textbook handoffs retain source notices under
`provenance/<edition>/`, including the original CC BY 4.0 licences, attribution
and review decisions. The Physics handoff also contains the arithmetic
notices. Repository MIT terms continue to cover code rather than replace
the source licences. Prepared text and run output remain local ignored data;
the public repository contains the preparer, audit and reproduction details.

## Reproduce the preparation

With the admitted prepared editions and original prose schedule present:

```powershell
python scripts/development_schedule.py --out runs/prose-arithmetic-physics-curriculum-v2
python tests/development_schedule.py --out runs/development-schedule-check-v2 --report reports/development-curriculum.json
```

Use fresh output directories. The first complete handoff is
`runs/prose-arithmetic-physics-curriculum-v2/after-arithmetic-1/curriculum.sg`;
the final handoff is
`runs/prose-arithmetic-physics-curriculum-v2/after-physics-4/curriculum.sg`.
The protocol records the authenticated input paths and generated files,
including the five successive self-contained schedule editions.

The [CPU audit](../reports/development-curriculum.json) independently enumerates
target positions, reconstructs every cumulative source, preserves every old
stage and verifies that all 258 selected documents appear once. It also
checks 21 copied attribution/provenance files and rejects 17 altered or
reserved inputs, including a changed parent learning-rate policy. No native
model command or CUDA work was part of that audit.

## Conditions for the learning experiment

The later [Physics errata review](physics-followup-errata.md) found seven
passage corrections in three of the selected modules. This unconsumed
continuation retains the original Physics text and must be regenerated from
a newly admitted corrected edition before Physics learning starts. The
separate review copies are not themselves an admitted edition.

Native admission of this particular schedule remains untested. A learning
experiment must declare a completed prose checkpoint, executable, learning
and replay policies, and assessment schedule before resuming its full weights,
optimizer and live state through append-only extension. Preparation does not
choose between the 105M and 411M models or between optional membrane policies.

All added stages retain rate scale one, `new` source scope and ordinary byte
loss weight one. The complete question and solution remain targets; this
proposal introduces no answer emphasis. Stage-balanced replay will divide a
fixed reservoir across nine groups as they become active. The existing
prose replay distribution cannot be assumed to remain unchanged.

The grouping addresses a specific resource cost in the current replay policy.
The four arithmetic topics contain only 237, 269, 275 and 28 source windows.
Separate native stages would each receive an equal stage quota and replay
selection share despite their small sizes. Applying the current policy's
`stored = min(seen, quota)` rule at the final endpoint gives this conditional
projection for a 16,384-slot reservoir:

| Arithmetic grouping | Final groups | Retained prose windows | Unused reservoir slots |
| --- | ---: | ---: | ---: |
| One stage, retained topic order | 9 | 7,284 | 1,011 |
| Four separate stages | 12 | 5,464 | 4,651 |

These are calculated occupancy counts under the declared complete exposure,
not a simulated replay trajectory, a native result or proof of improved
retention. Grouping preserves 1,820 more old prose windows at that capacity
and avoids giving 28 geometry/graphing windows their own equal replay share.
It still leaves a small underfilled arithmetic quota. No checkpoint, capacity
change or new replay algorithm is selected by this calculation.

Assess new arithmetic and Physics validation loss before learning, after
arithmetic, and after Physics, alongside the existing prose/reader monitors
and raw generations. Measure retention at both handoffs and preserve failed
answers. The reserved chapters remain unscored during selection. Improvement
in next-byte loss alone would not demonstrate arithmetic solving, correct
scientific explanations, or conversational ability.

The earlier [astronomy continuation](astronomy-curriculum.md) remains a
separate, unconsumed proposal. It is not silently inserted into this schedule.
The running 411M prose study and queued CUDA acceptance work retain their
original inputs. Neither preparation nor a successful input audit admits a
model for teaching or reproduction.
