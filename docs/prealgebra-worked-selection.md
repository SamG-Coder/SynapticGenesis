# Reviewed arithmetic worked examples

The selected arithmetic supplement contains **168 training lessons**, 29
validation lessons and 24 reserved test lessons. All 261 candidates from the
[source-integrity review](prealgebra-source-integrity.md) were read in full.
The assistant retained 221 and excluded 40 because of contradictory steps,
missing assumptions, incomplete notation or factual claims outside this
arithmetic review. This is an assistant review, not independent expert
certification.

The [passage decisions](../data/prealgebra-passage-review-v1.json) identify every
candidate by its module, exercise and extracted-text hash, with an individual
reason. The [admitted selection](../data/sources-prealgebra-v1.json) contains
only retained complete pairs. Admission covers this selection, not the whole
textbook. No model has learned it yet, and preparation creates no native
curriculum or teacher/reproduction admission.

| Split | Complete lessons | UTF-8 bytes including document separators | Alphabetic word-like units |
| --- | ---: | ---: | ---: |
| Training | 168 | 93,269 | 11,065 |
| Validation | 29 | 13,574 | 1,498 |
| Reserved test | 24 | 10,652 | 1,075 |

The word-like count uses the declared alphabetic-word regular expression;
numbers and much of the formula notation are not words under that measure.
This is a small question-and-solution supplement to the existing multi-million
word reading collection. It is not a large new language corpus or evidence
that any model can solve arithmetic.

## What the complete review changed

Some text-complete examples still require surrounding chapter assumptions.
For example, an extracted solution simplifies `sqrt(x^2)` to `x` without
retaining a nonnegative-variable assumption. Other examples cancel a variable
without preserving its original nonzero restriction. Such examples are
excluded from this standalone edition rather than silently qualified or
rewritten. These decisions do not assert that the full chapter lacks the
necessary context.

Other exclusions have direct internal contradictions. A division example
switches from 10,519 back to a copied 645 in one formula. A graphing example
correctly derives the intercept `(3, 0)` but then calls it `(-3, 0)`. Another
solution says to multiply two factors while displaying four. A finite decimal
prefix followed by an ellipsis is also insufficient to establish the claimed
irrationality. The review records these specific failures.

Undated geographic assertions, unverified survey statistics, medical advice,
regulatory claims and physical formulas with unstated conditions are omitted
as whole lessons. The selection retains ordinary mathematical hypotheticals,
including stated prices, unit rates and explicitly titled simple-interest
calculations. No source correction, generated replacement answer or image
transcription enters the edition.

## Pairs, evaluation boundaries and attribution

The entire **Properties of Real Numbers** chapter is assigned to validation,
and **Polynomials** to reserved test. Selected examples from the other chapters
are training. The training fragments group whole numbers/algebraic language,
integers/fractions, decimals/percents, and selected geometry/graphing. These
are content stages, not biological ages or measured mastery.

The preparation protects sixteen existing validation/test corpus or authored
probe files. Old heldouts take precedence, followed by new test, validation
and training. A matching normalized whole lesson, question, or paragraph of
at least 120 characters excludes the **entire later lesson**. Authored held-out
probe contexts are protected as well. No paragraph is removed from an
otherwise retained question/solution pair. Mathematical signs, decimal points
and variable case survive normalization.

No conflicts were found by those exact checks in this edition. This does not
establish the absence of paraphrases, changed-number templates, short shared
phrases or conceptual overlap. Whole-chapter evaluation involves topic shift;
it is not an identically distributed random split. No reserved test was scored.

The source is the official
[OpenStax Prealgebra repository at commit bba5f524](https://github.com/openstax/osbooks-prealgebra-bundle/tree/bba5f5244066884797f730149d918e33f2beefd1).
This pinned 2024 edition declares CC BY 4.0. The prepared directory retains
its license, original preface, collection metadata, source specification,
passage decisions and attribution to Lynn Marecek, MaryAnne Anthony-Smith,
Andrea Honeycutt Mathis, OpenStax at Rice University and the credited
contributors. The repository MIT license covers code and does not replace
the source license.

## Preparation and verification

`scripts/prepare_prealgebra.py` authenticates the full reviewed candidate
inventory, the earlier source audits, all 75 source modules, and every
question/solution extraction. It uses the existing CNXML/math conversion
through `worked_candidate`; it does not introduce another math serializer.
The separate `corpus/paired_selection.py` owns whole-pair overlap handling.

The [verification report](../reports/prealgebra-worked-selection.json) checks
every prepared lesson against its reviewed extraction, all split/stage bytes,
attribution and original license files. Fixtures exercise a duplicate question
with a changed answer, a protected answer paragraph, older held-out text,
signed/decimal/case-sensitive mathematical expressions and quoted probe
contexts. Eight malformed pair/probe cases and six altered review/source cases
are rejected. These are CPU preparation checks; no CUDA model call is made.

With the pinned source cache used by the earlier notation audit, fresh output
paths can reproduce the candidate bundle and prepare the admitted edition:

```powershell
python scripts/review_prealgebra_integrity.py --out runs/prealgebra-candidates-rebuilt
python scripts/prepare_prealgebra.py --candidates runs/prealgebra-candidates-rebuilt/candidates.json --out data/prepared/prealgebra-worked-rebuilt
python tests/prealgebra_selection.py --prepared data/prepared/prealgebra-worked-rebuilt --candidates runs/prealgebra-candidates-rebuilt/candidates.json --out runs/prealgebra-selection-recheck.json
```

The prepared local edition is `data/prepared/prealgebra-worked-v1-pinned`.
Its three split payloads contain complete reviewed extracts separated by
`0x1e`. The native reader already treats that byte as a document boundary.
An eventual learning experiment still needs declared source exposure,
retention measurements and raw answer probes before claiming a benefit.
