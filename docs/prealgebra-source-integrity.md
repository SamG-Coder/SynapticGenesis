# Image-dependent arithmetic solutions

This report records the initial candidate audit. The subsequent
[complete passage review and selected edition](prealgebra-worked-selection.md)
retains 221 complete examples and excludes 40; the whole book remains
unadmitted.

The [notation audit](prealgebra-notation.md) established which formulas and
modules can be serialized. The subsequent
[source-integrity review](../reports/prealgebra-source-integrity.json) found a
separate problem: many worked solutions store essential steps in images with
empty or punctuation-only descriptions. A successful text conversion does not
by itself make those solutions complete learning targets.

Across the same 75 authenticated modules, the review found:

| Measurement | Count |
| --- | ---: |
| Media elements | 2,978 |
| Empty or punctuation-only media descriptions | 1,927 |
| Such descriptions with solution as their nearest problem/solution/figure ancestor | 1,793 |
| Such descriptions with an ancestor table description | 1,620 |
| Exercises containing both a problem and a solution | 5,240 |
| Pairs inside worked examples | 737 |
| Extractable worked-example candidates without images or explicit links | 261 |
| Candidate text bytes, before review, deduplication and splits | 138,202 |

Missing image descriptions are review flags, not proof that an entire example
is unusable. Conversely, the presence of a table description does not prove
that it faithfully replaces the image. Three inspected cases illustrate why:

- Module `m81270` asks for `x + 7` when `x = 3`. Its description leaves `x` in
  the substituted expression and claims the resulting `3 + x` is 10.
- Module `m81320` asks for `x - 11 = -3`; the text steps use `x = 8`, but its
  description changes the problem to `x + 11 = -3` and gives `x = -14`.
- Another example in `m81320` asks for `m + 4 = -5` and substitutes `m = -9`,
  while its description and final sentence switch to `m - 4 = -5`, `m = -1`.

The report retains the exact exercise/table IDs, problem XML, descriptions
and text needed to inspect these observations. No source correction or image
transcription was attempted. All three cases are excluded by the candidate
extractor's image rule.

## Worked examples for review

`scripts/review_prealgebra_integrity.py` keeps a problem with its own source
solution and title. It considers only exercises inside worked examples;
ordinary drills often rely on instructions outside the exercise. It rejects
embedded images, explicit links, malformed tables and empty formulas. This
does not establish that the surviving prose is independent of all surrounding
context, so candidates remain outside training admission.

Six candidates were inspected in full: classifying whole/counting numbers,
an `8 × 14 = 112` tile problem, a 20-degree temperature difference, comparing
`3/8` with `0.4`, simplifying `6(3x)`, and distinguishing vertical/horizontal
line slopes. Their extracted questions, calculations and answers agree in
these checks. The report preserves all six complete passages and their hashes.
This is a small sample review, not a factual audit of all 261 candidates.

The local candidate bundle includes the pinned source license, original
preface and attribution. It can support a later selected question/solution
edition after broader passage review, deduplication and protected splits.
The whole book remains unadmitted. No model learned these candidates.

## Verification

The [focused fixtures](../reports/prealgebra-integrity-fixtures.json) preserve
a complete question/answer pair and reject eight incomplete or dependent
variants, including images nested in either side, missing/duplicate solutions,
an external reference, an empty formula and an invalid table. The description
check distinguishes punctuation placeholders from a short numeric label.
The real review authenticates all 75 module hashes before examining them.

```powershell
python tests/prealgebra_integrity.py
python scripts/review_prealgebra_integrity.py --out runs/prealgebra-integrity-review-new
```

Use a fresh output directory and the pinned source cache from the notation
audit. These are CPU data-review operations. They change no native learning
policy, prepared training edition, source file or checkpoint.
