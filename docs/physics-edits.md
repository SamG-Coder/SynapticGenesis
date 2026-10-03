# Corrected Physics review edition

The [original extraction review](physics-source-review.md) identified three
rejected modules, two inconsistent isotope paragraphs and two missing diagram
descriptions. The corrected review now extracts all **99 content modules**:
**1,855,083 UTF-8 bytes and 299,768 word-like occurrences**, retaining 2,942
formulas and 565 media descriptions. The preface remains attribution only.

This is preparation for broader science learning. It is not an admitted source
edition, and the running 411M prose learner does not consume this material.
The [result](../reports/physics-edited-review.json) and
[verification](../reports/physics-edited-verification.json) preserve the actual
outputs' identities and the limits of the review.

The subsequent [selected Physics edition](physics-corpus.md) adds representative
passage review, five further text corrections, complete-chapter evaluation
splits and admission. The counts and checks below describe this earlier
extraction stage, whose reports remain unchanged.

## Explicit corrections

[Seven element-level decisions](../data/physics-edits-v1.json) are tied to the
original module bytes and serialized source elements. All original cached
files remain unchanged. The adapter changes copies before the existing
Physics converter runs:

| Source | Decision |
| --- | --- |
| Kepler's laws, `m54192` | Declare two columns for the five existing label/equation pairs. Preserve every entry and its row order. |
| Types of waves, `m54314` | Declare four columns for the two existing vocabulary rows, retaining their final empty cell. |
| Mechanical energy, `m54273` | Preserve `(I)` through `(IV)` as literal paragraph labels. The following choices and worked solution continue to refer to these labels. |
| Nuclear forces, `m54580`, two paragraphs | Correct two generic nuclide formulas to put atomic number Z below and mass number A above; correct the deuterium neutron subscript to one. |
| Reflection, `m54357` | Replace the three-dot media placeholder with a description of the directly inspected mirror ray paths. |
| Refraction, `m54365` | Replace the three-dot media placeholder with a description of the directly inspected pool diagram. Preserve the angle and refractive indices supplied in the question. |

The isotope changes follow `A = Z + N` and the notation in OpenStax's
[Properties of Nuclei](https://openstax.org/books/university-physics-volume-3/pages/10-1-properties-of-nuclei).
That source also identifies deuterium as having one proton and one neutron.
The existing, correctly written generic formula in a later decay question and
the tritium formula are preserved. These are local source corrections, not
global symbol substitutions or a review of every nuclear equation.

Both image files were fetched from the same pinned OpenStax commit and directly
inspected. Their SHA-256, Git blob identities, dimensions in the description,
and review notes are recorded in the decisions file. The pool image has no
angle label; no angle was measured from its pixels. The question provides an
underwater angle of 25 degrees and indices 1.33 and 1.00. An independent Snell-law
calculation gives approximately 34.2 degrees above water, matching the existing
34-degree option. No answer or alternative is invented or reordered.

## Empty questions and preserved meaning

The earlier source-class policy removes external exercise placeholders that
contain no local questions or solutions. Some surrounding `problem` and
`exercise` elements then render as empty `Question:` blocks. The corrected
edition removes **783 empty problem wrappers and 775 empty exercise wrappers**.

The rule runs after the established external-link omission. It removes only
whitespace-only exercise/problem/paragraph structures. Substantive text,
solutions, formulas, media and cross-references keep their containers. Text
after a removed element is preserved. This is separate from the seven explicit
source edits, and every removed wrapper ID is recorded per module.

All **93 modules outside the explicit corrections** match the original
extracted bytes after independently removing only empty `Question:` blocks.
The other six modules cover the three newly extractable modules, the isotope
module and the two described diagrams. Existing Astronomy and arithmetic
converters are unchanged.

## Validation and remaining work

The new checks independently verify all seven table rows, Roman item order and
the original answer, all three repaired formulas, both descriptions, and the
99 regenerated files. Twenty altered-source/edit cases are rejected, including
changed source bytes, ambiguous targets, wrong formula identity, a description
bound to the wrong image and escaped cache paths. Empty-wrapper fixtures retain
real questions, answers, formulas, diagrams, references and trailing prose.

The [compatibility record](../reports/physics-edits-compatibility.json) verifies
the existing study inputs and unchanged production CUDA sources. No native
model or GPU test is run for this source-processing stage.

Only the two formerly undescribed images have been inspected here. The presence
of the other 563 descriptions does not certify their diagram completeness.
Source selection, representative passage review, chapter-level evaluation
splits and overlap protection remain necessary before admitting a training
edition. No additional words have been added to the admitted training total.

```powershell
# Fresh output paths; download fetches only missing files from the pinned commit.
python scripts/review_physics_edition.py --download --out runs/physics-edited-new
python tests/physics_edits.py --review runs/physics-edited-new --report runs/physics-edited-new-verification.json
```

The pinned source's CC BY 4.0 license, original preface and collection, and an
attribution describing the modifications accompany the extracted text. The
repository's MIT license applies to code. No OpenStax endorsement is implied.
