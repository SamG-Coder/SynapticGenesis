# Physics source extraction review

The pinned OpenStax **Physics** book now has a reproducible review extraction:
96 content modules, **1,817,361 UTF-8 bytes and 292,745 word-like occurrences**.
These are candidate passages, not an admitted training edition. No curriculum,
split, checkpoint or model was created from this material.

The source is [OpenStax Physics at commit
dfb731c](https://github.com/openstax/osbooks-physics/tree/dfb731c737e5056750e792643fe6377425b0a067),
the high-school book, distinct from College Physics 2e. Its pinned LICENSE and
collection metadata declare [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The original preface, collection, license and attribution are preserved with
the review outputs. The repository's MIT license covers code, not this book.

## Evidence and scope

The [source specification](../data/physics-source-review-v1.json) pins all 100
modules and three auxiliary files with SHA-256 and Git blob hashes. Collection
order, chapter membership and source titles are checked before output is
created. The preface is attribution only; the 99 content modules include the
reference-tables appendix and material from 23 chapters.

The [source report](../reports/physics-source-review.json) records the complete
extraction inventory. All **3,065 raw formula elements** convert to explicit
text notation, including formulas inside content subsequently omitted. The
96 candidate modules retain **2,897 formula elements**. Conversion is neither
a factual check nor proof that a passage is self-contained.

The [verification report](../reports/physics-review-verification.json) records:

- Exact re-extraction of all 96 candidate files and the three remaining
  structural rejections.
- Ten formula fixtures, 33 rejected notation/structure cases and 13 rejected
  changes to source identities, review roles or output state.
- Preserved list headings, references, item order and text following omitted
  elements, plus explicit punctuation and external-exercise checks.
- Eight [sampled source observations](../data/physics-review-cases-v1.json).
  These authenticate the inspected passages; they do not certify the book.

The [compatibility record](../reports/physics-existing-corpus-compatibility.json)
also verifies unchanged Astronomy output for 184 modules, unchanged arithmetic
output for 221 selected lessons, and the immutable inputs of the running
native studies. No native runtime was built or executed for this stage.

## Explicit extraction policy

`scripts/corpus/physics.py` is an opt-in extension of the shared CNXML reader.
The Astronomy converter and the selected arithmetic edition keep their
existing behavior. Mathematical grouping, bold-symbol markers and the source's
single left subscript/superscript pair are retained. Unreviewed MathML styles
and more general tensor layouts fail. This notation follows the
[MathML prescript structure](https://www.w3.org/TR/MathML3/chapter3.html#presm.mmultiscripts);
it is not a typesetting facsimile.

Teacher notes, learning objectives, video/simulation/activity instructions and
vocabulary grids have explicit source-class omission rules. The successful
candidates omit 715 note elements, 72 vocabulary grids and 746 empty external
exercise placeholders. Those placeholders contain neither questions nor
answers in the pinned files. One uses a Unicode hyphen in its class name; that
exact alias is counted explicitly. No external exercise service is called.
Eight titled lists keep their headings, numbering and order.

The output repairs five reviewed CP1252 quote/dash characters, counting 169
replacements while preserving the cached originals. The mapping is documented
by [Unicode's Microsoft CP1252 table](https://www.unicode.org/Public/MAPPINGS/VENDORS/MICSFT/WINDOWS/CP1252.TXT).
Other C1 control characters remain rejected. No source equations or answers
are corrected by this policy.

## Remaining source problems

Three complete modules remain outside the candidate text:

| Module | Reason |
| --- | --- |
| `m54192`, Kepler's Laws of Planetary Motion | A key-equation table declares one column but contains two entries per row. |
| `m54314`, Types of Waves | An unmarked vocabulary grid declares three columns but contains four entries per row. |
| `m54273`, Mechanical Energy and Conservation of Energy | The list uses parenthesized upper-Roman labels, which the current reader cannot preserve. |

The candidates retain 552 media descriptions, of which two are only `...`.
They occur in a reflection problem and a refraction problem that refer to
figures. Images have not been inspected, and a nonempty description is not
proof that its diagram information is sufficient. There are no remaining
generic cross-reference substitutions in the current candidate files.

Inspection also found two internal inconsistencies in
[`m54580`, Nuclear Forces and Radioactivity](https://github.com/openstax/osbooks-physics/blob/dfb731c737e5056750e792643fe6377425b0a067/modules/m54580/index.cnxml):
the generic nuclide expression reverses the placement used by the isotope
examples, and the deuterium expression's neutron subscript disagrees with its
own prose. These source bytes remain intact in the review bundle. They must
be resolved or excluded before any passage is admitted.

The next data stage needs passage and diagram-dependency review, explicit
decisions on these source issues, protected evaluation splits and overlap
checks against existing material. The full 292,745 count cannot be added to
the admitted training total. It counts repeated words and notation labels,
not unique knowledge or the native model's byte tokens.

## Reproduce

Run from the repository root, using a fresh output directory. Existing cache
files must match the source pins; `--download` fetches only missing files from
the recorded commit.

```powershell
python -X utf8 scripts/review_physics.py --download --out runs/physics-review-local
python -X utf8 tests/physics_review.py --review runs/physics-review-local --out runs/physics-review-local-check.json
```

Original source text stays under ignored cache/review directories. The public
repository contains the source pins, policy, tests and evidence, with no
automatic training admission.
