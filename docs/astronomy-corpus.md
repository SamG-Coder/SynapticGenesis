# Selected astronomy material

The general learning collection now includes **466,756 additional training
word occurrences** from a pinned edition of OpenStax *Astronomy 2e*. Together
with the selected prose edition's 4,817,047 training words, this makes
**5,283,803 authored training word occurrences**. These are counts under the
same word-like regular expression, not unique facts, vocabulary entries or
independent examples. The native tokenizer continues to use UTF-8 bytes.

The astronomy selection is prepared for future learning. **No model has learned
it yet**, and preparing it does not alter the active prose founder's schedule.
The current 105M run remains an experiment on the original prose edition.

| Split | Modules | Text bytes | Word-like occurrences |
| --- | ---: | ---: | ---: |
| Training | 174 | 2,833,157 | 466,756 |
| Validation | 5 | 77,791 | 12,927 |
| Reserved test | 5 | 82,938 | 13,596 |

All of chapter 17, *Analyzing Starlight*, is validation; all of chapter 19,
*Celestial Distances*, is reserved test. A chapter's introduction, sections and
review material share its split. These splits introduce topic differences and
do not guarantee independence: definitions, related topics and paraphrases can
recur across chapters. Existing prose holdouts and then new test/validation
paragraphs take priority during exact deduplication of normalized paragraphs
at least 120 characters long. No protected paragraph survives in new training.

The [selection](../data/sources-astronomy-v1.json) pins the source commit,
original module hashes, reviewed extraction hashes, collection metadata,
license and preface. The [source review](../reports/astronomy-source-review.json)
retains one sampled passage per chapter, the selection limits and attribution.
The review also inspected representative equations and a spacecraft table
whose planet labels span several rows. It is not an exhaustive scientific or
editorial audit. Mission descriptions, object counts and other changing facts
reflect the archived edition.

The extractor keeps explanations, worked examples, questions, available
solutions and glossary entries. It serializes fractions, powers, subscripts,
roots and mathematical matrices explicitly as LaTeX. It expands merged table
labels across their covered rows and columns; this keeps, for example, Juno's
row associated with Jupiter. Local references are renumbered inside each
module. Images are not downloaded: source image descriptions, captions and
credits supply the available text. Unresolved references use generic wording,
which is a known loss of specificity.

Front/back matter and sidebars for external links, biographies, connections
and observing activities are omitted from learning. Module `m59901`, a chapter
review, is deferred because a two-column table refers to an undeclared third
column. The converter rejects that structure rather than silently repairing
the original. It also rejects unsupported MathML. Physics and prealgebra were
cached separately for research but are not admitted sources; their more complex
mathematical layouts still need review and conversion work.

Two explicit corrections are applied after verifying the original extraction.
The archived mission table gives Cassini's Jupiter flyby as December 2002;
[NASA's trajectory record](https://science.nasa.gov/resource/cassini-trajectory/)
places it on December 30, 2000. The table footnote also attributes a gravity
assist's energy to the planet's rotation. This is corrected to orbital motion,
consistent with [NASA's gravity-assist primer](https://science.nasa.gov/learn/basics-of-space-flight/primer/).
The selection records the exact before/after strings, reasons, references and
corrected text hash. Each edit must match exactly once. These corrections do
not imply that every factual statement in the book has been verified.

The [corpus audit](../reports/astronomy-corpus.json) passed mathematical fixtures,
merged-table alignment, malformed-format rejection, all 184 module
re-extractions, 30 sample identity checks, whole-chapter isolation, existing
holdout protection and changed-source rejection. The source-admission suite
also passed with five active editions and twelve negative cases. These are
CPU/data checks, not new model-quality results. The two explicit corrections
and rejection of an ambiguous repeated correction also passed.

```powershell
python scripts/prepare_openstax.py --out data/prepared/astronomy-v1-reviewed
python tests/openstax_corpus.py --out reports/astronomy-corpus.json
```

The current prepared directory is `data/prepared/astronomy-v1-reviewed`. The
earlier uncorrected draft is not the admitted selection. The current directory
contains separate training, validation and test files,
four topic-stage files, a manifest, attribution, the original license and
preface. It contains no automatic native curriculum. A separate
[append-only curriculum preparation](astronomy-curriculum.md) now declares one
complete pass of each topic stage after the prose curriculum. No model has
learned it yet; checkpoint selection, prior-memory checks and replay settings
must be bound by the subsequent learning experiment.

## Attribution and edition

Selections are adapted from *Astronomy 2e* by Andrew Fraknoi, David Morrison,
Sidney C. Wolff, OpenStax at Rice University, and the contributors credited in
the original preface. The source is
[OpenStax commit 9d7e69a](https://github.com/openstax/osbooks-astronomy/tree/9d7e69a2e0c9a651ad42254b9a93b701a6b08c10),
dated October 17, 2024. Its
[license file](https://github.com/openstax/osbooks-astronomy/blob/9d7e69a2e0c9a651ad42254b9a93b701a6b08c10/LICENSE)
and collection metadata declare
[Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).
The [preserved license](../data/licenses/openstax-astronomy-2024.txt) applies
to this source selection. The repository's MIT code license does not replace
it. Conversion, omissions, reference renumbering, table expansion, the two
explicit source errata and deduplication are modifications made by this project. No endorsement by
OpenStax or the authors is implied.
