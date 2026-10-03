# Selected Physics curriculum material

The admitted `selected-physics-v1` edition adds **90 complete training modules**,
containing **1,690,796 UTF-8 bytes and 273,773 alphabetic word-like occurrences**.
Four modules from **Momentum** supply validation text; five from **Electrical
Circuits** remain reserved for tests. The local prepared edition is
`data/prepared/physics-v1-reviewed`.

This adds introductory science to the existing reading collection. Preparation
does not start learning or change a running curriculum. The 411M prose study
continues with its original source schedule and frozen inputs. No model has
learned this Physics edition, and no question-answering or retention benefit is
established.

A later [targeted errata review](physics-followup-errata.md) identified seven
corrections in three training modules. Separate corrected copies and their
checks are available, but are not yet an admitted edition. The existing
unconsumed Physics continuation must be rebuilt from a newly admitted
corrected edition before learning begins; this historical selection is
preserved for reproducibility.

| Split | Complete modules | UTF-8 bytes including separators | Alphabetic word-like occurrences |
| --- | ---: | ---: | ---: |
| Training | 90 | 1,690,796 | 273,773 |
| Validation: Momentum | 4 | 63,407 | 9,951 |
| Reserved test: Electrical Circuits | 5 | 101,233 | 16,075 |

These word counts use the declared alphabetic-word expression, so they exclude
numbers and much of the formula notation. Native tokens remain UTF-8 bytes.
This supplement is substantially smaller than the 55-book prose training
collection; it is not a claim of millions of additional documents or independent
examples.

## Reviewed content and explicit adaptations

The [source specification](../data/sources-physics-v1.json) authenticates the
original collection, source modules, [corrected extraction](physics-edits.md),
and [passage review](../data/physics-passage-review-v1.json). The source remains
the official [OpenStax Physics snapshot](https://github.com/openstax/osbooks-physics/tree/dfb731c737e5056750e792643fe6377425b0a067).

Thirty representative paragraphs were read across all 23 teaching chapters,
selected notation/diagram modules and the appendix. This is an assistant review
of representative passages, not a full expert fact check of the book. The seven
earlier source-element repairs and empty-wrapper removal remain independently
recorded in the corrected extraction. Five further text adaptations address
observed misleading statements:

| Passage | Adaptation and evidence |
| --- | --- |
| Force | Say that net force changes velocity, avoiding the implication that maintaining motion requires net force. [OpenStax, Newton's first law](https://openstax.org/books/physics/pages/4-2-newtons-first-law-of-motion-inertia). |
| Linear momentum | A negative component refers to the chosen axis; momentum still points along velocity. [OpenStax, linear momentum](https://openstax.org/books/university-physics-volume-1/pages/9-1-linear-momentum). |
| Thermal equilibrium | Replace the misleading claim about empty space between Earth and the Sun with the statement that radiation transfers heat across empty space. [OpenStax, radiation](https://openstax.org/books/college-physics/pages/14-7-radiation). |
| Blackbody radiation | Distinguish an ideal blackbody from a real shirt; visible color alone does not specify infrared absorption and emission. [OpenStax, radiation](https://openstax.org/books/college-physics-2e/pages/14-7-radiation) and [blackbody radiation](https://openstax.org/books/university-physics-volume-3/pages/6-1-blackbody-radiation). |
| Periodic-table legend | Separate the background colors for element classes from the symbol colors for physical states. The directly inspected [source diagram](https://raw.githubusercontent.com/openstax/osbooks-physics/dfb731c737e5056750e792643fe6377425b0a067/media/CNX_APPhysics_AA_PeriodicPU_img.jpg) shows mercury and bromine as liquids despite their different element classes. |

Each correction matches one recorded source string in one identified module.
The other **94 modules** remain byte-identical to the corrected extraction.
The original source bytes are preserved. Three source diagrams have now been
directly inspected across the two review stages; the remaining source
descriptions are not certified as complete. Historical examples, introductory
approximations, and questions without worked answers remain in their book
context. Multiple-choice alternatives are not automatically treated as facts
or generated answers.

## Complete modules and evaluation separation

`scripts/prepare_physics.py` authenticates the review inputs, regenerates all
99 content modules, applies the five declared text corrections, and calls the
shared document selector. A conflict excludes the entire module instead of
removing a paragraph from a question, worked solution or explanation.

Eighteen existing validation/test corpus or authored-probe files are protected.
They take precedence over new test, validation and then training modules.
Exact normalized full text, paragraphs of at least 120 characters and authored
probe contexts are reserved. Signs, decimal points and variable case are
preserved. A rejected held-out module still reserves its other paragraphs,
preventing those passages from entering training merely because another part
of the module overlapped an older source.

All 99 modules survive these exact checks. This does not prove the absence of
shorter matches, paraphrases, shared templates or conceptual overlap. Reserving
whole chapters produces topic shift rather than an identically distributed
random split. No reserved test has been scored.

Training modules are grouped into four topic stages:

| Stage | Topics | Training bytes including separators |
| --- | --- | ---: |
| 1 | Units, motion and forces, chapters 1–5 | 432,177 |
| 2 | Mechanics, heat and thermodynamics, chapters 6–12 excluding Momentum | 344,863 |
| 3 | Waves, optics, electricity and magnetism, chapters 13–20 excluding Electrical Circuits | 572,267 |
| 4 | Quantum, atomic and particle physics, chapters 21–23, plus reference tables | 341,486 |

The four files together have three fewer separators than the combined training
file. These are engineering content groups, not biological ages or measured
mastery. An eventual learning study needs a separately declared continuation
schedule, exposure budget, retention controls and raw answer probes.

## Verification and reproduction

The [selection checks](../reports/physics-selection.json) compare every prepared
module, split and stage with the authenticated source and recorded adaptations.
They reject 27 altered review/source/policy cases. Fixtures also check whole
document overlap, protected contexts, unchanged mathematical signs, held-out
priority, and the other paragraphs of a rejected held-out module. Six malformed
document/protection cases are rejected.

The [arithmetic compatibility check](../reports/physics-pair-compatibility.json)
reconstructs the existing 221 complete worked lessons, retaining the same 40
review exclusions and every prepared output byte. The existing pair-selection
API keeps its prior behavior. [Admission checks](../reports/physics-admission.json)
cover all seven active editions and twelve rejected-input cases.
The [integration record](../reports/physics-selection-integration.json) checks
preserved study inputs, unchanged CUDA sources and runtime binaries, and the
single new registry entry. All work in this stage is CPU source preparation;
no new model command is issued.

With the existing eighteen protected files and fresh output paths:

```powershell
python scripts/prepare_physics.py --download --out data/prepared/physics-rebuilt
python tests/physics_selection.py --prepared data/prepared/physics-rebuilt --out runs/physics-recheck.json
python tests/corpus_selection.py --out runs/physics-admission-recheck.json
```

`--download` fetches only missing source files and the three inspected diagrams
from the pinned commit. Existing cache files must authenticate. The prepared
directory retains the original license, preface and collection, source
specification, source edits, passage review, and adaptation attribution.
The book's CC BY 4.0 terms remain separate from the code's MIT license, and no
OpenStax endorsement is implied.
