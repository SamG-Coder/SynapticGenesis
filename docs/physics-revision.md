# Corrected Physics edition and continuation

The `selected-physics-v2` edition incorporates the seven corrections from the
[follow-up review](physics-followup-errata.md). The complete preparation has
99 modules, with 90 available for training, four reserved for validation and
five reserved for tests. Only three training modules change. All 96 other
modules, both held-out splits, and topic stages 1, 3 and 4 retain their exact
previous bytes.

This stage is prepared and checked on `research/physics-revision`. The main
checkout's selection registry is pinned by the running membrane-objective
study and has not changed. Integrating this branch must wait until that
study no longer needs those pinned files. No model has trained on this
corrected continuation, and no checkpoint is selected or promoted by it.

## What changed

The retained source is the original OpenStax Physics repository snapshot
`dfb731c737e5056750e792643fe6377425b0a067`, with its original CC BY 4.0 notice,
preface and collection. This revision does not import text from a current
edition or change the retained source's attribution.

| Module | Reviewed changes | Bytes before | Bytes after |
| --- | --- | ---: | ---: |
| `m54273` | Consistent energy-conservation subtraction and numerical substitution | 10,976 | 10,959 |
| `m54287` | Distinguish total energy from a per-particle average; qualify absolute-zero statements for quantum ground-state motion | 13,296 | 13,069 |
| `m54290` | State that heat capacity depends on the amount of material | 22,348 | 22,405 |

The revision retains all five earlier text corrections and adds the seven
follow-up corrections, for 12 declared changes. The exact before/after text,
original XML identities, review and independent arithmetic checks accompany
the prepared edition as six additional provenance files. Question and
solution marker counts remain unchanged in the three corrected modules.
The earlier review covered representative passages; this targeted repair
does not certify every factual statement in the book.

| Split | Modules | UTF-8 bytes | Word-like occurrences |
| --- | ---: | ---: | ---: |
| Training | 90 | 1,690,609 | 273,731 |
| Validation | 4 | 63,407 | 9,951 |
| Reserved test | 5 | 101,233 | 16,075 |

The correction removes 187 bytes relative to the previous edition. It is
revised teaching material, not additional independent text. No validation
or test text becomes training material.

## Prepared learning schedule

The new continuation preserves all four completed prose stages. It then
introduces 168 reviewed arithmetic lessons in one native stage, retaining
their four-topic order, followed by four Physics stages. Each new document
has one source pass, ordinary answer weighting and rate scale 1.

| Added stage | Documents | Source observations | Next-byte targets | Ending observation |
| --- | ---: | ---: | ---: | ---: |
| Arithmetic worked examples | 168 | 809 | 92,934 | 217,098 |
| Units, motion and forces | 23 | 3,385 | 432,132 | 220,483 |
| Mechanics, heat and thermodynamics | 23 | 2,705 | 344,631 | 223,188 |
| Waves, optics, electricity and magnetism | 29 | 4,483 | 572,210 | 227,671 |
| Quantum, atomic and particle physics with reference tables | 15 | 2,675 | 341,457 | 230,346 |

There are 258 new documents, 284,796 word-like occurrences, 14,057 source
observations and 1,783,364 next-byte targets. These additions are relative to
the prose-only founder; the arithmetic and original Physics were already
prepared but remain unconsumed proposals. Corrections do not change the
number of 128-byte observation windows. Replay and generated text are not
included in these source-exposure totals.

The final cumulative stream has 313 documents and 28,680,296 bytes. Its
native schedule SHA-256 is
`b309b27d3051d0c0ae1c695d6ef9f5ccb08543d88ac5986f1fcefb2ca3f0bded`.
The existing stage-balanced replay budget would be redistributed over nine
groups. These counts do not establish retention, useful question answering,
an appropriate learning rate, mastery, or biological age progression.

## Verification and implementation

- [Edition audit](../reports/physics-revision-checks.json): independently
  reconstructs every module from the prepared base plus the exact reviewed
  passages, checks all split/stage bytes, authenticates 17 provenance files
  and rejects 28 altered selections or prepared inputs.
- [Continuation audit](../reports/physics-revision-curriculum-checks.json):
  independently enumerates all target positions, checks every source
  document occurs once, verifies all earlier native stages, explicitly
  compares 27 attribution/review copies and rejects 23 altered or reserved
  inputs. All generated-file hashes, including the remaining copied
  provenance, are also authenticated.
- [Compatibility comparison](../reports/physics-revision-compatibility.json):
  rebuilding the old edition produces 114 identical prepared files;
  rebuilding the old continuation produces 40 identical native `.dat` and
  `.sg` files. The [default-path audit](../reports/physics-revision-legacy-curriculum-checks.json)
  passes the original 17 input-rejection checks.

`scripts/corpus/physics_revision.py` binds the original edition and the
reviewed errata. The existing preparation and authentication modules reuse
this binding. `development_schedule.py` accepts explicit Physics source and
audit paths, retaining its original defaults for historical reproduction.
Implementation hashes resolve to the executing checkout, including when
raw/prepared caches are shared with another checkout.

All checks in this stage are host-side data and schedule checks. There are
zero native model commands, no new learning results and no change to the
shared C++/CUDA training/inference implementation.

## Reproduction

Use the branch's scripts and selection registry with the existing pinned
source cache, reviewed Physics v1 preparation, errata evidence, arithmetic
preparation and complete prose curriculum. Use fresh output/report paths.
From a checkout of this branch, the corresponding commands are:

```powershell
python -X utf8 scripts/prepare_physics.py --sources data/sources-physics-v2.json --out data/prepared/physics-v2-reviewed
python -X utf8 tests/physics_revision.py --report reports/physics-revision-local-checks.json
python -X utf8 scripts/development_schedule.py --physics data/prepared/physics-v2-reviewed --physics-sources data/sources-physics-v2.json --physics-audit reports/physics-revision-local-checks.json --out runs/prose-arithmetic-physics-corrected-local
python -X utf8 tests/development_schedule.py --edition runs/prose-arithmetic-physics-corrected-local --physics data/prepared/physics-v2-reviewed --physics-sources data/sources-physics-v2.json --physics-audit reports/physics-revision-local-checks.json --out runs/physics-revision-local-curriculum-checks --report reports/physics-revision-local-curriculum-checks.json
```

The saved evidence uses scripts from
`D:/SynapticGenesis/runs/physics-revision-worktree` with the main directory as
the cache working directory. Its prepared output is
`data/prepared/physics-v2-reviewed`; its continuation is
`runs/prose-arithmetic-physics-corrected-v1`. Prepared data and run directories
remain local ignored artifacts. The exact versioned source selection and
verification reports are committed on this branch.
