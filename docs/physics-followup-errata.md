# Follow-up Physics corrections before learning

Preparing numerical learning checks exposed additional problems in three selected Physics modules. Seven explicit changes now have separate review copies and an independently checked energy derivation. These copies are **not admitted training material**. The original selected edition, its prepared curriculum and the running prose experiments retain their existing bytes.

The unconsumed arithmetic/Physics continuation must be regenerated from a newly admitted corrected edition before a Physics learning experiment starts. Its existing hashes describe the original selection; they must not be reused for the corrected text. No checkpoint or teacher has learned these proposed changes.

## Findings and corrections

| Finding | Proposed change | Evidence |
| --- | --- | --- |
| A falling-rock solution reverses the conserved-energy subtraction, then uses a different inconsistent numerical expression. | Derive `KE2 = KE1 + PE1 - PE2`, giving `1960 - 980 = 980 J` for its stated conditions. | The original source XML and the publisher's [worked example](https://openstax.org/books/physics/pages/9-2-mechanical-energy-and-conservation-of-energy) contain the bad equation. The preceding conservation equation and an independent symbolic solve establish the correction. |
| Thermal energy is defined as the average energy of one particle. | Distinguish the system's energy from a per-particle average; qualify the simple temperature relation using a classical monatomic ideal gas. Update its glossary too. | The publisher's [kinetic-theory derivation](https://openstax.org/books/university-physics-volume-2/pages/2-2-pressure-temperature-and-rms-speed) distinguishes average molecular energy from the total. |
| Absolute zero is described as cessation of all molecular motion. | Qualify the body, summary and glossary to allow quantum ground-state motion. | The [quantum oscillator treatment](https://openstax.org/books/university-physics-volume-3/pages/7-5-the-quantum-harmonic-oscillator) gives a nonzero ground-state energy and explains why a bound particle cannot simply be motionless. |
| Heat capacity is said to have no mass dependence, beside `C = m c`. | State the mass dependence of heat capacity at fixed material and conditions, distinguishing it from specific heat. | The source's own formula and the publisher's [heat-capacity treatment](https://openstax.org/books/college-physics-2e/pages/14-2-temperature-change-and-heat-capacity). |

For the energy example, the source's symbolic expression gives **-980 J**, its separately printed numerical expression gives **2940 J**, and its stated answer is **980 J**. The corrected equation consistently gives 980 J. This is a substantive source error, not a whitespace or OCR issue.

The separate conversion of 25 degrees Celsius to 298 K is retained: it can represent rounding to the nearest kelvin. The review does not count this as another error or rewrite it merely because the unrounded value has decimals.

## Preserved provenance and verification

The [correction specification](../data/physics-followup-errata-v1.json) records each exact before/after passage, its reason and references, the original XML element identity, and hashes of source, extracted and corrected content. It operates on the previously pinned OpenStax snapshot `dfb731c737e5056750e792643fe6377425b0a067`, retaining that snapshot's original licence, preface, collection metadata and attribution. Current reference pages support the research findings; their text is not imported as new training passages.

`scripts/review_physics_errata.py` reuses the reviewed-edition authentication, exact correction application and protected-text reservations already used by the corpus pipeline. It emits three corrected text copies, three readable diffs and provenance in a fresh review directory. It creates no training split, source-registry entry or native curriculum.

The [CPU audit](../reports/physics-followup-errata-checks.json) checks all seven changes, reverses them to recover each original module exactly, preserves question/answer markers, verifies the three original attribution files and rejects 13 changed-input cases. SymPy independently solves the conservation equation; exact rational arithmetic checks the worked quantities and supplies a mass-dependent heat-capacity counterexample. These checks do not constitute a general scientific fact checker.

The [review result](../reports/physics-followup-errata-review.json) pins the inputs and output hashes. The three amended modules contain 187 fewer UTF-8 bytes in total. Their number of 128-byte source observations happens to remain unchanged under a single pass; their targets and hashes change, so a future curriculum must still be rebuilt and verified.

```powershell
python tests/physics_errata.py --out runs/physics-errata-check-new --prepared runs/physics-errata-review-new
```

The existing authenticated source cache and prepared Physics edition are required. The published local review is `runs/physics-errata-review-v1`. Only the cited passages and their directly repeated summary/glossary statements were inspected in this follow-up. Other statements, worked examples and diagram dependencies are not certified. New-source admission and actual language/retention measurements remain necessary.
