# Local validation — 3 October 2026

Host: Windows, NVIDIA RTX 5080, CUDA 13.3, MSVC 19.51. Model execution, learning, mutation and selection use the native C++/CUDA executable.

- Clean CMake/Ninja build succeeded.
- All seven native CTest suites passed: base cell, live state, graph decoding, replay, adaptive cell, consolidation and evolution.
- Independent CPU autograd checked LIF with reset and nonzero incoming state and ALIF. Maximum gradient discrepancy was below 4.5e-8.
- Independent scalar consolidation/AdamW reference passed; maximum updated-weight discrepancy was below 2.4e-7.
- Width growth and whole-block inheritance fixtures matched their expected forward results for both cells. Added neurons learned nonzero outgoing weights. Newborn eligibility, improvement ceilings, elite ranking, seeded mutation, size caps and lineage serialization passed.
- The CLI burn-in continuation test passed for both cells and both prefix policies.
- Prepared source/stage hashes and split composition were independently audited. No exact normalized paragraph of at least 120 characters overlaps the prepared splits. This does not test shorter overlap or paraphrases.

The [corpus manifest](foundations-v1-manifest.json) records four training readers (261,281 bytes including separators), one validation reader (32,348 bytes), and one test reader (113,004 bytes).

A fresh default-size founder trained for 1,200 updates per curriculum stage, 3,600 total, seed 1337, batch 16, context 128, base learning rate 0.0006 and TF32 training. A second founder used seed 2026 and 3,600 updates on the pooled training set. This is a smoke experiment, not a controlled curriculum comparison: seeds and content exposure differ.

The first inheritance demonstration produced two 1,449,472-parameter children from 1,186,304-parameter parents, expanding hidden width from 512 to 640. The demonstration deliberately used elevated parental growth probabilities (1.0 and 0.5) to exercise expansion. All children require further training and evaluation; crossover alone did not improve their held-out loss.

These checks establish mechanism and pipeline behavior. They do not establish conversational competence, human-like development, inherited intelligence, long-term retention or superior generalization. Population resource/lifespan validation is added in the next stage.
