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

The resource/lifespan stage also passed all seven native suites. The evolution test checks the GPU-buffer estimate against actual model buffer sizes for both neuron cells, old-age boundaries, dead-parent exclusion, credit admission and scarcity-dependent selectivity.

The [native population CLI integration result](population-cli.json) covers 12 invocations across four simulation ticks. A 1 MiB test credit limit admitted six children from a request for 32 and stopped at capacity. A zero-credit round admitted none. Both four-tick founders then died, remained archived and released their population credits. Untrained children were never admitted as parents. Changed evaluation data, duplicate round names and traversal IDs were rejected without advancing the persisted clock. An inherited child's training log correctly reports checkpoint continuation rather than random initialization.

## Learned family demonstration

The [family artifact](family-demonstration.json) records parent/child sizes, hashes, evaluations, mutations and birth decisions. The first two children each received 2,400 native updates on the pooled approved training sources, then were reevaluated on the population's original fixed protocol. The selection book was never a training target.

| Member | Parameters | Validation loss | Size-adjusted score | Required score below | Qualified to reproduce |
| --- | ---: | ---: | ---: | ---: | --- |
| Stronger founder | 1,186,304 | 2.00691 | 2.03063 | 3.00000 | Yes |
| First child | 1,449,472 | 1.99925 | 2.02824 | 2.02563 | No |
| Second child | 1,449,472 | 2.00010 | 2.02909 | 2.02563 | No |

Both children improved raw validation loss slightly, but their larger size and the required improvement margin kept them below the reproduction standard. The next round correctly selected the founders again. Its children are also generation 1; this experiment has not demonstrated a qualifying generation-2 child or an evolutionary performance advantage. This is useful evidence that the gate enforces its configured objective even when an offspring has a somewhat lower raw loss.

One final test-book evaluation of the staged founder's validation-selected checkpoint used 64 batches of 16 sequences with 128 targets (131,072 sampled target bytes). Byte cross-entropy decreased from 5.54386 at random initialization to 1.98813 at checkpoint step 2,700. Windows can overlap. This is evidence of language-pattern learning on the small curriculum; the sampled output remains incoherent. No claim of useful general conversation follows from this loss.

[Numerical artifacts](numerical-validation.json) collect the native checks and independent oracles. Raw books and model checkpoints stay local; the source allowlist, corpus hashes, code and compact evidence are public.

These checks establish mechanism and pipeline behavior. They do not establish conversational competence, human-like development, inherited intelligence, long-term retention or superior generalization.
