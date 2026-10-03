# Reproducible native learning updates

Earlier live experiments exposed a confound: identical initial arrays and source/replay histories could lead to different learned models. The [stopped replay pilot](../reports/replay-control-pilot.json) preserves that observation. Fixed-order gradient reductions now address a source of arithmetic variation while retaining the shared native learning/generation runtime.

## What changed

The [gradient reduction module](../src/gradient_reductions.cuh) gives each destination one writer and a defined addition order:

- Embedding gradients gather matching byte positions in increasing source order.
- RMS normalization computes each input gradient as before. Separate blocks in the same kernel reduce gain gradients over eight row lanes with a fixed tree. The input and gate paths accumulate in stream order.
- Each neuron writes its own gradient for a single sequence. Batched training writes per-sequence partials, then combines them in increasing batch order. The equations, surrogate derivative and detached reset are unchanged.

The live single-sequence path adds no persistent GPU allocation or checkpoint bytes. Batched execution adds `4 * batch * hidden` bytes for LIF, or three times that for cells with secondary state, in one reused scratch buffer. Population admission accounts for it and tests compare the estimate with actual explicit buffers.

The cuBLAS handle explicitly disallows its optional atomic implementations. This is already cuBLAS's default and does not establish a new matrix multiplication algorithm. Generation and training still hand off at completed boundaries; there is no concurrent unsynchronized weight mutation. Live session reports and batch start records identify `gradient_reductions: ordered_v1`.

[NVIDIA's reduction documentation](https://developer.nvidia.com/blog/controlling-floating-point-determinism-in-nvidia-cccl/) explains why unordered floating-point atomics can give different answers and why a fixed reduction tree can repeat on one GPU. [The cuBLAS reproducibility contract](https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility) also depends on toolkit, device and stream/workspace conditions. We test this implementation on the local GPU; neither reference proves our complete learner reproducible on every machine.

## Compatibility and interpretation

The parameter layout, neuron architecture IDs and checkpoint formats are unchanged. Old checkpoints remain readable and can continue learning. Changing the addition order changes rounding, so their future trajectory can differ from continuation with the old executable. Exact continuation comparisons require the same executable, GPU/toolchain, inputs and execution settings.

Repeatability is an experimental control. It does not improve the learning objective, establish better language skills, remove seed dependence or reproduce biological plasticity. Existing learning results remain evidence for the binaries identified by their reports; they are not silently replaced by new runs.

Initialization, replay selection, generated speech and evolutionary mutation still use their seeded random choices. Fixed arithmetic makes a repeated choice sequence easier to compare; different seeds and selected experiences can still produce different models.

## Reproduction

Preserve an executable built from the preceding commit `49c543b`, then build the current source and run:

```powershell
.\build.ps1
python scripts/reduction_experiment.py --legacy path/to/previous/synapticgenesis.exe --out runs/ordered-reduction-comparison
```

The driver uses already prepared selected `binding-v2` material, writes its protocol before learning and records executable/source/checkpoint hashes. Three fresh runs per executable observe 6,000 reading windows with replay and graph speech. A one-stage policy control checks that equivalent reservoir and stage-memory sampling give the same numerical history. Long continuations from one real old checkpoint compare uninterrupted learning with a process restart at observation 17,403. Batched training and decode measurements are separate from the live timing. The reserved language test partition is unused.

The native reduction suite includes scalar double-precision references, repeated-byte collisions, channel tails, and complete repeated updates for all five cells in strict FP32 and TF32. The independent CPU autograd fixtures validate the full derivative and Adam update, including incoming recurrence and weighted answer targets.

The [mechanism validation report](../reports/ordered-reductions-validation.json) records fourteen native suites, eleven CPU autograd fixtures and the policy/extension/population CLI checks for the identified executable. Forty curriculum-extension comparisons and first-stage replay conversion have zero saved-state discrepancy in this run. Dead-member learning remains rejected and archived models remain readable.

## Completed comparison

The [29-command experiment](../reports/ordered-reductions-comparison.json) compares the preceding executable with the ordered version on an RTX 5080, CUDA 13.3 and driver 616.64. The [compact summary](../reports/ordered-reductions-summary.json) includes full checkpoint hashes, exposure checks and every timing sample. This is repeated execution of seed **1337**, not a three-seed learning-quality study.

With frozen input and weights, the old and new forward logits match bit for bit. Repeating the old backward pass changes normalization-gain gradients by up to 4.66e-10 without regularization, or 9.31e-10 with the activity cost. The ordered backward pass repeats exactly. These are small numerical differences, consistent with changed addition order; they are not evidence that the old gradient formula was wrong. The eleven CPU autograd fixtures pass with a maximum plain-gradient error of 6.90e-8 and regularized error of 7.55e-8.

Over 6,000 real reading observations, the old executable produces three different learned checkpoints despite identical seed, initialization and selected input. Compared with its first run, maximum individual weight differences reach 0.40008 and recurrent-state differences reach 53.77. Held-out reader loss ranges from 2.23010 to 2.25138. All three ordered runs produce the same complete checkpoint and loss 2.24200. The ordered result lies inside the old range; this is a repeatability improvement, not a demonstrated reader-quality gain.

The equivalent one-stage reservoir/stage-memory control has identical replay descriptors, replay RNG and exposure counters in both binaries. The old model arrays diverge; the ordered arrays match exactly. Policy metadata differs by design between replay modes, so those two complete checkpoint files are not expected to have the same hash.

Both long continuations start from the same **old executable's** actual 6,000-observation checkpoint, preserving its learned weights, moments and replay history while admitting balanced replay. One then runs uninterrupted to 130,000 observations; the other stops at 17,403 and resumes in another process. Their complete final files are byte-identical:

```text
SHA-256: 6e8b41bf60444a5c551f0f8dd0388a163f8e32cb4f4814bedcd7f3d5131258eb
130,000 source observations
162,500 optimizer updates including replay
24,960 generated bytes
```

This checks the full state, not just a final loss: parameters, Adam moments, recurrence, source position, replay descriptors/counters and saved RNG. Final earlier-reader loss is 2.67059 versus the ancestor's 2.24613, so residual forgetting remains. The regular batched runs also produce identical ordered checkpoints across all three repeats at each tested shape. A preliminary 34,000-observation pilot passed the same restart check before the final protocol was run.

## Measured cost

All training uses the selective cell at C256/H512/L4, 1,716,736 parameters, with TF32 enabled. The live comparison includes replay and periodic graph speech. Medians are from three rotated-order repeats per executable; these are local measurements, not precise population estimates.

| Measurement | Previous reductions | Ordered reductions | Measured change |
| --- | ---: | ---: | ---: |
| Live loop, 6,000 reading observations | 9.295 s | 9.334 s | +0.42% |
| Batch 8 × context 128, 400 updates | 0.559 s | 0.570 s | +1.98% |
| Batch 16 × context 256, 400 updates | 1.310 s | 1.375 s | +4.97% |
| Strict-FP32 graph decode median | 117.14 microseconds/byte | 116.02 microseconds/byte | Similar |

Live timing excludes startup and initial/final evaluation, and includes live updates, replay, speech, logs and checkpoints. Batch timing starts after initial evaluation and ends before the final evaluation/checkpoint; it includes intermediate metric logging. The first legacy small-batch sample was slower (0.742 s); it remains in the report with the other samples. Repeated models see equal selected input and update counts, although the old and new learned trajectories differ because of rounding.

Decode uses the same old checkpoint in both executables, seven rounds of 1,024 generated bytes, and strict FP32. It includes host sampling, transfers and synchronization, excluding prompt warmup and capture. Ordinary and graph decoding agree exactly within each executable. The change targets backward arithmetic, and this timing variation does not establish an inference speed gain.

The single-sequence learner and checkpoint sizes do not increase. The two measured batch shapes add 48 KiB and 96 KiB of persistent reduction scratch, respectively. The tested live command also creates an optional batch-16 validation model, which allocates 96 KiB of this scratch even though held-out evaluation does not use backpropagation. These figures describe explicit buffers, not a measurement of total process or CUDA allocator overhead. Larger shapes, different GPUs/toolchains, concurrent training processes and cross-device bitwise reproducibility have not been established.

These checks remove an observed source of experimental variation while preserving the measured learning equations and live state. Consistent generalization, reduced forgetting and an evolutionary learning advantage still need separate demonstrations. The historical replay collector now rejects a mismatched executable before writing reports, preventing this build's identity from being attached to the previous learning study.
