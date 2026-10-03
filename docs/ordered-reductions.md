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
