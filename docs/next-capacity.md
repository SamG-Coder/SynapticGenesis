# 411M sizing and the original allocation limit

The selected width target uses 2,048 channels, 8,192 spiking neurons per
layer and eight layers: **411,028,496 parameters and 65,536 spiking neurons**.
These dimensions are within the current `Layout` limits. This note records
the original static sizing and construction limit. The subsequent
[shared-view change](live-view-construction.md) passed CUDA compatibility and
allocation checks, and the [411M capacity run](411m-capacity-results.md) now
establishes actual short live learning and generation on the RTX 5080.
The [calculation record](../reports/next-capacity-layout.json) retains the
source hashes, formula and agreement with three measured model layouts.

| Shape: channels / neurons per layer / layers | Parameters | Total spiking neurons | Five primary FP32 parameter arrays |
| --- | ---: | ---: | ---: |
| 256 / 512 / 4 | 1,951,624 | 2,048 | 0.0364 GiB |
| 512 / 2,048 / 8 | 27,260,432 | 16,384 | 0.5078 GiB |
| 1,024 / 4,096 / 8 | 104,851,472 | 32,768 | 1.9530 GiB |
| 2,048 / 8,192 / 8 | 411,028,496 | 65,536 | 7.6560 GiB |

The arrays in the last column are weights, gradients, two Adam moments and
the decay mask. They exclude activations, recurrent caches, token buffers,
CUDA/cuBLAS allocations, graphs and other applications. They are not complete
model memory or measured process peaks. The last shape doubles both widths
relative to 105M, giving twice the spiking neurons and approximately four
times the learned parameters at the same depth.

## Construction limit identified before the change

The earlier [bounded live memory change](bounded-live-memory.md) shared
weights, optimizer state, gradients and decay masks after constructing each
view. `LiveEngine` first constructed both `root` and `speaker`; only then did
it call `LiveViews::share`. `LiveViews::get` similarly constructed a complete
learning `Model` before attaching it to the root.

Consequently, that construction path allocated a second set of all five primary
arrays temporarily. For the proposed shape, those two sets alone total
**16,441,139,840 bytes, or 15.3120 GiB**, before any activation buffers or CUDA
overhead. The earlier 105M resident-array measurement did not capture this
construction peak. No 411M out-of-memory experiment has been performed.

Execution views now acquire the owner's shared buffers during construction,
while allocating their required scratch and recurrent state. Independent
root/frozen ownership, replay resets, the common forward path and checkpoint
bytes are preserved in the declared native comparisons. The
[verification record](../reports/shared-view-cuda-validation.json) separates
the compatibility, allocation and actual learning checks. This note remains
the record of why the construction change was needed.

## Capacity, data and learning quality

The completed [2M/27M/105M comparison](prose-size-results.md) used 55 selected
training books with 4,817,047 word occurrences. The larger networks did not
improve final validation under the original common 1,024-slot replay policy. The extra capacity
therefore needs both an appropriate learning configuration and a larger,
useful corpus; increasing neuron count alone is not evidence of progress.

The subsequent [broader-replay comparison](prose-replay-capacity-results.md)
improved both tested sizes; with 16,384 slots, the final validation means were
1.688728 for 2M and 1.668709 for 105M. The larger model can therefore benefit
from a different learning policy in this run. This remains one exploratory
seed and does not establish 411M language quality. Memory feasibility and
short-workload speed are now measured separately; full-curriculum learning
and quality at 411M remain untested.

The Chinchilla paper's abstract reports that compute-efficient transformer
training required increasing data along with parameter count. That supports
joint capacity/data planning, but supplies no validated token-to-parameter
ratio for this byte-based spiking architecture.
[Hoffmann et al., 2022](https://arxiv.org/abs/2203.15556).

The TinyStories abstract reports coherent constrained story generation with
models below ten million parameters on its synthetic corpus. This is evidence
that small-model language quality also depends on the task and data; it does
not establish broad general competence or performance of this spiking model.
[Eldan and Li, 2023](https://arxiv.org/abs/2305.07759).

Only those abstracts were inspected for this note. Neither paper nor its
associated dataset has been added to training. The selected arithmetic
[worked examples](prealgebra-worked-selection.md) are a small supplementary
edition, not the large increase in reading data needed for further scaling.
