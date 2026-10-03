# Faster associative recurrence with unchanged learning trajectories

The associative cell now keeps each forward thread's four matrix cells in registers and caches the current backward inputs in shared memory. The backward matrices use an extra padding column. The equations, accumulation order, persistent state and checkpoint format remain unchanged. Learning and generation still use the same forward kernel in `src/associative_memory.cuh`.

## What was tested

The original executable from the [early learning screen](associative-memory.md) was archived before editing. Its SHA-256 is `41a43f3ca3a69ba0770c2dd3a81dade9771ba35830730c07422693164c1dac88`. Both binaries resume the same seed-1337 checkpoint at observation 10,000 for seven paired 2,000-observation timing repetitions. Execution order alternates. Three further optimized runs reproduce the original 24,000-observation binding trajectories, one for each seed. Generation uses three paired repetitions, each with seven rounds of 512 bytes and strict FP32.

These are sequential, unprofiled native processes on an RTX 5080 with CUDA 13.3 and MSVC 19.51. Live elapsed time includes replay, speech, logging and checkpoint writes, excluding setup and final assessment. Decode timings exclude model loading and graph capture. The complete checkpoint files are compared, including learned weights, optimizer, recurrent memory, RNG and replay history. Repeated timing runs are not additional independent learning seeds.

## Result

The optimized live loop takes **14.3% less time**, a 1.167-times speedup. Graph decoding improves slightly; ordinary decoding is effectively unchanged in this small timing sample. Kernel savings do not translate into a comparable generation gain, where other operations and dispatch still contribute.

| Measurement | Original | Optimized | Time change |
| --- | ---: | ---: | ---: |
| 2,000 live observations, median of 7 | 6.6292 s | 5.6808 s | 14.3% lower |
| Graph generation, median of 3 seven-round medians | 152.496 microseconds/byte | 149.554 microseconds/byte | 1.9% lower |
| Ordinary generation, same repetition scheme | 488.595 microseconds/byte | 489.116 microseconds/byte | 0.1% higher |

All fourteen short runs produce the same complete checkpoint bytes. The three optimized continuations from observation 10,000 to 34,000 also reproduce the original final checkpoint files exactly. All paired generation samples match, and graph versus ordinary decoding has zero state/logit difference in the measured rounds. There is no change to the previously measured learning results at this endpoint.

All **16 native suites** pass. The seven exported associative float fixtures remain byte-identical to the archived executable. Both independent CPU autograd fixtures pass: maximum unweighted/weighted logit error is `8.94e-8` / `1.04e-7`, and gradient error is `3.73e-8` / `1.19e-7`. The optimized executable SHA-256 is `46de182c3454eceeeb27d3c7120bb969e86da9c2304c67f1614a87ca095df86e`.

The [full report](../reports/association-runtime.json) includes every repetition, kernel case, complete-checkpoint identity and numerical fixture. It also records the profiler limitation and timing boundaries. This improves the associative implementation's cost; it does not remove the substantial cost difference from the simpler selective cell in the original screen.

## Kernel investigation

[NVIDIA's shared-memory guidance](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#shared-memory-and-memory-banks) motivated the initial extra-column hypothesis. Padding alone was **slower**. Caching the current query, key, value, upstream gradient and normalization factors helped; combining that cache with padding helped further. This interaction means the simple padding hypothesis was insufficient.

The table gives seven-round median CUDA-event timings for one sequence of 128 bytes, with 100 recurrence invocations captured per round. It excludes projections, normalization, optimizer work and host transfer.

| Reverse recurrence | Microseconds | Change from packed, uncached |
| --- | ---: | ---: |
| Packed, uncached control | 259.936 | — |
| Padding alone | 266.128 | 2.4% slower |
| Cached inputs, packed | 221.329 | 14.9% lower time |
| Cached inputs and padding | 187.642 | 27.8% lower time |

The forward register variant takes 99.622 microseconds versus 111.726 for the shared-matrix control, about 10.8% lower time. Both forward variants in this isolated comparison include hoisted inputs and loop unrolling; that shared control is not the exact old executable. Only the archived-executable live/decode comparison measures the full before/after change.

All ten cases (batch sizes 1 and 4; sequence lengths 1, 17, 74, 128 and 512) produce byte-identical forward history, reads and final states, and byte-identical gradients across all four backward variants. The fixtures include nonzero incoming memory, near-zero vectors where normalization epsilon matters, and saturated gates. The optimized backward block adds 784 bytes of temporary shared storage; it adds no global or persistent model allocation. Forward execution has an explicit 256-thread launch contract.

Nsight Compute could not access this machine's GPU performance counters (`ERR_NVGPUCTRPERM`). No profiler-affected timing is used. Timings and disassembly support the implementation experiment; counter measurements did not establish a bottleneck breakdown or an energy benefit.

## Reproduce

The benchmark module `src/associative_bench.cuh` retains all layout controls and is separate from runtime orchestration. It also supplies the sixteenth native CTest suite. `scripts/association_runtime_experiment.py` owns the archived-binary comparison; `scripts/summarize_association_runtime.py` authenticates and publishes its evidence.

```powershell
.\build.ps1
.\build\synapticgenesis.exe association-layout-bench --out runs/association-final-kernels --rounds 7 --iterations 100
python scripts/association_runtime_experiment.py --out runs/association-runtime-panel
python scripts/summarize_association_runtime.py
```

Use fresh experiment directories. Reproducing the binary comparison requires the preserved original executable and learned checkpoints from the early screen. Those local binaries and checkpoints are excluded from the public source repository; source commits, executable identities and checkpoint hashes are recorded with the results.

The performance change does not establish better learning, general conversation, biological development or superiority over a conventional language model. The associative cell remains experimental and LIF remains the default. The original learning screen and its quality results remain attributed to the original executable.
