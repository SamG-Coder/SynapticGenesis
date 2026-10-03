# Borrowing parameters during live view construction

Status: **CUDA ownership, numerical, checkpoint/restart and caller checks
passed; 411M allocation measured**. The
[CUDA verification record](../reports/shared-view-cuda-validation.json) identifies
the tested sources, executables and raw evidence. The earlier
[host verification record](../reports/live-view-construction-host.json) preserves
the preparation stage. The [evaluation extension](shared-evaluation-views.md)
uses the same constructor and bounded cache. Existing studies retain their
pinned binaries and inputs. The separate [411M capacity result](411m-capacity-results.md)
covers actual learning and generation, beyond allocation alone.

The [411M capacity calculation](next-capacity.md) identified temporary duplication
of five parameter arrays when constructing a speaker, document tail or replay
view. The existing runtime first creates a complete model, then releases its
parameter workspace while attaching the owner's arrays. At 411,028,496
parameters, that temporary workspace alone is **8,220,569,920 bytes (7.656 GiB)**.
This is the predicted reduction in explicit float allocations, not a measured
whole-process VRAM saving.

`Model(Model &owner, int time, ModelViewState)` constructs the view with empty
weight, gradient, Adam-moment and decay buffers. It allocates its own activation
scratch and then shares the owner's five parameter arrays and synaptic memory.
`LiveEngine` uses this constructor for its speaker; `LiveViews` uses it for tail
and replay shapes. Both use the existing forward, backward and optimizer
implementations. The observer used by the allocation diagnostic is absent from
ordinary builds.

State ownership follows the existing sequential execution contract:

- The root owns its learning workspace. A frozen model retains its separate
  forward-only allocation policy and cannot become a learning view's owner.
- Speaker and tail views share recurrent state with the root. Replay views keep
  independent recurrent state. All views own their temporary activations.
- Shared pointers preserve parameter, optimizer, synaptic and shared recurrent
  allocations if the original owner is destroyed first.
- A view never initializes the owner's moments or decay mask. Batch/context
  validation now runs before any CUDA allocation, including standalone models.
- Tail and replay caches still hold at most four shapes each. Shared recurrent
  buffers are still briefly constructed before aliasing; this change targets
  the much larger parameter workspace rather than every temporary allocation.

Checkpoint format, equations, precision settings, replay selection and view
eviction policy are unchanged by the patch. Native comparisons now verify
exact learned-state compatibility on the declared small fixtures.

## Completed validation

The native runtime and separate `synaptic-view-allocation-probe` compile with
CUDA 13.3.73 and MSVC 19.51 for architecture 120. The diagnostic's `host-test`
checks four associative layouts (2M, 27M, 105M and 411M) and rejects 32 invalid
batch, context and model-dimension constructors without requesting CUDA work.
Its allocation observer records zero successful float-buffer allocations.

After the width study exited successfully, the declared GPU gates ran
sequentially. All of the following passed:

1. Twelve legacy-versus-borrowing allocation cases and 48 exact
   forward/backward/optimizer/state comparisons across all six cells, batches
   one/three, FP32/TF32 and shared/independent recurrence. Nonzero optimizer and
   synaptic history, destruction of the original owner, 26 owner/shape guards
   and release of all observed float allocations are covered.
2. Six bounded scorer caches across 54 shape accesses, including eviction,
   unchanged owner data and exact retained float-buffer accounting.
3. All 18 existing CTest checks, including frozen-forward and graph behavior.
4. Forty-eight `live` commands against the preserved runtime: complete
   checkpoint and speech identity across ragged streams, followed by exact
   resumed checkpoints for all six cells in FP32 and TF32.
5. The evaluation extension's 240 native commands and 204 exact comparisons.
6. Allocation-only measurements through `LiveEngine` and both bounded caches,
   including eviction: explicit float-buffer peaks of 2,738,903,280 bytes
   (2.551 GiB) at 105M and 9,476,925,680 bytes (8.826 GiB) at 411M. All observed
   float allocations were released. These particular commands perform no
   forward pass or learning.

The allocation diagnostic counts successful `Buf` float allocations and final
ownership releases. It excludes raw integer token buffers, CUDA/cuBLAS internals,
graphs, host memory and other applications. `cudaMemGetInfo` values at named
boundaries are snapshots, not continuous whole-process peak measurements.

From this worktree, these commands reproduce the core checks with fresh paths:

```powershell
.\build\view-allocation-probe\synaptic-view-allocation-probe.exe self-test --out runs\view-allocation-cuda-v1
ctest --test-dir build --output-on-failure
D:\SBILM\.local\venv\Scripts\python.exe -X utf8 tests\live_views.py --out runs\view-allocation-compat-v1 --old-exe D:\SynapticGenesis\build\synapticgenesis.exe --new-exe build\synapticgenesis.exe
.\build\view-allocation-probe\synaptic-view-allocation-probe.exe measure --channels 1024 --hidden 4096 --layers 8 --out runs\view-allocation-105m-v1
.\build\view-allocation-probe\synaptic-view-allocation-probe.exe measure --channels 2048 --hidden 8192 --layers 8 --out runs\view-allocation-411m-v1
```

Use fresh output directories. Preserve executables referenced by earlier
studies; use explicit paths when comparing or running the validated runtime.
