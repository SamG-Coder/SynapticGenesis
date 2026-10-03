# Borrowing parameters during live view construction

Status: **research branch; native builds and CPU preflight passed; CUDA checks
pending**. This candidate is not yet merged into the training runtime. The
[host verification record](../reports/live-view-construction-host.json) identifies
the first candidate's tested sources, executables and completed checks. The
[evaluation extension](shared-evaluation-views.md) records the subsequent
candidate and additional pending caller/cache checks. The existing replay and
early-width studies keep their pinned binaries and inputs.

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
eviction policy are unchanged by the patch. Exact learned-state compatibility
still requires the pending GPU checks; source inspection is not that proof.

## Completed validation

The native runtime and separate `synaptic-view-allocation-probe` compile with
CUDA 13.3.73 and MSVC 19.51 for architecture 120. The diagnostic's `host-test`
checks four associative layouts (2M, 27M, 105M and 411M) and rejects 32 invalid
batch, context and model-dimension constructors without requesting CUDA work.
Its allocation observer records zero successful float-buffer allocations.

The existing live-view comparison accepts explicit old/new executable paths;
its default paths and 48-command numerical/restart protocol are preserved.
Python compilation and argument parsing passed. CTest discovers all 18 existing
runtime checks; discovery does not mean those tests have run on this candidate.

## Pending validation

After the existing GPU studies finish, run these checks sequentially. No new
GPU experiment has been queued for this change.

1. The diagnostic's `self-test`: 12 legacy-versus-borrowing allocation cases and
   48 forward/backward/optimizer/state comparisons across all six cells, batches
   one/three, FP32/TF32 and shared/independent recurrence. The fixture uses nonzero
   optimizer and synaptic history, destroys the owner before later updates,
   and checks invalid/frozen-owner rejection. These counts describe planned
   coverage, not passing CUDA results.
2. All 18 existing CTest checks, including frozen-forward and graph behavior.
3. `tests/live_views.py` against the preserved runtime: complete checkpoint and
   speech identity across ragged streams, then exact resumed checkpoints.
4. Allocation-only measurements at 105M and 411M, through the production
   `LiveEngine` and both bounded caches, including eviction. This does not
   initialize learned weights, execute a forward pass or establish training fit.
5. Only after those gates pass, a separate controlled learning/generation
   capacity run can establish usable 411M memory and speed.

The allocation diagnostic counts successful `Buf` float allocations and final
ownership releases. It excludes raw integer token buffers, CUDA/cuBLAS internals,
graphs, host memory and other applications. `cudaMemGetInfo` values at named
boundaries are snapshots, not continuous whole-process peak measurements.

From this worktree, the future GPU commands begin with:

```powershell
.\build\view-allocation-probe\synaptic-view-allocation-probe.exe self-test --out runs\view-allocation-cuda-v1
ctest --test-dir build --output-on-failure
D:\SBILM\.local\venv\Scripts\python.exe -X utf8 tests\live_views.py --out runs\view-allocation-compat-v1 --old-exe D:\SynapticGenesis\build\synapticgenesis.exe --new-exe build\synapticgenesis.exe
.\build\view-allocation-probe\synaptic-view-allocation-probe.exe measure --channels 1024 --hidden 4096 --layers 8 --out runs\view-allocation-105m-v1
.\build\view-allocation-probe\synaptic-view-allocation-probe.exe measure --channels 2048 --hidden 8192 --layers 8 --out runs\view-allocation-411m-v1
```

Use fresh output directories. Keep the candidate separate from preserved
executables until its CUDA and compatibility results have been reviewed.
