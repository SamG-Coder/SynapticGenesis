# Shared workspaces for evaluation and prefix context

Status: **candidate builds and CPU protocol checks passed; GPU compatibility
pending**. This extends the [live borrowing constructor](live-view-construction.md)
on the same research branch. The running studies retain their original runtime.
The [verification record](../reports/shared-evaluation-views-host.json) separates
executed checks from the native comparisons still to run.

## Remaining allocation paths

The language probe scorer previously cached every requested prompt length. Each
view initially allocated a full learning model and then shared only weights,
Adam moments and recurrent state. Its own gradient and decay arrays remained
allocated. At 411,028,496 parameters those two retained FP32 arrays cost
**3,288,227,968 bytes (3.0624 GiB) per cached length**, excluding activations.
The cache had no count bound.

`probes::Scorer` now uses the existing `LiveViews` cache, sharing all five primary
parameter arrays and retaining at most four shapes. Each score or greedy decode
finishes using its view before the next view lookup can evict it. The CUDA
synchronization at eviction is unchanged from live execution. Strict FP32
scoring, zero-state answer scoring and reset-before-greedy behavior are retained.

Other sequential callers now use the same constructor:

| Caller | Recurrent state ownership |
| --- | --- |
| Batch-training burn-in prefix | Shared with the learner |
| Context benchmark target | Shared with the prefix model |
| Decode benchmark captured model | Independent from the regular decoder |
| Delayed-cue evaluation prefix | Independent from the learner |
| Delayed-cue evaluation query | Shared with the evaluation prefix |

These callers use the existing forward computation. Prefix context remains
detached from target-window backpropagation. Parameter/optimizer sharing does
not introduce concurrent execution or a new learning rule. Legacy sharing APIs
remain available to numerical controls and frozen-owner rejection tests.

The four-view bound is a count limit, not a byte budget. Long prompts can still
require large activation buffers, and no claim is made that every allowed
411M prompt/batch shape fits on a 16 GB GPU. Actual allocation and learning
measurements are still required. Standalone sampling and frozen teachers retain
their existing allocation policies in this stage.

## Verification boundaries

The candidate runtime and the optional allocation diagnostic compile. The
diagnostic's CPU layout/constructor preflight passes again: four sizes and
32 rejected constructors without CUDA work. Its pending native self-test now
also exercises six scorer caches across 54 shape accesses, checking shared
parameter pointers, unchanged owner data, the four-shape limit and exact
retained float-buffer accounting after eviction.

`tests/view_callers.py host-test` prepares and audits a separate compatibility
protocol without launching a model. It passes 19 deliberate comparator
corruptions: changed scores, samples, checkpoint bytes, output fields, missing
timing fields and broken probe-order independence are rejected.

The pending `run` protocol uses six tiny synthetic model architectures and both
training/decoding precision settings. Scoring and context measurements retain
their original strict FP32 behavior. No book checkpoint is modified or learned
from, and no synthetic fixture/model is admitted to general training.

| Native command | Planned calls | Required comparison |
| --- | ---: | --- |
| `train` | 72 | Exact warm/reset prefix and resumed checkpoints |
| `context-bench` | 24 | Exact reported context/state-ablation results |
| `language-probes` | 96 | Exact scores, predictions and greedy bytes in two probe formats and both item orders |
| `decode-bench` | 24 | Exact sampled bytes and numerical fields; timing excluded |
| `memory-bench` | 24 | Exact cue results, checkpoint and training metrics; timing excluded |

The 240 calls yield 204 comparisons, including order-independence checks within
each runtime. Reports are compared after each matching pair and again at the
end. Fixture and executable hashes are recorded; checkpoint/sample/result
hashes are saved on successful completion. These are compatibility checks, not
controlled speed measurements or language-quality evaluations.

After the running GPU studies finish, add this command to the existing native,
live/restart and allocation gates, using a fresh output directory:

```powershell
D:\SBILM\.local\venv\Scripts\python.exe -X utf8 tests\view_callers.py run --out runs\shared-view-callers-cuda-v1 --old-exe D:\SynapticGenesis\build\synapticgenesis.exe --new-exe build\synapticgenesis.exe
```

The original `9dcd605` candidate executables are preserved under
`build/preserved-live-view-v1/`. Neither this compatibility protocol nor a new
capacity study has been queued automatically.
