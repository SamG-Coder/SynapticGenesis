# Frozen teacher allocation

Frozen teachers supply probability targets during selected-source replay. They do not update their own parameters. Allocating their gradient arrays, Adam moments and backward scratch space consumes GPU memory without contributing to the shared forward computation.

`src/model.cuh` owns model buffers, recurrent state and the existing execution methods. `ModelBuffers::learning` remains the default for learners and live generation views. `ModelBuffers::frozen_forward` allocates weights and every buffer read or written by the common forward path, while omitting learning-only arrays. The forward kernels and `Model::forward_device` stay identical between modes. Teacher replay explicitly requests the frozen allocation mode; there is no second inference implementation.

The frozen mode omits parameter gradients, both Adam moments, the decay mask, target storage, loss/gradient arrays and backward scratch space. It retains the associative cell's `dy` buffer because that buffer also holds a forward readout. It also retains all common forward caches, including the associative matrix history currently written by that kernel. Further cache reduction would require another measured change to the forward implementation.

Full checkpoint save/load, target loss, backward, optimizer updates, teacher-loss application and parameter sharing require learning buffers. A rejected operation must leave weights, recurrent state and an existing checkpoint file unchanged. Teacher packaging continues to use the authoritative host checkpoint reader before uploading the validated frozen weights.

Live generation views retain learning buffers: the same one-byte view also learns short observed document tails. Their weights, optimizer history and recurrence still share the learner's state. The allocation change applies to frozen teachers only.

## Resource accounting

`src/model_memory.cuh` estimates the explicit allocations for both modes. Teacher admission uses the frozen estimate plus probability-target storage; it still checks the user budget and available GPU memory with the existing 64 MiB reserve. Estimates exclude CUDA/context/cuBLAS overhead. Population learner accounting continues to use the default learning mode.

The intended result is fewer resident teacher bytes while preserving complete learner behavior. Smaller explicit allocations do not establish lower latency, energy use, better dialogue or increased retention.

## Verification protocol

The native teacher-replay suite compares all six cell types at three batch/context shapes in strict FP32 and TF32. It checks exact logits, recurrent state, teacher probabilities, actual buffer sizes, rejected mutations and twelve CUDA graph cases. Existing replay, restart and lifecycle checks remain in place.

The learned continuation comparison includes every architecture and seed from the completed retention study. The original executable and new executable each resume the same 160,000-observation teacher branch for 512 additional observations. These include 128 replay updates and one 96-byte speech event. Complete checkpoint files and transcripts must match byte for byte, teacher contribution counters must increase equally, and parent checkpoints and immutable teacher bundles must remain unchanged. This is a compatibility check, not a second quality study.

```powershell
.\build.ps1
python tests/teacher_compatibility.py --out runs/frozen-teacher-ordinary-compatibility
python tests/teacher_large_smoke.py --controls runs/teacher-integration-compatibility --out runs/frozen-teacher-large
python tests/teacher_replay_cli.py --out runs/frozen-teacher-standalone
python tests/population_teachers_cli.py --out runs/frozen-teacher-population
python tests/frozen_teacher_compatibility.py --out runs/frozen-teacher-learned-compatibility
python tests/distillation_oracle.py
```

The independent objective oracle retains its strict cold first-Adam-step failure and returns an unsuccessful exit. A complete memory-sanitizer pass over the native teacher-replay suite is also required before publishing this allocation change. Observed continuation durations may be recorded, but shared desktop GPU use prevents interpreting them as an isolated speed benchmark.

## Completed allocation and compatibility results

The [validation report](../reports/frozen-teachers-validation.json) verifies lower explicit teacher allocations for every learned seed and architecture, with identical complete learner checkpoints and generated transcripts after the 512-observation continuation.

| One frozen teacher, batch 1 / chunk 128 | Original bytes | Frozen-forward bytes | Reduction |
| --- | ---: | ---: | ---: |
| Selective H512 | 43,547,648 (41.53 MiB) | 14,767,616 (14.08 MiB) | 66.09% |
| Selective H588 | 49,302,976 (47.02 MiB) | 16,646,336 (15.88 MiB) | 66.24% |
| Associative H512 | 52,161,184 (49.74 MiB) | 18,439,200 (17.58 MiB) | 64.65% |

The tested two-teacher bundle falls from **95,577,248 to 33,075,232 bytes** (91.15 to 31.54 MiB). Its uninterrupted/resumed results remain identical to the original runtime for all three established learner shapes. These counts include target buffers and exclude CUDA/cuBLAS overhead; the existing physical-memory reserve remains in force.

All **18 native suites** pass. The frozen-mode fixtures cover **36 allocation/forward combinations and 12 graph cases**, including exact logits, recurrent state, teacher probabilities and rejected learning/checkpoint mutations. The nine learned continuations preserve complete checkpoints, speech, update counters and immutable bundles. All 67 earlier numerical fixture files, three ordinary continuations, 67 standalone CLI actions and 25 population CLI actions also pass their compatibility checks.

A complete CUDA memory-sanitizer run over `teacher-replay-test`, including the new frozen-mode cases, reports **zero errors**. The independent objective oracle still returns failure for the same cold associative first-Adam-step discrepancy: **9.20519e-6 versus a 5e-6 tolerance**. All its gradient checks and the separate nonzero-history update fixture pass. This storage change does not resolve or reclassify that known numerical limitation.

The production forward method is also verified textually unchanged by the module extraction. Memory savings establish a smaller teacher working set with preserved tested behavior. They do not establish faster learning, improved retention or more coherent language.
