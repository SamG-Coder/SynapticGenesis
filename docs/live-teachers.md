# Frozen teachers in the shared live learner

SynapticGenesis can attach one or two frozen checkpoints to selected-source replay in native C++/CUDA. Teachers contribute next-byte distributions; the learner still receives the actual source targets and uses the same weights, backward computation and optimizer as its live generation stream. This is an implemented teaching mechanism. A measured benefit for retention, general dialogue or reproduction remains unproven.

The [earlier narrative study](narrative-learning.md) improved new-book loss while losing an earlier binding skill. Teacher-assisted replay is a candidate response to that observed regression. The [objective specification](teacher-objective.md) defines the probability mixture, temperature scaling and known numerical limits.

## Immutable selected-source bundle

`teacher-pack` validates full native checkpoints, copies them, and writes a manifest identifying the copies, model dimensions, objective settings and selected source. It imports no external pretrained model. Supply this project's learned checkpoints and an explicitly selected source edition:

```powershell
.\build\synapticgenesis.exe teacher-pack --teacher-a runs/parent-a/latest.ckpt --teacher-b runs/parent-b/latest.ckpt --data selected-prefix.dat --out runs/teachers --temperature 2 --strength .5 --mixture .5
.\build\synapticgenesis.exe live --curriculum curriculum.sg --replay stage --teacher-bundle runs/teachers --teacher-memory-mib 512 --out runs/learner --updates 1000
.\build\synapticgenesis.exe live --resume runs/learner/latest.ckpt --curriculum curriculum.sg --teacher-bundle runs/teachers --out runs/learner --updates 2000
```

These example paths must point to existing selected material and a schedule covering the requested updates. Omit `--teacher-b` for a single teacher. Teachers may have different cell types and widths because all models use the same 256-byte output vocabulary. Two entries must identify distinct checkpoint files by content; population entries must also identify distinct members. A frozen earlier self can be used for a retention experiment; it does not become a second biological parent.

The package contains `teachers.sg`, `source.dat`, `teacher-0.ckpt` and optionally `teacher-1.ckpt`. The selected source must be an exact, document-aligned prefix of the final cumulative curriculum edition. Earlier editions and later selected editions can therefore be eligible, but a teacher is queried only after the learner has actually observed a window and replay selects it. A selected future document does not itself generate an update. Partial document excerpts, changed source bytes and changed teacher snapshots reject. The FNV identities and native payload checksums detect consistency errors; they are not cryptographic signatures or proof of a model's complete training provenance.

The first attachment upgrades stage-replay state from live version 5 to 6 while retaining source exposure, weights, moments, recurrent state, replay descriptors, RNGs and optional consolidation history. A resumed learner requires the identical bundle, even when teaching is disabled. Ordinary read-only sampling and evaluation need only the checkpoint.

## Update and state boundaries

Each live tick first learns the next observed source window. At the configured replay interval it selects an already observed window using the existing stage policy. If that document is within the bundle's prefix, it evaluates the frozen teachers and adds their KL objective to the actual-byte objective. Existing answer emphasis weights both terms. Other replay windows use the ordinary objective. Teacher guidance changes neither the source cursor nor replay selection.

Replay has independent recurrent scratch while sharing the learner's weights and optimizer. The learner's live membrane, trace and optional associative matrix remain separate from replay state. Graph generation continues to read the updated shared weights at completed update boundaries. Generated speech is never supplied as a training target.

Teachers use the production `Model::forward` with strict FP32 and reset state for each replay window. Their parameters and moments never update. Each teacher has one allocation at the learner's chunk length. Short replay windows are padded with zero bytes; only the causal prefix provides targets. This bounds teacher scratch without allocating a model for every tail length. Fixed-shape future-padding perturbations test causality across all six cells. Changing GEMM shapes can introduce separate floating-point differences, so these tests do not compare a padded forward with a differently shaped forward and assume exact equivalence.

`--teaching off` pauses teacher contributions while preserving their identities and cumulative counters. The setting persists across resume until explicitly changed with `--teaching on`. A zero-strength bundle also uses the ordinary objective without allocating teacher GPU buffers. Temperature, strength, mixture and snapshots remain fixed for that checkpoint's policy. Replacing a teacher or retuning the policy midstream is not implemented; use a separately declared experiment branch from a checkpoint before admission.

## GPU resources

`--teacher-memory-mib` caps the **additional explicit teacher GPU buffers**, defaulting to 512 MiB. Admission sums each teacher's complete `Model` allocation at batch one and the learner's chunk length, plus `4 * chunk * 257` bytes for target probabilities and scratch. The calculation is shared with population model admission and includes the currently allocated teacher gradients and optimizer arrays, even though frozen teachers do not update them. A future inference-only storage optimization would need separate evidence.

Admission also requires the current free GPU memory to cover those buffers plus a 64 MiB reserve for library/runtime overhead. Insufficient memory rejects before a learner output directory or log is written. The budget covers teacher buffers, not all learner views, validation buffers, host checkpoint copies or other processes. `session.json` records required teacher bytes, the requested cap, free bytes at admission and actual assisted updates/pairs. CUDA/cuBLAS overhead is not included in the explicit buffer total. Disabled or zero-strength policies allocate zero additional teacher GPU buffers, but still validate bundle files on the host.

The population's virtual food credits still describe resident-equivalent members, while these additional teacher allocations have an explicit physical admission check. Population operations are locked for the session; the population clock and reproduction cannot advance concurrently with it. This is a bounded resident implementation, not predictive neuron paging.

## Population origins and old age

Registered teaching uses member IDs and the current canonical checkpoint:

```powershell
.\build\synapticgenesis.exe teacher-pack --population runs/population --teacher-a founder-a --teacher-b founder-b --data selected-prefix.dat --out runs/parent-teachers
.\build\synapticgenesis.exe population-live --population runs/population --id birth-child-0 --curriculum curriculum.sg --validation validation.dat --replay stage --teacher-bundle runs/parent-teachers --updates 1000
```

Package creation and first attachment require alive registered teachers. First attachment also requires the snapshot to equal that member's current canonical checkpoint. After attachment, a parent's own subsequent learning does not silently replace the child's frozen teacher. Every active population session verifies the member record and checks that the teacher is still alive at the locked simulation tick. The teacher may be a parent or another registered member; teaching origins do not rewrite biological-style lineage.

When a teacher dies of old age, active teaching rejects. An alive child can continue selected-source learning with the same bundle and explicit `--teaching off`; identities and prior contribution counters remain recorded. Re-enabling a dead teacher rejects. A dead child cannot start another population learning session even with teaching disabled. Archived checkpoints remain available for read-only sampling/evaluation.

Standalone bundles cannot be newly attached through `population-live`; population bundles require that command. Adopting an already teacher-assisted standalone founder into an active population teaching policy has no migration path yet. Population registration and evaluation can read that archive, but its original standalone bundle cannot be resumed as registered teaching. Fresh offspring inherit weights/settings through the existing procedure and start with empty teaching/live histories.

## Durable format and telemetry

Live extension 6 preserves the version-5 replay layout and appends 32 little-endian `uint64` policy words before optional SI arrays. The existing full live checksum includes this payload. `meta[31]` continues to describe only replay words. Architecture IDs remain independent from this live-format version.

| Words | Meaning |
| --- | --- |
| 0–3 | `SGTEACH1` magic, policy schema, active flag, teacher count |
| 4–6 | Manifest identity, selected-prefix identity, eligible document count |
| 7–10 | Online/replay counts at admission, assisted update/pair counters |
| 11–13 | Temperature, strength, mixture as canonical 32-bit float bits |
| 14–15 | Population-scope flag, replay-pair count at admission |
| 16–23, 24–31 | Per teacher: full-file identity, native payload identity, parameter count, C/H/L/cell, member-record identity |

Unused second-teacher words must be zero. Native loading validates lengths, settings, bounded dimensions, teacher identities and counter relationships before accepting a checkpoint. Bundle membership strings remain in the immutable manifest whose identity is in the checkpoint. Teacher penalty is logged separately from observed-source cross-entropy; neither its magnitude nor lower source loss alone establishes retained skills.

## Verification boundaries

The [integration report](../reports/live-teachers-validation.json) records the current executable, source hashes, native/CLI evidence and the retained numerical limitation. Checks cover exact resume with all six cells, weighted answers, optional SI and graph speech; short-window causality and fixed target allocation; source eligibility accounting; disabled/zero-strength exact controls; budget/corruption rejection; population locks, teacher death and child death. Ordinary large-shape continuation checkpoints and prior numerical fixtures are compared with the earlier executable's saved results.

The recorded run passes 18 native suites and 105 CLI actions across standalone teaching, population lifecycle, ordinary compatibility and a larger-model smoke. Twelve small cell/SI restart cases and all three established 1.7–1.95 million parameter shapes produce identical uninterrupted/resumed checkpoints. The larger shapes use TF32 learner math and strict FP32 teachers, with 95,577,248 additional explicit GPU bytes (about 91.2 MiB) for the tested two-teacher bundle. CUDA race checking reports zero hazards in the teacher probability and penalty kernels throughout the replay suite; the unfiltered run was stopped for instrumentation cost and has no complete result. These are mechanism checks, not a learning or speed comparison.

```powershell
.\build.ps1
python tests/teacher_replay_cli.py --out runs/teacher-replay-check
python tests/population_teachers_cli.py --out runs/population-teachers-check
python tests/teacher_compatibility.py --out runs/teacher-compatibility-check
python tests/distillation_oracle.py
```

The independent objective oracle still fails one strict cold first-Adam-step check at 9.21e-6 versus its unchanged 5e-6 tolerance; its parameter gradients and the separate nonzero-history fixture pass. Runtime/lifecycle checks do not erase that failure. No useful-language comparison has yet been run with this integrated teaching policy. A subsequent fixed comparison must report retention and new learning together, all declared seeds, actual samples, complete live cost and teacher memory.
