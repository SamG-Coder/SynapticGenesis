# Native spiking learner

## Signed LIF cell

The default `signed_lif_v1` model uses a fixed 256-byte vocabulary, byte embedding, four residual blocks and an output projection. Default channel width is 256 and each block has 512 spiking neurons, for 1,186,304 trainable parameters. There are no imported model parameters or pretrained tokenizers.

Each block computes:

```text
z[t]      = W_in * RMSNorm(x[t]) + b_in
beta      = sigmoid(learned_leak)
u[t]      = beta * (u[t-1] - stop_gradient(s[t-1])) + z[t]
s[t]      = +1 when u[t] >= 1, -1 when u[t] <= -1, otherwise 0
x_next[t] = x[t] + W_out * s[t] + b_out
```

Forward spikes are hard and signed. Backward uses a triangular surrogate:

```text
ds/du ~= 0.3 * (max(0, 1-|u-1|) + max(0, 1-|u+1|))
```

The reset contribution of the previous spike is detached. Independent CPU autograd checks both zero and nonzero incoming recurrent state. Training uses truncated backpropagation and AdamW. The optional activity cost penalizes mean absolute spike activity; reported language cross-entropy excludes that cost.

Weights, membrane state, normalization and optimizer state are floating-point. cuBLAS handles dense projections. `--fast` enables TF32 math in training; numerical checks and evaluations use strict FP32. The implementation does not establish a hardware efficiency advantage over dense language models.

Training uses [fixed-order gradient reductions](ordered-reductions.md) for embedding, normalization gains and batched neuron parameters. A focused CUDA module owns these reductions. Each destination has one writer, avoiding dependence on block scheduling for floating-point accumulation. Single-sequence live execution adds no persistent scratch; batch execution adds a small reused per-sequence buffer included in population memory accounting. Arithmetic repeatability is tested for the local GPU/toolchain, not promised across platforms or binary versions.

## Adaptive cell

`--cell alif` creates `signed_alif_v2` with an additional activity-adaptation state per neuron and two learned parameters per neuron:

```text
threshold[t] = 1 + softplus(learned_scale) * adaptation[t]
spike[t]     = sign-threshold(membrane[t], threshold[t])
reset[t]     = membrane[t] - stop_gradient(threshold[t] * spike[t])
adaptation[t+1] = rho * adaptation[t] + (1-rho) * abs(spike[t])
rho          = sigmoid(learned_adaptation_leak)
```

This adds 4,096 parameters at the default size. The adaptation state modulates excitability. It does not grow synapses, encode chronological age or implement structural development. State units are byte steps, not biological milliseconds.

## Filtered spike cell

`--cell trace` creates the experimental `signed_trace_lif_v3`. It keeps the default signed LIF membrane and adds a learned fading trace of each neuron's signed spikes:

```text
rho         = sigmoid(learned_trace_leak)
gamma       = softplus(learned_trace_gain)
trace[t]    = rho * trace[t-1] + (1-rho) * spike[t]
emission[t] = spike[t] + gamma * trace[t]
x_next[t]   = x[t] + W_out * emission[t] + b_out
```

Trace time constants initially span 8 to 1,024 byte steps; training adjusts decay and gain. Shared LIF parameters start identically for a given seed and size. The trace path adds 4,096 parameters at default size, giving 1,190,400 total. It adds one recurrent float per neuron and one cached emission per neuron per training position. Unlike ALIF's nonnegative adaptation state, this trace is signed and affects the output signal directly. Disabling its gain recovers ordinary LIF output.

This experiment is inspired by the persistence and different decay times of [postsynaptic responses](https://neuronaldynamics.epfl.ch/online/Ch3.S1.html). The byte-step filter is an engineering approximation; it does not model receptor chemistry or biological age. A long passive decay is not proof of useful memory.

Training differentiates through the trace using the same spike surrogate. The membrane reset remains detached, and the activity penalty applies to raw spikes rather than the continuous emission. Independent autograd checks include signed incoming trace state, weighted answer targets and regularization. Streaming, graph decoding, checkpoint restart, replay, SI, inheritance and function-preserving width growth support this cell. Its continuous emission cannot use ternary spike additions or the current indexed-spike paging benchmark; these paths reject it explicitly. The dense resident path remains active.

## Input-dependent trace readout

`--cell gated` creates the experimental `signed_gated_trace_lif_v4`. It uses the same membrane, spikes and stored trace as the trace cell, with an additional learned projection from the current block input:

```text
read[t]     = 2 * sigmoid(W_gate * RMSNorm(x[t]) + b_gate)
emission[t] = spike[t] + gamma * read[t] * trace[t]
```

The read coefficient lies between 0 and 2. Gate weights and biases initialize to zero, so the coefficient initially equals one and recovers the trace cell exactly at the same dimensions/seed. Initialization consumes no additional random numbers. The gate projection shares the block's normalized input, and both gradient paths contribute to the input and normalization gain. The nonzero-gate CPU fixture checks that second path explicitly.

At width 256, four blocks and 512 neurons per block this gives 1,716,736 parameters. Using 344 neurons gives 1,197,280, close to the default trace model's 1,190,400. It adds a cached gate value per neuron per training position and one gradient scratch buffer, but no additional recurrent state. Population accounting includes these buffers, and the learned gate rows/biases are inherited with their neurons. All cell types remain distinct breeding compatibility groups.

Input-dependent selection is motivated by [selective state-space research](https://arxiv.org/abs/2312.00752). This is a much smaller output-gating experiment over spiking state, not a Mamba implementation or a verified biological mechanism. The gate does not itself establish correct content retrieval. It remains optional; the default is LIF.

## Input-dependent trace retention

`--cell selective` creates `signed_selective_trace_lif_v5`. It keeps the signed LIF membrane and surrogate, and lets the normalized block input change each neuron's trace retention:

```text
gate[t]      = W_gate * RMSNorm(x[t]) + b_gate
retain[t]    = sigmoid(learned_trace_logit + gate[t])
trace[t]     = retain[t] * trace[t-1] + (1-retain[t]) * spike[t]
emission[t]  = spike[t] + softplus(learned_trace_gain) * trace[t]
```

The effective retention offset is the sum of the learned trace logit and gate bias. The gate's matrix and bias start at zero, recovering the ungated trace recurrence and its initial 8–1,024-byte base time constants. All initial parameter arrays match the read-gated cell at the same dimensions and seed. Their later dynamics differ: the selective coefficient changes the stored state, whereas the read gate scales the state only when emitting it.

This follows the distinction between temporal selection and output multiplication in [Gu and Dao, section 3.5](https://arxiv.org/html/2312.00752v2#S3.SS5). The implementation is a small input-dependent filter over signed spikes. It retains dense projections, floating-point state and surrogate-gradient training; it does not implement the complete Mamba architecture or a biological learning rule.

Its parameter, cache, recurrent-state and population-memory sizes equal the read-gated cell: 1,716,736 parameters at C256/H512/L4, with two persistent floats per neuron. Checkpoint architecture IDs remain distinct even though payload shapes match. Inheritance copies the retention-gate rows and biases with their neurons, and zero outgoing weights on added neurons preserve the newborn's function before learning.

Backpropagation uses the **next timestep's** retention factor to carry gradients from the future trace. Both the base trace logit and projected input gate receive the derivative of the current retention coefficient. CPU autograd checks nonzero gate matrices, signed incoming trace state, weighted answers, regularization and Adam updates. The [selective-trace protocol](selective-trace.md) records the measured language behavior; the default remains LIF.

## Fast associative memory

`--cell associative` adds a learned 32-by-32 fast matrix to each selective trace block. Normalized queries and keys, bounded values, and learned decay/write gates drive a delta-rule update during the common forward path. The matrix participates in graph decoding and is differentiated through during each training chunk. Its [equations, state contract, costs and research limits](associative-memory.md) describe architecture ID 6.

## Shared learning and inference

The live engine has execution views with different chunk sizes over the same weights, Adam moments and neuron state. Observed chunks update parameters; generation then sees those updates directly. CUDA graph decoding captures generation without changing the initial neuron state, and reads the current shared parameter allocation without recapture.

Training and generation are ordered. CUDA streams hand off at completed boundaries; simultaneous unsynchronized read/write access is not used. This is one interleaved learning-and-generation runtime. Gradients stop at chunk boundaries even though recurrent state persists.

The source cursor advances across every within-document next-byte pair, including short tails. The recurrent state resets at document boundaries. Prompt and generated bytes enter the same recurrent stream but never become training targets. Generation can therefore affect the following observed chunk's context. This is a declared experimental protocol, not an optimal learning schedule.

Replay stores document/offset/length descriptors for previously observed source windows. A default 1,024-entry reservoir uses 24 KiB of descriptors plus a policy header; source bytes are separately resident in CPU memory. Replay shares weights and optimizer history but starts its own recurrent state from zero. Its updates and byte exposure are counted separately. `--replay recent` provides a newest-window control; `--replay none` disables replay.

With a curriculum, optional `--replay stage` divides a fixed slot budget among introduced source stages, maintains a separate lifetime reservoir per group, and samples nonempty groups equally. Group ownership follows when the document was introduced, including later observations of an earlier source. [Policy, conversion limits and experimental protocol](stage-replay.md). The host-side `StageReplayView` validates the saved layout and `StageReplay` owns its mutations, while `ReplayMemory` retains shared scheduling. Both policies use the same native model update and inference code.

`--core-scale` scales the learning rate for embeddings and spiking blocks; the output head keeps the full rate. It does not eliminate backpropagation or optimizer bookkeeping.

An explicit version-3 curriculum may emphasize answer targets in selected lesson documents. Both live observations and replay use the same per-window normalized weighted loss, while inference and held-out evaluation stay unchanged. The policy is bound by the curriculum identity and restored with its document annotations. See [answer-emphasis semantics](live-curriculum.md). Independent CPU autograd checks the weighted gradients for all six cell types, including incoming recurrent state and activity regularization.

## Selective consolidation

Optional `--consolidation si` records a per-parameter path integral of task gradient times actual optimizer displacement. At each update and completed document boundary:

```text
path += -task_gradient * actual_weight_displacement
# At the document boundary:
importance += max(0, path) / ((weight-reference)^2 + damping)
reference = weight
path = 0
```

Later updates add `2 * strength * importance * (weight-reference)` to the task gradient before clipping and AdamW. The path integral records the unpenalized task gradient and actual displacement. This is an adaptation of synaptic-intelligence ideas to the current optimizer/document boundaries.

Reference, importance and path use three extra float arrays (12 bytes per parameter); a scratch task-gradient array adds another four bytes per parameter on the GPU. CPU serialization buffers require additional memory. Importance is an optimization statistic, not a verified measure of semantic knowledge or neuronal age. Small or zero-strength controls are necessary: preventing weight movement can also prevent useful learning.

## Checkpoints and storage

Checkpoints include a checked architecture version, metadata, parameters, Adam moments and a payload checksum. Architecture versions 1–6 represent LIF, ALIF, filtered-spike LIF, gated trace readout, selective trace retention and associative trace memory. ALIF/trace and read-gate/retention-gate pairs use equal-sized layouts with different semantics; checked versions prevent mixing them. The architecture version and live-extension version occupy separate header fields. Live extensions add recurrence/cursor/RNG state, then replay, then consolidation history. Corrupt lengths, mismatched dimensions, unsupported policies and invalid history are rejected. The diagnostic reader is not a replacement for the native loader's complete validation.

`live --resume` restores the complete saved stream and requires the same source corpus and prompt. With `--curriculum`, live extension v4 permits declared append-only source transitions and binds all scheduled editions to the checkpoint. It preserves replay descriptors, RNGs and optional SI history. Its policy header uses words 14 and 15 for the combined schedule/source hash and zero-based stage index; hyperparameter slot 7 stores the base learning rate. Optional SI arrays follow the replay payload only when policy word 9 equals 1. The checksum covers all policy fields, state and arrays. Earlier checkpoint versions remain readable.

Live extension v5 retains that common header and adds grouped replay: one group-count word, five words per source group (exclusive document end, observed windows, stored windows, replay updates and replay target pairs), then grouped three-word descriptors. It requires replay mode 3. The bounded loader verifies group ranges and counters before accepting it, including for read-only inference. Ordinary reservoirs can convert during the first curriculum stage without changing observed history; later conversion is rejected. This live-format version is separate from the neuron architecture ID.

Live extension v6 keeps that grouped replay payload and adds a fixed 32-word teaching policy before optional SI arrays. It records bundle/source identities, one or two teacher checkpoint identities, objective settings, optional population scope and contribution counters. The checksum covers all of it. Teacher-assisted resume requires the identical bundle; read-only sampling, evaluation and inheritance use the native checkpoint alone. [Live teacher semantics and word layout](live-teachers.md).

The transition begins at the first added document with the normal document reset pending. A partially observed document's SI path is consolidated at this explicit boundary; an already completed document is not consolidated twice. Weights, moments and replay are retained. Source advancement is a bounded exposure schedule, not an assessment of developmental mastery. [Protocol and commands](live-curriculum.md).

A new stream started with `--checkpoint` inherits parameters and optimizer history, then initializes a new stream and requested memory policy. `sample` and `evaluate` use saved weights with fresh recurrence. Ordinary `train --resume --allow-new-corpus` permits an explicitly declared data change but does not preserve a live stream's cursor/history.

`train --burn-in N --burn-policy warm` supplies detached prefix context before a sampled target window. The `reset` control pays for the same prefix but clears recurrence before targets. Both settings persist through ordinary checkpoint resume. The standard validation evaluator uses reset-state windows.

Neuron-indexed outgoing-weight storage and predicted-route paging are benchmark experiments. The working learner uses resident weights. Correctness and useful latency at larger memory pressure must be shown before routing/paging becomes part of the active runtime.

## Code organization as the project grows

The [project conventions](../AGENTS.md) make modularity part of ongoing implementation. Growing responsibilities should become focused modules with explicit interfaces: neuron kernels, model state, the live learning loop, curricula, checkpoints and population rules. CLI dispatch should delegate to those modules as commands expand. Shared training/inference state and checkpoint semantics remain common across these boundaries.

Structural changes should accompany the feature that needs them and preserve numerical/runtime behavior through the relevant existing checks. This convention records the intended organization; it does not claim that every current implementation has already been split into a separate translation unit.

The fast associative component owns its CUDA recurrence and allocations in `src/associative_memory.cuh`; model orchestration owns its learned parameter layout and state sharing. `src/associative_tests.cuh` adds focused causal and memory controls while reusing the common neuron/live-state suite. `tests/associative_reference.py` shares an independent CPU matrix equation between gradient and scoring oracles, avoiding a second runtime implementation.

`src/associative_bench.cuh` owns isolated layout controls and timings; production model orchestration always selects the default shared forward/backward implementation. Forward matrix cells stay in registers, while the reverse kernel caches current inputs and pads shared matrices. The [runtime experiment](association-runtime.md) separates kernel measurements, complete learned trajectories and archived-binary timings. Python experiment and report modules orchestrate native commands and verify evidence; they do not supply another learner.

The architecture comparison reuses one experiment driver for early and longitudinal schedules. `scripts/native_experiment.py` owns command journaling, manifest verification and shared binding assessment; `scripts/experiment_checkpoint.py` supplies a diagnostic checkpoint view. Endpoint orchestration and reports stay outside runtime kernels, and the native loader remains the authority for checkpoint validity.

Selected teaching-data preparation shares literal fact/query rendering in `scripts/binding_lessons.py`; each preparation script owns its edition's selection, split rules and manifest. Experiment drivers share the read-only checkpoint view in `scripts/experiment_checkpoint.py`, while native code retains authoritative checkpoint validation and all model computation. The [lesson-diversity experiment](lesson-diversity.md) verifies the original prepared bytes after this extraction.

As paired curriculum experiments grow, `scripts/native_experiment.py` owns sequential command journals, source-file authentication and common binding assessments. Drivers retain their own experimental policies and source admission. `tests/binding_learned_oracle.py` provides the shared independent CPU check for their final models. The [ordering comparison](curriculum-order.md) checks complete prior smoke checkpoints after this extraction rather than treating a file move as evidence of preserved behavior.

The narrative continuation reuses those journals and assessments, with selected-book manifest authentication in the shared helper. `scripts/narrative_experiment.py` owns the fixed continuation policy, per-book evaluation and samples; `tests/narrative_experiment.py` independently counts source exposure and checks inherited replay state. It calls the existing append-only preparer and native live command. No additional learner or model forward path is introduced.

Learned-model numerical diagnosis is also separate from the live runtime. `tests/learned_trace_dump.cu` builds as an optional standalone executable through `build.ps1 -TraceDiagnostic`, reusing the existing native forward implementation without replacing the study binary. The CPU reference exposes trace collection and an explicit spike-override hook only for diagnostics; ordinary scoring never uses the override. `tests/learned_threshold.py` localizes the recorded threshold discrepancy, while `tests/binding_cpu_audit.py` reports every final development answer and score disagreement without changing the strict oracle tolerance.

The trace executable now accepts selective and associative checkpoints. `tests/trace_comparison.py` owns the shared trace comparison and single-spike intervention; the two learned-threshold drivers own case selection and provenance. Four earlier selective cases retain every numerical result and all 104 native trace files exactly after this extraction. `tests/associative_history.py` separately tests memory-history/readout reliance using fixed learned weights, with reporting and plots in `scripts/`; it does not modify the production forward path.

`tests/score_diagnosis.py` generalizes post-hoc diagnosis of failed fixed-group checks across studies. It selects the largest checked score discrepancy per failing model, authenticates the checkpoint and executable, repeats the original native group, and calls the shared trace comparison. The narrative study uses this interface without adding another forward implementation or weakening the independent oracle's tolerance.

The existing byte classifier and target-weight validation live in `src/language_objective.cuh`. `src/distillation.cuh` adds an optional objective over frozen teacher logits while using the same `Model::forward` and backward implementation. `src/distillation_tests.cuh` owns synthetic native fixtures; `tests/oracle.py` independently differentiates the combined loss and `tests/distillation_oracle.py` preserves every fixture's result, including a strict first-Adam-step failure. Three ordinary continuation checkpoints and 67 prior numerical fixture files remain identical after extraction. [The teacher-objective stage](teacher-objective.md) supplies the numerical foundation.

Teacher integration extracts the existing persistence responsibility into `src/checkpoint.cuh`. Its shared host reader validates full payloads for ordinary loading and teacher packaging; the GPU loader uploads that validated state. `src/teaching_state.cuh` owns the fixed durable record, `src/teacher_bundle.cuh` owns immutable source/snapshot identity, and `src/teacher_replay.cuh` owns bounded teacher execution and objective application. `src/model_memory.cuh` provides one allocation estimate for evolution and teacher admission. Population lifespan/ownership checks live in `src/population_teachers.cuh`, behind the authority interface used by the live command. The live loop owns when replay occurs; these modules do not introduce another model forward implementation.

`src/model.cuh` now owns `Cache`, `Model`, buffer lifetime, execution views and the existing forward/backward/update methods. Its learning and frozen-forward allocation modes share the same production forward code. Frozen teachers omit learning-only buffers; live generation views keep them because those views also learn short observed document tails. `src/frozen_model_tests.cuh` owns exact allocation/forward/state and mutation-guard comparisons, called by the existing teacher-replay suite. The common allocation counter in `src/test_fixtures.cuh` independently checks `src/model_memory.cuh`. The [completed storage stage](frozen-teacher-memory.md) preserves learned checkpoint behavior while reducing teacher memory.

`src/teacher_replay_tests.cuh` checks causality, recurrence isolation and complete restart state. `tests/teacher_replay_cli.py` and `tests/population_teachers_cli.py` verify separate-process admission, rejection, lineage and lifespan behavior. `tests/teacher_compatibility.py` repeats preserved ordinary continuations and verifies older fixture hashes. Research orchestration remains outside the runtime and no language-quality claim follows from these integration tests.

The retention comparison reuses the existing narrative book assessments and sample generator. Grouped exposure records now live in `scripts/experiment_checkpoint.py`, accepting both ordinary v5 and teaching v6 checkpoints; earlier narrative records retain their original shape and pass the complete prior execution audit. `scripts/teacher_retention_experiment.py` owns paired orchestration and fixed policy, with independent execution audit, summary and figure modules. It calls the same native live command for both branches.

Source formats now have a `scripts/corpus/` package. Its `storybooks.py` reader authenticates selected Markdown pages, publisher text and attribution. `scripts/prepare_early_readers.py` owns selection roles, protected evaluation material, assembly and attribution output, reusing the existing fetch helper. The original Gutenberg preparation path stays intact. The [early-reader audit](early-readers.md) verifies exact output reproduction and rejects source or split contamination; neither module performs model computation.

`scripts/corpus/openstax.py` owns shared CNXML structure, references and tables, with an explicit optional MathML extension callback. `prealgebra_math.py` supplies the additional arithmetic notation used during source review. The default Astronomy subset and its recorded outputs remain unchanged. The [notation audit](prealgebra-notation.md) keeps conversion separate from source admission and rejects unresolved source structures; these modules perform no model computation.

`scripts/review_prealgebra_integrity.py` adds source-content inspection above that converter. It records missing image descriptions and isolates paired worked examples for review, retaining source identity and attribution. Manually inspected observations live in `data/prealgebra-review-cases-v1.json`; neither those observations nor candidate extraction bypass the training-selection registry. [Review evidence and limits](prealgebra-source-integrity.md).

`scripts/corpus/physics.py` owns the optional Physics notation and source-class omissions. `scripts/review_physics.py` authenticates the pinned collection, modules and notices, then produces isolated review text and a structural/media inventory. Source observations and source/extraction tests remain separate from that converter. The [Physics review](physics-source-review.md) creates no curriculum, changes no admitted edition and performs no model computation.

Read-only activity analysis reuses the existing native trace executable. `scripts/activity_metrics.py` owns pure array statistics, `scripts/inspect_native_activity.py` authenticates and inspects one native window, and `scripts/activity_panel.py` owns the fixed multi-model source selection. `scripts/summarize_activity.py` audits saved artifacts and aggregates the completed panel without launching the model. The [activity census](activity-diagnostics.md) adds no forward implementation, learning policy or structural mutation.

Controlled adaptation diagnostics live in `experiments/`. `adaptation_io.cuh` owns source-window admission and authenticated restart metadata; `adaptation_probe.cu` owns disposable-copy initialization, the fixed evaluation policy and calls to the existing production model methods. Source authentication and schedules live in `scripts/adaptation_sources.py`, study orchestration in `scripts/adaptation_experiment.py`, and independent state/exposure auditing in `tests/adaptation_experiment.py`. The [diagnostic contract](adaptation-probe.md) keeps experimental policy out of the live learner and preserves the shared forward/backward/optimizer implementation. CMake's `sg_cuda_target` function supplies the common CUDA linkage and compiler settings to production and diagnostic targets.

Replay interference diagnostics use an optional `LiveSourceObserver` in `src/live.cuh`, around the actual source update and before scheduled replay. `experiments/replay_priority_scoring.cuh` owns independent candidate sampling, reset-state scoring views and journals; `replay_priority_probe.cu` calls the production live loop without changing replay selection. Its views share learned parameters while owning recurrence and strict-FP32 handles, preserving the learner's math setting. Experiment orchestration, independent policy/CPU checks and summary calculations remain separate Python modules. The [measured diagnostic](replay-priority-diagnostic.md) documents exact learned trajectory controls, added cost and a retained strict CPU score failure. No checkpoint format or default learning policy changes.

`experiments/replay_score_graph.cuh` optionally captures the existing reset-state scoring forward and byte classifier. Pinned input/target buffers and loss readback belong to each captured scoring view; live writes complete before a scoring pass, and scoring completes before learning resumes. The [captured scoring comparison](replay-score-graphs.md) verifies exact scores and live trajectories while measuring capture cost inside the loop. It does not change replay selection or add model equations.

Both replay experiments now reuse `experiments/replay_candidate_scoring.cuh` for scoring views and accounting. `replay_two_choice.cuh` owns the optional two-candidate selection policy, and `replay_selection_identity.cuh` owns its resume sidecar. The live observer's default replay callback returns the ordinary choice; explicit overrides are checked against the stored reservoir and uniformly chosen stage. The [selection comparison](replay-selection.md) failed its retention gate, so the policy remains confined to its separate experiment executable.

`experiments/capacity_probe.cu` measures larger randomly initialized models through the production `LiveEngine` and `live_tick`, with optional fixture checkpoints. Its Python driver owns source authentication and matched size comparisons; explicit executable and shape arguments preserve the original defaults. No model equations or production defaults are changed. The original [capacity report](capacity-scaling.md) and [411M result](411m-capacity-results.md) separate measured speed and live buffers from untested language-quality scaling.

`src/live_views.cuh` bounds source-tail and replay scratch caches to four shapes each and owns sharing of the live learner's gradient/decay buffers. The live loop executes these views sequentially, preserving the common weights, optimizer and appropriate recurrent-state ownership. The permanent graph speaker is outside eviction. [Compatibility and measured memory savings](bounded-live-memory.md) include preserved-runtime and restart comparisons.

Policy-only host audits use `experiment_checkpoint.policy_checkpoint` to read ordinary grouped-replay headers and descriptors without copying neural arrays. The independent `tests/stage_replay_reference.py` accepts an optional observation callback while preserving its existing default trajectory. `scripts/prose_replay_coverage.py` binds those reconstructed choices to the actual curriculum document order and native endpoint state. The [coverage audit](prose-replay-coverage.md) changes no production replay policy or model computation.

Retention publications share `scripts/retention_evidence.py` for bounded checkpoint-state reads, native argument reconstruction and verification of raw book/sample artifacts. The restart-control publisher delegates these checks without changing its recorded result values. The [extraction audit](../reports/retention-publication-extraction.json) verifies 51 completed commands and rejects six altered evidence cases; it performs no new model computation.

`scripts/checkpoint_assessment.py` owns the common native book-loss and raw-generation measurements for complete-book experiments. `prose_evaluation.py` preserves the original size-study policy, while `prose_retention_inputs.py` owns scoped parent admission, bounded checkpoint evidence and exposure checks for the learning-rate continuation. `prose_retention_lr.py` owns its paired native sessions, exact resumed-control gate and result comparisons. The [learning-rate protocol](prose-retention-lr.md) reuses the production live command and introduces no model equations or second inference path.

`scripts/corpus/paired_selection.py` owns exact overlap reservations for complete question/solution pairs, including existing authored probe contexts. `scripts/prepare_prealgebra.py` authenticates individually reviewed source exercises and assembles the admitted arithmetic edition through the existing CNXML/math extractor. A conflicting lesson is omitted in full; questions and answers are never partially deduplicated. The [worked-example selection](prealgebra-worked-selection.md) preserves source attribution and whole-chapter evaluation boundaries without changing native learning or inference.

The founder helper's `live_arguments` owns the shared native launch arguments, including explicit replay capacity with an unchanged default. `scripts/prose_replay_capacity.py` owns the fresh capacity controls, initial-array identity checks and paired book results, reusing the founder and checkpoint assessment modules. Its [experiment contract](prose-replay-capacity.md) and [completed results](prose-replay-capacity-results.md) retain the difference between matched update counts and changed replay targets. Generic prose counter inspection uses the bounded policy reader instead of a smaller experiment-specific replay-word limit.

`checkpoint_assessment.verify_saved` reconstructs fixed assessment arguments and validates recorded books and raw generations without model execution. `scripts/publish_replay_capacity.py` adds founder/session, checkpoint, exposure and baseline comparisons above that shared reader. The existing coverage auditor can explicitly select a completed larger-reservoir experiment while retaining its original default and exact baseline rows. These publication and coverage paths do not alter learning, inference or source admission.

The [live view construction change](live-view-construction.md) adds a borrowing constructor to `Model`. `LiveEngine` and `LiveViews` attach the common parameter workspace during construction, preserving shared speaker/tail recurrence and independent replay recurrence. `experiments/view_allocation_probe.cu` owns optional allocation accounting and lifetime/numerical diagnostics; normal runtime builds have no observer. CUDA numerical/ownership, checkpoint/restart and caller compatibility checks pass, as do 411M allocation and the declared short live learning/generation workload.

The [evaluation extension](shared-evaluation-views.md) reuses the same bounded cache for `probes::Scorer` and the borrowing constructor for prefix training and context/decode/cue benchmarks. `tests/view_callers.py` owns the preserved-runtime CLI comparisons and their CPU-only protocol/comparator checks. Its 240 GPU commands and 204 comparisons pass; the change introduces no second forward implementation.

The [saved 411M study](prose-411m.md) reuses the founder launcher and checkpoint assessments with explicit executable and save-interval selection. `large_founder_inputs.py` owns source admission, storage reservations and bounded checkpoint facts; `prose_large_founder.py` owns full-size checkpoint preflight, preserved-baseline replay and sequential full-curriculum assessment. The existing launcher and assessment verifier retain their defaults. New orchestration does not implement model computation or alter the native learning path.
