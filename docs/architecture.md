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

## Shared learning and inference

The live engine has execution views with different chunk sizes over the same weights, Adam moments and neuron state. Observed chunks update parameters; generation then sees those updates directly. CUDA graph decoding captures generation without changing the initial neuron state, and reads the current shared parameter allocation without recapture.

Training and generation are ordered. CUDA streams hand off at completed boundaries; simultaneous unsynchronized read/write access is not used. This is one interleaved learning-and-generation runtime. Gradients stop at chunk boundaries even though recurrent state persists.

The source cursor advances across every within-document next-byte pair, including short tails. The recurrent state resets at document boundaries. Prompt and generated bytes enter the same recurrent stream but never become training targets. Generation can therefore affect the following observed chunk's context. This is a declared experimental protocol, not an optimal learning schedule.

Replay stores document/offset/length descriptors for previously observed source windows. A default 1,024-entry reservoir uses 24 KiB of descriptors plus a policy header; source bytes are separately resident in CPU memory. Replay shares weights and optimizer history but starts its own recurrent state from zero. Its updates and byte exposure are counted separately. `--replay recent` provides a newest-window control; `--replay none` disables replay.

With a curriculum, optional `--replay stage` divides a fixed slot budget among introduced source stages, maintains a separate lifetime reservoir per group, and samples nonempty groups equally. Group ownership follows when the document was introduced, including later observations of an earlier source. [Policy, conversion limits and experimental protocol](stage-replay.md). The host-side `StageReplayView` validates the saved layout and `StageReplay` owns its mutations, while `ReplayMemory` retains shared scheduling. Both policies use the same native model update and inference code.

`--core-scale` scales the learning rate for embeddings and spiking blocks; the output head keeps the full rate. It does not eliminate backpropagation or optimizer bookkeeping.

An explicit version-3 curriculum may emphasize answer targets in selected lesson documents. Both live observations and replay use the same per-window normalized weighted loss, while inference and held-out evaluation stay unchanged. The policy is bound by the curriculum identity and restored with its document annotations. See [answer-emphasis semantics](live-curriculum.md). Independent CPU autograd checks the weighted gradients for all five cell types, including incoming recurrent state and activity regularization.

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

Checkpoints include a checked architecture version, metadata, parameters, Adam moments and a payload checksum. Architecture versions 1–5 represent LIF, ALIF, filtered-spike LIF, gated trace readout and selective trace retention. ALIF/trace and read-gate/retention-gate pairs use equal-sized layouts with different semantics; checked versions prevent mixing them. The architecture version and live-extension version occupy separate header fields. Live extensions add recurrence/cursor/RNG state, then replay, then consolidation history. Corrupt lengths, mismatched dimensions, unsupported policies and invalid history are rejected. The diagnostic reader is not a replacement for the native loader's complete validation.

`live --resume` restores the complete saved stream and requires the same source corpus and prompt. With `--curriculum`, live extension v4 permits declared append-only source transitions and binds all scheduled editions to the checkpoint. It preserves replay descriptors, RNGs and optional SI history. Its policy header uses words 14 and 15 for the combined schedule/source hash and zero-based stage index; hyperparameter slot 7 stores the base learning rate. Optional SI arrays follow the replay payload only when policy word 9 equals 1. The checksum covers all policy fields, state and arrays. Earlier checkpoint versions remain readable.

Live extension v5 retains that common header and adds grouped replay: one group-count word, five words per source group (exclusive document end, observed windows, stored windows, replay updates and replay target pairs), then grouped three-word descriptors. It requires replay mode 3. The bounded loader verifies group ranges and counters before accepting it, including for read-only inference. Ordinary reservoirs can convert during the first curriculum stage without changing observed history; later conversion is rejected. This live-format version is separate from the neuron architecture ID.

The transition begins at the first added document with the normal document reset pending. A partially observed document's SI path is consolidated at this explicit boundary; an already completed document is not consolidated twice. Weights, moments and replay are retained. Source advancement is a bounded exposure schedule, not an assessment of developmental mastery. [Protocol and commands](live-curriculum.md).

A new stream started with `--checkpoint` inherits parameters and optimizer history, then initializes a new stream and requested memory policy. `sample` and `evaluate` use saved weights with fresh recurrence. Ordinary `train --resume --allow-new-corpus` permits an explicitly declared data change but does not preserve a live stream's cursor/history.

`train --burn-in N --burn-policy warm` supplies detached prefix context before a sampled target window. The `reset` control pays for the same prefix but clears recurrence before targets. Both settings persist through ordinary checkpoint resume. The standard validation evaluator uses reset-state windows.

Neuron-indexed outgoing-weight storage and predicted-route paging are benchmark experiments. The working learner uses resident weights. Correctness and useful latency at larger memory pressure must be shown before routing/paging becomes part of the active runtime.

## Code organization as the project grows

The [project conventions](../AGENTS.md) make modularity part of ongoing implementation. Growing responsibilities should become focused modules with explicit interfaces: neuron kernels, model state, the live learning loop, curricula, checkpoints and population rules. CLI dispatch should delegate to those modules as commands expand. Shared training/inference state and checkpoint semantics remain common across these boundaries.

Structural changes should accompany the feature that needs them and preserve numerical/runtime behavior through the relevant existing checks. This convention records the intended organization; it does not claim that every current implementation has already been split into a separate translation unit.

Selected teaching-data preparation shares literal fact/query rendering in `scripts/binding_lessons.py`; each preparation script owns its edition's selection, split rules and manifest. Experiment drivers share the read-only checkpoint view in `scripts/experiment_checkpoint.py`, while native code retains authoritative checkpoint validation and all model computation. The [lesson-diversity experiment](lesson-diversity.md) verifies the original prepared bytes after this extraction.

As paired curriculum experiments grow, `scripts/native_experiment.py` owns sequential command journals, source-file authentication and common binding assessments. Drivers retain their own experimental policies and source admission. `tests/binding_learned_oracle.py` provides the shared independent CPU check for their final models. The [ordering comparison](curriculum-order.md) checks complete prior smoke checkpoints after this extraction rather than treating a file move as evidence of preserved behavior.
