# Optional membrane objective: isolated integration

This candidate is on `research/membrane-regularization`. It compiles the full
native learner and passes host checkpoint/policy checks. It has **not yet run
CUDA learning**, demonstrated parity with the preserved executable, or improved
language. The 411M study remains on its frozen runtime and original policy.

The [saved-membrane census](membrane-regularization.md) motivated this candidate.
It showed regions where the optional spike activity penalty's local derivative
would be zero. The candidate adds a direct derivative outside that surrogate.

## Shared learning calculation

For pre-reset membrane `u`, coefficient `cost`, and band `b`:

```text
point_cost = 0.5 * max(abs(u) - b, 0)^2
auxiliary_objective = cost * mean(point_cost over batch, time, neurons, layers)
direct_gradient = cost / (B*T*H*L) * sign(u) * max(abs(u) - b, 0)
```

The traced neuron backward kernel adds `direct_gradient` to its membrane
adjoint. The existing detached reset and incoming chunk boundary stay detached.
The derivative then travels through the learned leak and input projection as
part of ordinary backpropagation. It is not multiplied by the spike surrogate.

The same `Model::backward` path serves source chunks, partial chunks and replay.
Teacher assistance and answer weighting affect the language objective as before;
the membrane mean is unweighted across its own neuron positions. SI, if enabled,
observes the resulting combined gradient. No new persistent GPU arrays are added.

Positive cost is restricted to trace, gated, selective and associative cells
(IDs 3–6). LIF and ALIF reject it; ALIF's changing threshold would require its own
objective design. There is no new architecture ID or inference forward path.
The disabled kernel specialization contains no membrane-cost arithmetic.
That source-level property still needs an old/new executable parity test.

The new `live` options are `--membrane-cost` in `[0,100]` and `--membrane-band`
strictly between 1 and 2. Cost defaults to zero, and band to 1.5. No positive
coefficient is recommended yet. A band supplied without a positive cost is
rejected. Reported `loss`/`replay_loss` continue to describe the observed language
objective; they do not silently include the membrane term. Enabled runs also
record their cost and band in metrics and the session report.

## Durable policy

Legacy checkpoint versions 0–6 retain their existing ordering. Enabled live
checkpoints set `meta[17]=7`, add 64 bytes immediately after the original
288-byte header, and carry the underlying live version (1–6) in that extension.

| Extension word | Meaning |
|---|---|
| 0 | `SGPMEM1` magic, encoded in a 64-bit word |
| 1 | Policy schema 1 |
| 2 | Underlying live policy version, 1–6 |
| 3 | Cost as finite positive float32 bits, upper 32 bits zero |
| 4 | Band as float32 bits, upper 32 bits zero |
| 5–7 | Reserved, must be zero |

The extension is validated by the lightweight header reader before learned
arrays are allocated. The full reader adjusts all array/policy offsets, checks
the complete extent, and authenticates the new policy along with the previous
live state. Teacher and SI payloads still depend on the underlying policy.
Curriculum setup, grouped replay conversion and teacher binding update that
underlying version while keeping the membrane configuration.

`--resume` preserves cost and band and rejects attempts to override them.
`--checkpoint` starts a new stream, inherits the objective by default, and
permits an explicit replacement or `--membrane-cost 0`. Disabling an objective
does not remove the effect of earlier learning or change source provenance.

The legacy Python experiment reader explicitly rejects extension 7 instead of
misreading its shifted arrays. Frozen study checkouts remain untouched. The
native reader is the format validation authority for this candidate.

While this experiment is unvalidated, enabled-policy checkpoints are rejected
by population registration, population learning, parent eligibility and teacher
packing/loading. The child-state function also rejects them, so reproduction
cannot silently discard a parent's objective. These are policy-format guards;
they do not establish source/ancestry admission for a checkpoint whose objective
was later disabled. No candidate or descendant is approved for reproduction.

## Verification boundaries

The host target includes the actual native reader and policy code. It loads
76 independently serialized fixtures spanning all legacy versions, wrapped live
versions 1–6, cells 3–6, and optional SI with curriculum and teacher payloads.
It checks exact values and the 64-byte extent change, corrupt/truncated policies,
checksum damage, feature transitions, new-stream inheritance, explicit disable,
teacher binding, and CLI rejection before GPU allocation.

The recorded host results and artifact identities are in
[the integration report](../reports/membrane-policy-integration.json).
These fixtures do not execute GPU checkpoint saving or continued learning.

The separate, explicitly selected `gpu-test` mode is compiled but pending. It
produces eight full-model gradient fixtures (four traced cell types, batch 1/2)
with nonzero recurrent state and membranes beyond surrogate support. The CPU
autograd oracle differentiates the full model independently, including weighted
language targets, combined activity/membrane penalties and nonzero Adam history.
Eight native restart cases cover graph speech, one-byte tails, stage replay,
curriculum changes and optional SI. Running a native fixture alone will not count
as passing its independent oracle.

```powershell
./build.ps1 -SkipTests
./build.ps1 -MembraneCheckpointTest
./build/membrane-checkpoint-test/synaptic-membrane-checkpoint-test.exe host-test --out runs/membrane-host-new
```

Only after the whole 411M driver finishes training **and** its assessments:

```powershell
./build/membrane-checkpoint-test/synaptic-membrane-checkpoint-test.exe gpu-test --out runs/membrane-gpu-new
python tests/oracle.py runs/membrane-gpu-new/cell-6-batch-2
```

All eight gradient fixtures require their oracle, plus old/new executable
comparisons with the objective disabled. After those checks, select coefficients
in declared, matched learning trials using held-out language loss, retention and
raw generations. A lower spike rate alone will not select this candidate.
