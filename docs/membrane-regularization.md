# Direct membrane regularization: evidence before a learning trial

The existing optional spike-activity objective cannot directly correct a
neuron whose membrane remains outside the spike surrogate's support for an
entire backward chunk. The preserved native traces show that this condition
is much more common in the earlier large founders after learning. This gives
a specific reason to test a membrane objective, while the running 411M study
continues unchanged.

The original census stage implements and checks a candidate scalar objective in C++/CUDA,
and measures its potential reach using saved arrays. It does **not** integrate
the candidate into the learner or demonstrate improved language. All nine
audited checkpoints have `activity_cost=0`: the existing activity penalty was
disabled in these founders. The analysis describes what that optional penalty
could directly influence if enabled at their recorded states.

The subsequent [isolated policy integration](membrane-policy.md) adds an
optional shared-backward path and checkpoint extension on the research branch.
Its compile/host checks do not establish CUDA learning or language benefit.
The running 411M study still uses the preserved earlier runtime.

## What the existing backward computation allows

The signed trace cell uses thresholds at -1 and +1. Its local spike surrogate
is zero at `abs(u) >= 2`, where `u` is the pre-reset membrane. The activity term
enters [the trace backward kernel](../src/trace_neuron.cuh) as
`activity_scale * spike`, before multiplication by that surrogate. Its own
contribution to the membrane gradient therefore follows

```text
local_activity[t] = activity_scale * spike[t] * surrogate(u[t])
activity_membrane_gradient[t] = local_activity[t] + beta * activity_membrane_gradient[t+1]
```

The backward chunk starts with zero future carry. If all its local activity
terms are zero, this particular contribution remains zero throughout that
neuron's chunk. Increasing its coefficient cannot change a product with zero.
This follows from the implemented equations; it is not a diagnosis that the
whole neuron or model has stopped learning. Task loss, other layers, other
timesteps, gate paths, optimizer history and weight decay can still change
parameters. A quiet neuron also has zero activity cost, so the reported
condition additionally requires firing at **every** position in the chunk.

## Measurement on preserved native traces

The [declared audit inputs](../data/membrane-penalty-audit-v1.json) select all
nine cases of the existing [matched trace panel](prose-spike-results.md): three
sizes at initialization, stage one and stage four. Each case contains eight
independent 1,024-byte source windows, with fresh recurrence at the beginning
of each window. The analysis partitions them into 128-byte intervals, matching
the usual backward length without rerunning a neural forward pass.

The table reports the percentage of **neuron/chunk intervals** that fire on all
128 bytes while having no local activity-penalty signal. It is not a percentage
of permanently inactive neurons or the actual live training chunks.

| Founder | Initialization | Stage one | Stage four |
| --- | ---: | ---: | ---: |
| 2M | 0.0168% | 0.0656% | 0.7782% |
| 27M | 0.0257% | 2.1680% | 4.3448% |
| 105M | 0.0200% | 19.7961% | 27.7881% |

For the final 105M model, this is **582,758 of 2,097,152 neuron/chunk intervals**.
The per-layer fractions range from 1.30% to 55.45%. The aggregate is weighted
by the counted intervals; all layers within a given case have equal sizes.

The [complete result](../reports/activity-penalty-reach.json) records all
windows and layers, the 960 authenticated membrane/spike files, checkpoint
identities and saved activity-cost settings. Every recomputed firing and
zero-surrogate fraction matches the earlier published trace result. The
source checkpoints, raw arrays and current learner are unchanged.

These earlier founders use their recorded learning policy, including the
smaller replay reservoir. This is not a measurement of the current 411M run,
the newer replay-capacity comparison, or all possible text contexts.

## A candidate that supplies a direct correction

The isolated helper in
[`experiments/membrane_penalty.cuh`](../experiments/membrane_penalty.cuh)
defines the following point cost, with candidate band `b=1.5`:

```text
excess = max(abs(u) - b, 0)
cost = 0.5 * excess^2
direct_membrane_derivative = sign(u) * excess
```

For example, at `u=50`, the existing local activity contribution is zero,
while the candidate derivative is 48.5 before its objective weight and
normalization. At `u=-50`, that derivative is -48.5. It vanishes inside the
band. The candidate reaches every fully firing/no-activity-signal interval
counted above, by construction; that property is not a language-quality test.

The band lies inside the outer zero-surrogate boundary, so a correction need
not stop outside the available surrogate region. It is an engineering candidate,
not a biological target or a selected optimal value. A future learner would
add its derivative directly to the membrane adjoint, before propagation through
the existing detached-reset recurrence, rather than multiplying it by the spike
surrogate. Its caller would normalize across batch, time, neurons and layers,
then apply a declared strength.

The unweighted mean point cost in the final snapshots is approximately 49.35
for 2M, 432.52 for 27M and 5,126.22 for 105M. Thus a seemingly small coefficient
can have a substantial effect, especially when applied to an already learned
model. These values alone do not determine gradient norms after recurrence,
clipping or AdamW, and do not select a coefficient.

The candidate does not require a different inference forward computation.
It would add an objective when learning from an admitted observation; generated
speech would still supply no verified training target. No regularizer is
currently applied to any model, and no existing checkpoint is rewritten.

## Research connection and boundaries

[RecDis-SNN (CVPR 2022)](https://openaccess.thecvf.com/content/CVPR2022/html/Guo_RecDis-SNN_Rectifying_Membrane_Potential_Distribution_for_Directly_Training_Spiking_Neural_CVPR_2022_paper.html)
proposes a loss acting on membrane distributions. Only its publisher abstract
was accessible here, so its detailed objective was not reproduced.
[Wang, Cheng and Lim (2023)](https://arxiv.org/html/2304.13289) describe a
symmetric distribution-divergence objective and a separate parametric surrogate
method for classification. Our signed cell and point penalty differ from that
method; its results do not establish language or retention gains here.

[Membrane Potential Batch Normalization (ICCV 2023)](https://arxiv.org/html/2308.08359)
instead normalizes the membrane and folds trained statistics into inference
thresholds. Its method motivates checking membrane scale, but adapting such
statistics to one continuously learning causal stream would need an explicit
update contract. The direct penalty is a simpler candidate to evaluate first.
The [research ledger](../reports/membrane-regularization-research.json) records
which sections were read and failed access attempts. No paper data, external
implementation or pretrained weights enter training.

## Verification and the next learning gate

The optional CUDA target compiles the same point helper for host and device.
Only the host reference is executed while the 411M learner occupies the GPU.
It passes 11 point checks, 28 input-drive finite differences and four leak-coefficient
finite differences through short recurrences. Spike resets are held fixed
during perturbation to test the declared detached-reset derivative. Maximum
drive and leak errors are approximately `2.30e-10` and `2.07e-8`.
[Host result](../reports/membrane-penalty-host.json).

The [array-statistics checks](../reports/activity-penalty-reach-checks.json)
compare 28 synthetic window/chunk combinations against scalar loops and reject
eight malformed inputs. They distinguish non-firing activity-cost zeros from
zero spike surrogates, both signs, and the effect of the backward boundary.
All eight build selectors and 28 conflicting-selector pairs pass. This is
compile and CPU evidence; device numerical agreement is still untested.
The [integration record](../reports/membrane-penalty-integration.json)
authenticates all 589 preserved study references and confirms that the 42
production source files, existing binaries and source-admission registry
are unchanged.

Before any language trial, integrate the term through the shared backward
path in an isolated candidate, verify its disabled setting against the
unchanged runtime, and check CUDA gradients plus interrupted/resumed learning.
Declare any new checkpoint policy explicitly. Then compare fresh unchanged,
spike-penalty and direct-membrane-penalty learners with matched source, replay,
speech and optimizer budgets across multiple seeds. New-text loss, earlier
retention and raw generations decide whether it helps; reduced firing or
finite-gradient correctness cannot promote it by themselves. The independent
411M capacity comparison retains its original settings.

```powershell
./build.ps1 -MembranePenaltyReference
./build/membrane-penalty-reference/synaptic-membrane-penalty-reference.exe --out runs/membrane-host-new.json
python tests/activity_penalty_reach.py --out runs/membrane-counts-new.json
python scripts/activity_penalty_reach.py --out runs/membrane-reach-new.json
```

The final command uses the preserved local trace files. It does not launch
CUDA inference or create another model.
