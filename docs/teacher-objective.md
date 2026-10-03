# Numerical foundation for parent feedback

The [narrative continuation](narrative-learning.md) learns new book statistics while losing an earlier binding skill. The next candidate is to rehearse selected source examples with both their actual targets and an earlier model's predictions. That could support retention and later teaching by two parents, but it needs a controlled learning comparison.

This document records the **C++/CUDA objective and its numerical foundation**. The subsequent [live-teacher integration](live-teachers.md) attaches it to selected replay windows with durable policy, memory limits and registered teacher lifespans. Tests exercise that runtime, but no retention, speed or generational improvement is yet claimed.

## Research basis and specific choice

[Hinton, Vinyals and Dean](https://arxiv.org/abs/1503.02531) describe learning from softened model predictions alongside actual labels, with a squared-temperature factor to balance gradient scale. [Dark Experience Replay](https://papers.nips.cc/paper_files/paper/2020/file/b704ea2c39778f07c617f6b7ce480e9e-Paper.pdf) studies rehearsal of historical predictions, including an additional actual-label objective. Its stored trajectory logits and squared-logit loss differ from the frozen-teacher probability objective proposed here. Their results motivate an experiment; they do not establish that this spiking byte model will retain skills better.

For input position `i`, both teachers receive the same selected source prefix as the learner. Their 256-byte prediction distributions are combined:

```text
q_i = mixture * softmax(teacher_A_logits_i / temperature)
    + (1-mixture) * softmax(teacher_B_logits_i / temperature)

p_i = softmax(learner_logits_i / temperature)

loss = sum_i weight_i * (
           CE(actual_next_byte_i, learner_logits_i)
         + strength * temperature^2 * KL(q_i || p_i)
       ) / sum_i weight_i
```

One teacher is also supported. The actual-byte cross-entropy keeps temperature one. Teacher predictions are constants for differentiation; all parameter gradients belong to the learner. The mixture operates on output distributions, so teacher and learner neuron counts may differ while sharing the fixed byte vocabulary. Actual targets keep their own coefficient of one: increasing teacher strength also changes the combined gradient scale and must be controlled in learning experiments.

Teacher agreement can preserve shared mistakes. Future comparisons need actual-target-only controls and independently defined development answers. A frozen earlier self is a retention control; using two distinct parents is a separate teaching comparison.

## Module boundary and memory

`src/language_objective.cuh` owns the existing byte classifier, target-weighting kernel and shared weight validation. The classifier's computation is unchanged. `src/distillation.cuh` owns teacher probability preparation and the optional KL gradient. It accepts logits from the existing `Model::forward`, and its result separates observed-source loss, teacher penalty and combined loss. It introduces no second neuron forward implementation.

The caller runs an observed-target forward, prepares teacher targets for the same byte positions, applies the objective once, then calls the existing backward/update. Optional answer weights apply to both terms. Temperature is fixed within a target object and shared by teacher/learner softmax; supported bounds are 0.25–16, with strength 0–100 and mixture 0–1. These are engineering bounds, not biological ages or recommended learning settings.

The target object owns `4 * N * (256 + 1)` GPU bytes for `N` positions: probabilities and per-position validation/loss scratch. At 128 positions this is **131,584 bytes (128.5 KiB)**, excluding teachers, learner, CUDA overhead and host copies. Preparation checks finite teacher logits; invalid settings or unprepared targets reject before modifying learner gradients. Numerical failure during objective execution aborts the update.

Ordinary live learning does not construct the optional object. Its model buffers, checkpoint format and recurrence sharing remain unchanged. The subsequent integration uses live format 6 for teacher-assisted learners and budgets complete teacher working sets plus target buffers. It records identities and source eligibility, enforces registered teacher lifespans and preserves teacher policy on restart. Those guarantees are checked separately from the numerical fixtures described below.

## Verification and retained failure

The [validation report](../reports/distillation-validation.json) records 17 passing native CTest suites and seven independent objective fixtures covering all six cell types. CUDA Compute Sanitizer race checking reports zero hazards, errors or warnings for the native objective suite. Fixtures use two frozen models with different dimensions and include nonzero learner recurrence, ordinary/answer-weighted targets, temperatures 0.5/2/4, and an explicitly separate optimizer-history case. Native controls also check exact zero-strength behavior with both weighting modes, teacher weight/moment/state isolation, single/self teachers, mixture endpoints, zero probabilities, large logits and invalid input rejection.

The independent CPU implementation checks the combined loss, every parameter gradient, activity-regularized gradients and Adam updates. Maximum discrepancies across the fixtures are:

| Quantity | Maximum absolute difference |
| --- | ---: |
| Teacher probabilities | 3.73e-9 |
| Teacher penalty | 1.02e-6 |
| Logit gradients | 1.49e-8 |
| Parameter gradients | 8.94e-8 |
| Activity-regularized gradients | 1.12e-7 |

**One strict end-to-end first-Adam-step check fails and remains a failure.** In the associative fixture, a CPU gradient of 1.5881653e-8 differs from CUDA's 1.5279511e-8 by 6.02e-10 at one parameter. The first update with zero moments magnifies this into a maximum weight difference of 9.21e-6, exceeding the unchanged 5e-6 tolerance. Applying Adam to the actual CUDA gradient instead agrees within 5.96e-8, localizing the amplification to the differing input gradients. This optimizer diagnostic does not turn the independent check into a pass.

A separate fixture starts that same associative objective with declared nonzero moments at step seven. It passes with a maximum update difference of 2.98e-8. The original zero-moment case is retained alongside it. All parameter-gradient and regularized-gradient checks pass, including the failing update case. The shared oracle now saves the complete result before reporting an Adam tolerance failure; `tests/distillation_oracle.py` runs every case, retains each result and exits unsuccessfully while any strict failure remains.

The [compatibility report](../reports/distillation-compatibility.json) verifies **67 previous numerical fixture files byte-for-byte**, all seven existing hard-target CPU oracles, and complete continuation checkpoints for selective H512, selective H588 and associative H512 after the objective-module extraction. Those continuations replay the existing 128-to-384-observation narrative rehearsal. They demonstrate preservation of the ordinary path under that checked workload; they do not establish numerical equivalence on every possible input.

```powershell
.\build.ps1
python tests/distillation_oracle.py
```

The second command intentionally returns failure for the recorded first-step tolerance case. Lowering a tolerance or silently dropping that fixture would hide the limitation. All fixtures are synthetic verification models; none supplies weights or data to a population member.

## Integrated runtime and remaining comparison

The [live implementation](live-teachers.md) records a fixed bundle, its selected source prefix, frozen model identities, optional registered member origins and assistance counters. Teacher parameters stay frozen, while the learner's weights and live state remain shared between learning and generation. Bundle replacement and automatic teacher selection remain outside the implemented policy.

A subsequent declared experiment should branch every eligible parent into actual-target-only and teacher-assisted replay with matched source exposure and replay descriptors. It should report new-book loss, retained binding, actual samples, complete live cost and teacher memory use. The policy, coefficients and endpoints must be fixed before that comparison, and its results must include regressions. Automatic developmental promotion and reproduction based on these skills require their own acceptance rules.
