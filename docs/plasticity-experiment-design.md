# Measuring development before changing the learner

The next diagnostic should measure how experience changes learning on the same new material. Our completed [retention study](teacher-retention.md) demonstrates forgetting, while the [activity census](activity-diagnostics.md) finds no never-firing neurons in its sampled windows. Neither result establishes that the models have lost their ability to learn.

This note extends the [initial research review](research-live-plasticity-2026-10.md) with a concrete pilot design. The [source and proposed-protocol record](../reports/plasticity-experiment-design.json) pins the current evidence, checkpoint candidates and selected reading edition. **The adaptation pilot has not run.** Its native diagnostic driver and execution checks still need implementation before any learning result can be claimed.

## What the additional research changes

Lyle and colleagues distinguish learning flexibility from representational capacity. Their 2023 study measures adaptation to new probe objectives, reports plasticity loss without saturated units, and examines both optimizer instability and loss geometry. Resetting optimizer state alone does not improve plasticity in their reported appendix comparison. That makes a reset a useful diagnostic control, not an assumed cure. The reviewed material includes the primary definition, probe methodology, motivating examples and Appendix B.1. [ICML 2023 paper](https://proceedings.mlr.press/v202/lyle23b/lyle23b.pdf)

Their later work separates several mechanisms, including shifts in preactivation distributions, growth in parameter magnitude and changes in unit behavior. It finds benefits from combining interventions in its settings; one scalar such as neuron activity is insufficient. Its definition focuses on optimization, and explicitly permits including optimizer state. The publisher abstract and the 2024 preprint's definitions, mechanisms and selected experiment details were reviewed; the publisher PDF was unavailable to the browser tool. [Published record](https://proceedings.mlr.press/v274/lyle25a.html), [preprint v1](https://arxiv.org/html/2402.18762v1)

Hernandez-Garcia and colleagues compare renewal of individual weights with renewal of whole units. In their supervised experiments, weight renewal is more effective in some small or layer-normalized networks. One utility estimate is `abs(weight * gradient)`; it estimates the first-order effect of zeroing that weight, not the full effect of resampling it. The study also distinguishes fitting ability from generalization. Main definitions, method, comparisons and optimizer/tuning appendices were reviewed in preprint v2; external code was not run. [Published record](https://proceedings.mlr.press/v330/hernandez-garcia26a.html), [preprint v2](https://arxiv.org/html/2508.00212v2)

Prakash and colleagues connect reduced trainability to curvature in a linearized ReLU analysis and report benefits from feature-rank and L2 regularization. Only the publisher abstract was reviewed, so its detailed assumptions and implementation remain unverified here. Our centered emission participation ratio is not their Hessian spectrum, and the inference does not transfer automatically to surrogate gradients and discrete spike decisions. [ICML 2026 record](https://proceedings.mlr.press/v306/prakash26b.html)

These studies add connection-level renewal to the candidate list. They do not establish which intervention will improve SynapticGenesis. Our model uses RMS normalization, spiking recurrence and continuous traces; the cited feed-forward and reinforcement-learning experiments differ materially.

## Pilot: equal new reading, different prior experience

Start with the existing associative H512/C256/L4 architecture and all three established seeds: 1337, 2026 and 31415. This is a bounded study of the architecture with the strongest measured binding retention, not a claim that other cells behave the same way.

For each seed, compare a fresh initialization with three experienced states: the 130,000-observation parent, the 190,000-observation ordinary-replay descendant and its frozen-self-teaching counterpart. Each experienced state gets two diagnostic copies: one carries its Adam moments and update counter, while the other resets those moments and the counter but preserves identical weights. That makes **seven starts per seed**.

Use both fixed learning rates, `0.000075` and `0.0003`, without choosing the better rate afterward. Use batch 1, context 128, strict FP32, core-rate multiplier 1, Adam betas 0.9/0.95, epsilon `1e-8`, weight decay 0.01 and norm clip 1. There are **42 trajectories** of at most 4,096 updates. Endpoints are 0, 64, 256, 1,024 and 4,096 updates. Every trajectory receives 524,288 new-source target-pair presentations; these are repeated presentations, not that many unique bytes.

Use the already prepared [early-reader edition](early-readers.md): ten selected training stories, three development stories and three reserved test stories. Its ten training stories contain only 14,201 bytes including separators. This is a small adaptation pilot, not a sufficient general-language corpus. Its publisher levels are not biological ages. The recorded prior studies have not trained on this edition; shared vocabulary and broader subject matter are expected.

Before execution, materialize a deterministic 4,096-window schedule per seed. Choose a training document uniformly and a valid start position uniformly within that document, retaining 129 bytes for 128 next-byte targets. All starts and both learning rates for that seed use the identical schedule. Record source document, byte offset, window hash and complete schedule hash. Never cross a document separator or include development/test text in training. If an admitted training document is shorter than 129 bytes, reject the prepared protocol rather than silently dropping it.

Reset recurrent state at each sampled training window. Use unweighted byte loss. Teacher feedback, earlier-source replay and consolidation penalties are absent in these temporary diagnostic copies; this isolates adaptation under a common policy. It does not reproduce the full live system's retention behavior. Evaluation also uses explicitly recorded reset windows and never updates parameters. The production model and kernels remain the shared computation path.

## Read the outcomes separately

| Question | Measurement |
| --- | --- |
| Can the model fit the admitted new material? | Training-source loss before learning and at every endpoint; full curve and loss reduction |
| Does new learning generalize? | Development-story loss per byte, both pooled and per story, at the same endpoints |
| Does prior experience help or hinder? | Each experienced trajectory versus the matching fresh trajectory, separately for both rates; show starting and final losses |
| Does optimizer history explain a difference? | Carried-state versus reset-state copy of exactly the same experienced weights |
| What earlier skill is lost? | The established binding development assessment and earlier book assessments before and after adaptation |
| What does adaptation cost? | Presented pairs, update counts, resident bytes and elapsed update/evaluation time with shared-GPU limits |

Report normalized trapezoidal area under each fixed-checkpoint loss curve alongside the raw points. A smaller loss reduction can occur because a model started much better; it is insufficient evidence of lost plasticity. A low training loss with worsening development loss indicates a different problem from failing to fit the training examples. Report every seed, rate and source result, and treat related checkpoints as related observations. Three seeds support a pilot, not a broad statistical guarantee.

This first pilot has no default-algorithm promotion gate. Worse adaptation than fresh at a fixed budget is evidence about this task and optimizer setting, not proof of irreversible ageing. Stronger claims require repeated tasks from a comparable task family, additional seeds and a matched live continuation with replay. The reserved tests stay unused during this diagnostic design and tuning.

## Implementation boundary and subsequent choices

The native checkpoint reader must authenticate the original files. Diagnostic copies must leave them unchanged. A state-carrying copy needs the stored global Adam update count as well as both moments; a reset copy needs all three reset together. Recurrent states and histories must not leak between probe windows. The driver must verify exact source exposure, initial-weight equality between carry/reset copies, finite outcomes and uninterrupted/resumed probe equivalence. Existing strict independent numerical failures remain visible and must not be relabeled as passes.

If the pilot reveals poor adaptation, investigate actual gradient and parameter-step geometry before choosing a remedy; the [AdamW projection counterexample](gradient-conflict-research.md) explains why gradient agreement alone is insufficient. Connection renewal and unit renewal should then be compared with random replacement and no replacement at a matched parameter budget. Any later live policy needs explicit handling of affected Adam, recurrent, consolidation and lineage state, with retention measured separately.

Population lifetime, amount of experience, curriculum progress and the history of an individual connection should remain separate variables. A model's age can schedule a test; measured learning and retention should determine whether a developmental or reproductive change is beneficial. This is an engineering recommendation for the project, not a validated biological model of childhood or ageing.
