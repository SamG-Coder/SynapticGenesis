# Research: replay conflicts and the actual optimizer step

This follow-up to the [plasticity review](research-live-plasticity-2026-10.md) examines a possible retention intervention. No gradient-projection algorithm is implemented by this note.

[Gradient Episodic Memory, Lopez-Paz and Ranzato (2017)](https://papers.nips.cc/paper/7225-gradient-episodic-memory-for-continual-learning.pdf) constrains updates using gradients on stored examples from earlier tasks. Its evaluation separates new learning, forgetting and transfer. This is relevant to our measured retention problem, but the paper's classification results do not establish a benefit for a spiking byte model.

[A-GEM, Chaudhry et al. (2019 preprint version)](https://arxiv.org/html/1812.00420v2) replaces separate earlier-task constraints with one reference gradient from sampled episodic memory. A conflicting gradient is projected onto the reference constraint. Its cheaper sampled-average protection can leave individual tasks worse. The paper also measures early learning curves and separates hyperparameter-selection tasks from final evaluation tasks. Main methods, update derivation and the average-versus-worst-case discussion were reviewed; the external implementation was not run.

## Why our AdamW update needs separate treatment

For a smooth earlier loss with gradient `r`, a plain step `delta = -eta * g` predicts a first-order change `r dot delta`. Requiring `r dot g >= 0` controls that estimate for this particular step. Adam's moments and per-coordinate scaling change the direction; weight decay and different core/head rates can change it further. Our [optimizer kernel](../src/spike_lm.cu) uses all of these mechanisms.

The [computed two-coordinate example](../reports/gradient-projection-geometry.json) makes the issue concrete. Let `r = [0.2, 1]` and the new gradient be `[2, -1]`. Projecting gives approximately `[2.11538, -0.42308]`, whose dot product with `r` is zero. Applying the current global norm clip preserves that zero. With zero initial Adam moments, the coordinate normalization then produces an approximately `[-0.001, +0.001]` parameter step at learning rate `0.001`. The old linear loss increases by approximately **0.0008**. Plain gradient descent on the projected vector has zero first-order change instead.

This is an illustrative arithmetic counterexample, not a measured failure of the current learner or the cited implementation. The calculation uses the native optimizer's beta values and epsilon with decay disabled and equal parameter learning rates; extra complications are unnecessary to demonstrate the problem. It does not alter the separate, already reported strict first-Adam-step numerical test failure.

```powershell
python scripts/gradient_projection_geometry.py --out reports/gradient-projection-geometry.json
```

For our surrogate-trained spiking model, even a favorable first-order estimate is insufficient: a finite step can change discrete spike decisions. A future experiment must record the proposed and actual parameter deltas, and measure actual replay loss before and after an update on the same fixed source windows and recurrent-state conditions.

## Candidate diagnostic before an intervention

At unchanged checkpoint weights, compute gradients for a selected new-source window and selected replay windows. Record their norms and pairwise products, then compare those products with the actual AdamW displacement and old/new losses. Keep per-source results because an average can hide a harmed skill. Use training replay examples for any update decision; independent evaluation data measures the outcome.

A subsequent projection experiment needs explicit optimizer-state semantics, a zero-reference-gradient case, source/teacher attribution, matched source exposure, replay cost, and restart checks. It should compare against ordinary replay and teacher-assisted replay without claiming that an algebraic projection guarantees lifelong retention. This complements the [activity diagnostics](activity-diagnostics.md); it addresses interference, while fresh-task learning curves are still needed to assess plasticity.
