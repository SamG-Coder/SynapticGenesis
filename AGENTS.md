# Project conventions

## Code organization as the project grows

Keep implementation easy to extend as features accumulate. When a responsibility grows or a change would duplicate logic, extract a focused module with a clear interface as part of that stage.

- Keep neuron equations and CUDA kernels, model configuration/state, the live learning loop, curriculum/source handling, checkpoint persistence, population/evolution rules, and CLI dispatch as distinct responsibilities.
- Keep the CLI entry point thin as commands grow. Put feature behavior beside the module that owns it, and keep numerical tests and experiment orchestration separate from runtime code.
- Training and inference share one model and state contract. Module boundaries must preserve that shared design and avoid separate implementations of the same forward computation.
- Prefer simple structures and explicit ownership. Add directories and interfaces when existing responsibilities justify them; a folder hierarchy alone does not create useful modularity.
- For structural changes, preserve checkpoint compatibility and measured behavior, run the checks relevant to the moved code, and document the resulting boundaries in `docs/architecture.md`.

Model computation remains native C++/CUDA. Python may prepare selected data, orchestrate experiments and provide independent test references.

## Training sources

Use `data/training-selection.json` for current book editions and continuation bases. Retired editions and models that learned from them are historical artifacts, not inputs to new learning, teaching or reproduction. Review new passages before admitting an edition. A source or checkpoint rename does not reset its provenance. Do not rewrite historical source manifests or claim that removing text from a list removes its influence from learned weights.

## Delivery

Commit and push each completed, verified stage. Keep research results and their limitations together, including regressions.

## Naming

Preserve names supplied by the user. Do not add alternative place/project names or naming subtitles unless explicitly requested.
