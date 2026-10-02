# General developing models: curriculum age and generations

SynapticGenesis targets a general small spiking model, native C++/CUDA first, that develops through increasingly difficult content and produces new generations from two existing models plus new material. The eventual population is a small town of models with different developmental stages.

## Two independent forms of progress

- **Developmental age:** the model's curriculum stage, accompanied by actual learned skills, exposure counts and evaluation evidence. The user's one-year/two-year labels describe these stages; they are not validated human cognitive ages.
- **Generation:** the model's lineage. Two selected models can contribute to a new child, whose parents and birth checkpoint are recorded. Advancing an existing model's age does not create a new generation.
- **Simulation age:** elapsed population ticks since birth. This controls lifespan and death independently of curriculum mastery or generation.

A child's own exposure age starts at zero. Inherited capability must be tested, so a child need not repeat material it already demonstrates mastery of. Neither elapsed runtime nor its parents' ages establish its skills. Founder models retain the original from-scratch requirement. The design uses both inheritance and teaching: two parents contribute model DNA and learned structures, while teachers supply selected source material and eventually separately recorded feedback. Children can mutate inherited learning settings, and sometimes gain neurons within a declared size budget. Only independently evaluated, sufficiently fit members are eligible to reproduce.

## Proposed age curriculum

These are initial engineering stage definitions, not claims about exact human developmental milestones.

| Stage label | Content progression | What an evaluation must distinguish |
| --- | --- | --- |
| Age 1 | Familiar entities, actions, properties, simple associations and short exchanges | Recall of specific examples versus correctly handling new combinations |
| Age 2 | Short sentences, relations, simple requests and questions | Correct relations and following one instruction versus fluent repetition |
| Age 3 | Short stories, event order, basic quantities and simple causes | Remembering relevant events and explaining a supported relation |
| Later stages | Broader reading, longer context, multiple constraints and new subject areas | New learning, retention of earlier skills and generalization to unseen material |

Each stage needs a versioned source manifest, separate training/development/test partitions and concrete skill probes before it can promote a model. Content becomes harder gradually; earlier material remains in a bounded replay mixture. A stage transition should change content distribution and, when experiments support it, learning rate/plasticity. It should not automatically freeze all older connections. Current tests found that strong weight protection hurt both old and new language performance.

Candidate general material should be selected and reviewed at the document or lesson level. Only selected sources enter training. Parent-provided targets, when implemented, must retain source/parent provenance and be measured separately from externally supplied content. Evaluation answers cannot be supplied by the same parents whose children are being selected. Unseen examples and independent target information are needed to detect shared errors.

## Parent–child selection cycle

1. Train founder models through selected curriculum stages with different seeds or declared learning settings.
2. Evaluate skill coverage, retention, adaptation, response time and memory use under equal declared budgets.
3. Choose two useful, sufficiently different parents and create child candidates using the chosen contribution method.
4. Train children on new selected material, with controlled retention of earlier material and any chosen parent teaching.
5. Compare children with both parents, a fresh model and a single-parent continuation at matched training budgets.
6. Select demonstrated improvements for the next generation and retain the evidence and lineage of rejected candidates.

This is an artificial evolutionary selection process. Producing one child from two models alone does not perform selection. Population-based training is a precedent for selecting model states and learning schedules, but does not establish that two-parent crossover will improve this spiking model. [Jaderberg et al., 2017](https://arxiv.org/abs/1711.09846)

For weight inheritance, compatible tensor shapes and the same byte vocabulary are necessary but insufficient. Independently learned hidden units can have different roles; arbitrary arithmetic mixing can destroy behavior. Model-alignment research motivates testing compatible common-lineage parents and aligned inheritance against simple controls. [Ainsworth et al., Git Re-Basin](https://arxiv.org/abs/2209.04836)

For teaching, keep both parents as frozen evaluators of the same selected examples, and train the child's predictions against a declared combination of actual targets and parent distributions. Their agreement is not proof that an answer is correct. Compare to actual-target-only learning. A fresh child's optimizer and transient neuron state must have explicitly defined initialization; histories cannot be blindly averaged.

BabyLM provides relevant small-data and curriculum research. Its findings show mixed curriculum results, so age ordering needs a shuffled-content control under the same exposure budget. It is a research reference, not an automatic addition of its entire corpus to this project's allowlist. [BabyLM findings](https://aclanthology.org/2023.conll-babylm.1/)

## Required model record

The implemented population registry records model IDs, parent payload hashes, generation, birth procedure, simulation birth tick, inherited lifespan, mutation settings and reproduction ceilings. Per-round reports record fresh held-out evaluations, selected blocks, births, resource limits and old-age deaths. Dead models release population credits while their checkpoints remain archived. Developmental-stage mastery, cumulative source manifests and teacher contributions still need a fuller individual record. Curriculum advancement should cite an evaluation artifact and preserve the previous checkpoint. A browser/laptop export should carry inference weights and required neuron state separately from the training history.

The native spiking learner, continuous learning/generation, replay, adaptive neurons, selective consolidation and selected general foundation curriculum are implemented. A [native live curriculum](live-curriculum.md) advances fixed exposure stages in one runtime and preserves history across stage transitions and restarts. Native population selection, two-parent block inheritance, bounded size mutation, GPU resource pressure and age-related death are implemented; automatic mastery promotion, teacher-generated targets, structural pruning and a town simulation remain future work.
