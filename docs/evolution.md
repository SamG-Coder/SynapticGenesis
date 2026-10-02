# Native population and inheritance

`population-add` and `evolve` execute in the C++/CUDA executable. The population directory holds model checkpoints, lineage records and per-round evaluation reports. Models train with the existing native `train` and `live` commands.

## Register two trained founders

```powershell
.\build\synapticgenesis.exe population-add --population runs/population --id founder-a --checkpoint runs/founder-stage-3/best.ckpt --data data/prepared/foundations-v1/validation.dat --max-score 3 --growth-chance 0.1 --setting-mutation-chance 0.2
.\build\synapticgenesis.exe population-add --population runs/population --id founder-b --checkpoint runs/another-founder/best.ckpt --data data/prepared/foundations-v1/validation.dat --max-score 3 --growth-chance 0.2 --setting-mutation-chance 0.3
.\build\synapticgenesis.exe evolve --population runs/population --data data/prepared/foundations-v1/validation.dat --round round-1 --children 2 --seed 1337
```

Supply a second model trained from selected sources. Registration copies its checkpoint into the population and gives it a founder record; it does not automatically make it eligible to reproduce. Names must be distinct alphanumeric IDs with optional hyphens/underscores. Existing IDs and round names cannot be overwritten.

The first registration fixes the evaluation corpus hash, batch size, context, evaluation batch count and size-cost coefficient in `population.sg`. Later registrations and rounds preserve that protocol. The default uses 8 sequences, 128 target bytes, 32 sampled evaluation batches and strict FP32. A corpus identical to the recorded training corpus is rejected.

## Selection and improvement gate

The native evaluator freshly scores each model on identical held-out windows. It never accepts a model's own claimed fitness. Lower is better:

```text
score = next-byte cross-entropy + size_cost * parameters / 1,000,000
```

The default size cost is 0.02 per million parameters. It is a declared resource preference, not a measured energy model. Founders require a completed training update and a score strictly below their registration ceiling. Living eligible models are ranked, and the best fraction becomes the parent pool (default one half before scarcity adjustments, with a minimum of two when available). Resource limits exclude oversized models.

Two distinct compatible elites are selected using the recorded RNG seed. Compatibility currently requires equal channel width, block count and cell type. Hidden-neuron counts may differ. Children receive a reproduction ceiling equal to the better parent's measured score minus the required improvement (base default 0.005, increased under scarcity). They cannot reproduce at birth, even if their initial loss happens to be lower. After training they must beat that ceiling and rank among the elites.

This reuses a development set for selection; it is not proof of improvement on independent tasks. Keep final test data out of reproduction decisions. Generalization, multiple seeds and matched-budget controls remain necessary.

## Inherited DNA and mutation

- Embedding and readout come from the better-scoring parent.
- Whole residual blocks come from either parent. The default donor probability is 0.25 per block, with at least one donor block when crossover is enabled. `--crossover 0` gives an inheritance control.
- Learning rate starts at the geometric mean of parental rates; a mutation can scale it between one half and twice that value, within bounds. Weight decay, gradient clip and activity cost use parental means.
- Growth probability and setting-mutation probability are inherited as parental means. Parents may therefore carry different developmental settings.
- A width mutation attempts roughly 25% more hidden neurons, rounded to groups of eight, with `--max-hidden` and `--max-parameters` limits. Defaults are 2,048 hidden neurons and 4,000,000 parameters. The child starts with at least the larger parental width.

New neurons have randomly initialized input connections and zero output connections. With crossover disabled, width expansion preserves the original forward function at birth while allowing new outgoing connections to learn. Native tests check this for LIF and ALIF. This is an engineering construction related to function-preserving network expansion, rather than a simulation of biological growth. [Net2Net](https://arxiv.org/abs/1511.05641)

Crossover is separate from width expansion: swapping blocks between independently trained networks can degrade predictions. This implementation does not align learned channel bases. Children are candidates for further learning and must pass the gate; their birth is not an accepted improvement.

Each child starts with fresh optimizer moments, recurrence, replay and consolidation history. Parent payload hashes, IDs, generation, selected blocks, sizes, settings and birth losses are recorded in the round JSON. `member.sg` holds the durable lineage and reproduction gate.

## Train a child, then run another round

```powershell
.\build\synapticgenesis.exe train --resume runs/population/round-1-child-0/latest.ckpt --allow-new-corpus --data data/prepared/foundations-v1/train.dat --validation data/prepared/foundations-v1/validation.dat --out runs/population/round-1-child-0 --steps 3000
.\build\synapticgenesis.exe evolve --population runs/population --data data/prepared/foundations-v1/validation.dat --round round-2 --children 2 --seed 1338
```

Train the other child similarly if desired. Selection reads each member's `latest.ckpt`. A weak child stays ineligible; there is no automatic promotion. A child's generation is one plus the larger parent generation, independently of its training exposure or curriculum stage.

Calls are bounded by their requested children and training updates. There is no unattended reproduction service. Use one writer per population directory.

## GPU memory as population food

Each round reads actual free/total GPU memory with `cudaMemGetInfo`. By default it reserves 1,024 MiB for workspace/other use and assigns half of the remaining free memory as population credits. `--food-fraction`, `--reserve-mib` and an optional lower `--food-mib` cap control that policy. Zero food allows the clock to advance but prevents births.

Each living individual consumes 20 bytes per parameter for weights, gradients, Adam moments and a decay array, plus its recurrent stream state. This is a virtual accounting policy: the current population is checkpointed on disk and executed sequentially. It does not assert that every member is GPU-resident. Additional birth/evaluation admission checks account for the explicit native execution buffers and the reserve. CUDA allocations can still fail if another process changes available memory after the check.

At pressure `p = clamp(living_credits / capacity, 0, 1)`:

```text
effective elite fraction = configured fraction * (1 - 0.9 * p)
required child improvement = base improvement * (1 + 4 * p)
```

The parent pool and improvement requirement tighten as resources fill. Parent selection is recalculated before each birth. The minimum pool of two never overrides fitness, lifespan or capacity checks. A birth requires enough remaining credits and temporary execution space; otherwise the round records `resource_limit`. Resource limits are admission rules, not an invitation to allocate every free GPU byte.

```powershell
.\build\synapticgenesis.exe evolve --population runs/population --data data/prepared/foundations-v1/validation.dat --round round-3 --children 8 --food-fraction 0.5 --reserve-mib 1024 --seed 1339
```

## Lifespan, death and continued generations

`population-add --lifespan 20` gives a founder 20 simulation ticks by default. Registration records the current population tick as birth time. Each completed `evolve` call advances the clock once, including rounds with no births. A tick is an explicit simulation step, not a human year, a training update, or the model's curriculum level.

A member dies when `current_tick - born_tick >= lifespan`. Dead members are excluded from evaluation/parent selection and consume no population credits. Their checkpoints and lineage stay on disk. The round report records age, lifespan and `old_age` as the death reason. Children inherit the rounded mean parental lifespan and start at age zero, independently of their generation number.

If all qualified adults die before their children improve enough to reproduce, the population can stop producing children. The system records `no_eligible_pairs`; it does not bypass the quality gate to keep births going. Registered founders can seed a new population when explicitly requested.

Version-2 population/member files persist ticks and lifespans. Version-1 records remain readable and begin at tick zero with a 20-tick lifespan because the old format had no age information. The round JSON records both resource decisions and lineage. Keep independent backups for long experiments: population rounds are a single-writer research protocol, not a transactional database or a background service.
