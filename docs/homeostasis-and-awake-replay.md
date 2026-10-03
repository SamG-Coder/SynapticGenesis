# Research: activity regulation and replay during ongoing learning

Review dated **3 October 2026**. The useful engineering distinction is between responding to an input, regulating activity, and preserving learned information. These processes can share one model while operating on different timescales. This follow-up examines four primary studies and the existing CUDA implementation. It does not introduce a new learning rule or claim biological equivalence. Reading boundaries are in the [source ledger](../reports/homeostasis-and-awake-replay.json).

## What the primary evidence supports

**Activity can regulate synaptic strength in both directions.** Turrigiano and colleagues manipulated activity in cultured rat cortical neurons and measured miniature excitatory currents. Reduced activity increased their amplitude; sustained disinhibition decreased it. Their distribution analysis supported approximately multiplicative changes. This is evidence for compensatory regulation in that preparation, rather than a rule that all connections strengthen with age. It does not determine a target firing rate or update interval for our byte-based model. [Nature, 1998](https://www.nature.com/articles/36103), [university-hosted paper](https://pages.ucsd.edu/~msereno/_107B-201_2007/readings/02.11-ActivScalesQuanta.pdf).

**The sensor and response have separate timing constraints.** Zenke, Hennequin and Gerstner simulated recurrent spiking networks with triplet spike-timing-dependent plasticity. Their stability analysis required rapid detection of activity changes relative to plasticity. This is a computational result under specified network and learning-rule assumptions, not a measurement establishing one universal biological timescale. Their model also exhibits failure regimes when regulation is too slow or too fast. [PLOS Computational Biology, 2013](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003330).

**Adding feedback can itself destabilize a network.** Harnack and colleagues analyzed intrinsic-excitability regulation. Recurrence, nonlinear responses and delay through intermediate filters constrained controller speed; a controller stable for an isolated neuron could destabilize a recurrent network. Their discussion explicitly relates these constraints to the need for sufficiently rapid synaptic regulation. These papers examine different controllers and assumptions. Neither establishes that more feedback, more timescales or faster adjustment is automatically better for our architecture. [PLOS Computational Biology, 2015](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1004357).

**Memory-related activity can matter during wakefulness.** Jadhav and colleagues disrupted hippocampal sharp-wave ripples while rats learned a spatial alternation task. A control delivered stimulation after the ripple. Disruption impaired the outbound component while measured place fields and later rest reactivation remained intact. The intervention supports a causal role for awake ripples in that task; it does not isolate replay content from every other ripple-associated process or specify a language-model replay algorithm. [Science, 2012](https://pubmed.ncbi.nlm.nih.gov/22555434/), [author-hosted paper and supplement](https://rnel.rice.edu/pubs/Jadhav%20et%20al_2012_Awake%20Hippocampal%20Sharp-Wave%20Ripples%20Support%20Spatial%20Memory.pdf).

## Mapping these ideas to the current implementation

The following distinctions come from repository inspection and are our engineering interpretation, not results from the papers.

| Existing mechanism | What it changes | What it does not establish |
| --- | --- | --- |
| Membrane state and selective spike traces | The response to preceding bytes within a stream | Durable protection of previously learned skills |
| Associative matrix state | Temporary key/value associations during the shared forward pass | A separate permanent store or a hippocampus equivalent |
| Optional ALIF cell | Thresholds depend on a recent absolute-spike average | Lifelong activity regulation in the associative cell |
| Source replay | Additional updates on stored descriptors into selected material | The best timing, mixture or amount of rehearsal |
| Synaptic-importance penalty | A surrogate estimate discourages movement of selected weights | A causal measurement of each connection's importance |
| Population age and lifespan | Reproduction eligibility, resource accounting and death | A measurement of an individual's learning capacity |

The associative cell uses `trace_fwd` with fixed firing boundaries at -1 and +1, a learned selective trace, and continuous emissions. Its signed trace is not a running estimate of firing frequency: positive and negative spikes can cancel. The separate ALIF cell already has activity-dependent thresholds, but its trace is transient stream state. Combining it with the associative cell would require a declared new cell definition, its backward equations and checkpoint semantics. It cannot be inferred from the fact that both features exist somewhere in the repository. See [trace dynamics](../src/trace_neuron.cuh), [adaptive dynamics](../src/adaptive_neuron.cuh) and the [shared model](../src/model.cuh).

Our [activity census](activity-diagnostics.md#completed-census) found no neuron that never fired across its sampled windows. That weakens a never-firing-neuron trigger for those observations; it does not demonstrate ideal activity, feature usefulness or preserved learning capacity. Continuous emissions also make spike count alone an incomplete measure. A lower spike count would not establish faster CUDA execution because the current matrix operations are dense.

## Two bounded research directions

**First, test replay timing without changing replay quantity.** The [existing replay implementation](../src/live_replay.cuh) stores selected-source window descriptors and uses separate recurrent state with the same weights. A useful follow-up could compare evenly interleaved rehearsal with short bursts using exactly the same source windows, total updates, objective weights and maximum storage. Measure earlier skills, new-source development loss, interruption latency and the largest transient regression between bursts. Report both end-of-run quality and the cost of remaining responsive while learning. An event-triggered arm would additionally need a declared trigger and a matched budget; otherwise extra rehearsal can masquerade as better scheduling. The biological study motivates the question, not a particular interval.

**Second, investigate bounded activity feedback only if diagnostics identify a useful target.** A candidate could maintain a running nonzero-spike estimate and a bounded per-neuron threshold offset, updated from admitted observations. With four layers of 512 neurons, two FP32 arrays alone would occupy 16 KiB; counters, reductions, serialization and execution cost are additional. This is an accounting estimate, not an implemented allocation or performance result. A new target activity range must be justified by retained skills and adaptation, not by forcing every neuron to one arbitrary rate.

Any such feedback needs an explicit contract:

1. Separate the sensor's averaging interval from controller gain and bounds. Measure oscillation, saturation and recovery after input-distribution changes before a longer study.
2. Preserve one CUDA forward implementation for learning and generation. Apply durable policy updates at completed observation/update boundaries; keep evaluation and generated-only text from silently changing the learning policy.
3. Persist controller state for exact continuation, distinguish it from disposable replay/query state, and include its memory in population budgets. State inheritance by a child is a separate controlled choice.
4. Compare with an unchanged model and a simpler fixed-threshold or fixed-penalty control under identical selected-source exposure. Count all extra updates and diagnostics.
5. Require gains in the intended outcome without concealing earlier-skill loss. Stable activity alone is not useful language or preserved plasticity.

The native [reading-adaptation diagnostic](adaptation-probe.md) is the immediate test of fitting, generalization and retention. Its declared conditions remain fixed while it runs. These research proposals do not alter it, select a winning partial result, or promote a new production policy. If new stories are learned while earlier skills disappear, protecting those skills has stronger local evidence than adding neurons solely because the model is older. If learning capacity deteriorates on multiple comparable tasks, then selective renewal or additional capacity can be tested with the controls in the [adaptation research design](plasticity-experiment-design.md).

## Development and generations

Keep three records distinct: population lifetime, curriculum exposure, and measured capability. Let reproduction fitness include retention and the ability to learn unfamiliar selected material, alongside useful responses and resource cost. A large or old model should not win by definition, and a newborn model's zero exposure counter should not conceal inherited capability. Those are evaluation design choices for SynapticGenesis, not biological claims.

The existing lineage, lifespan, resource-pressure and size-mutation mechanisms can support that experiment. Demonstrating improvement still requires children to outperform declared parent, continuation and fresh-start controls. None of the four studies reviewed here establishes that averaging two trained networks produces a better child. The [development design](general-development.md) retains that separate evidence requirement.

Research papers remain references. They have not been added to the training allowlist, and no external model weights or paper implementations were used for this review.
