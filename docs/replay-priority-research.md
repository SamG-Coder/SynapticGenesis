# Choosing memories for live rehearsal

The completed [rehearsal comparison](live-reading-results.md) found a partial
retention benefit from more replay, with higher runtime cost and continued
forgetting. The [reading-breadth protocol](reading-breadth-experiment.md) tests
content and repetition separately from a new replay selector. This note proposes
a diagnostic: measure which stored experiences a new update damages
before changing how the live learner selects them. The
[first completed measurement](replay-priority-diagnostic.md) now reports that
signal and its runtime cost. No priority selector is implemented or promoted.

Schapiro and colleagues studied 24 people learning 15 novel objects. Their
fMRI analysis associated weaker initial object memory with more hippocampal
reactivation during subsequent awake rest. Later reactivation predicted
subsequent memory; its association with improvement across a 12-hour interval
was specific to the group that slept. This supports investigating selective
rehearsal. It does not identify a CUDA scoring rule, equate prediction loss with
human memory strength, or make an observation counter a biological age.
[Primary study and figures](https://pubmed.ncbi.nlm.nih.gov/30254219/).

Mattar and Daw model memory access as improving future decisions. Their
navigation simulations prioritize an experience by the expected benefit of
updating a decision and the expected future need for that decision. This is a
normative theory, rather than direct evidence that the brain ranks language
windows by loss. Its useful engineering lesson is to distinguish difficulty
from expected benefit. Our learner currently has no measured equivalent of
future decision need.
[Primary paper](https://pubmed.ncbi.nlm.nih.gov/30349103/).

Aljundi and colleagues' Maximally Interfered Retrieval, MIR, offers a machine
learning comparator. It estimates a new-data update, then ranks a random subset
of stored examples by how much their losses would increase. A second criterion
also uses the best previously observed loss. The final classifier update mixes
new and retrieved examples. The paper evaluates classification streams including
MNIST and CIFAR-10; this does not establish a benefit for our spiking language
model. Its virtual SGD step also differs from our history-dependent AdamW and
sequential source/replay updates.
[Primary paper, section 3.1 and Algorithm 1](https://proceedings.neurips.cc/paper/9357-online-continual-learning-with-maximal-interfered-retrieval.pdf).

Three candidate signals would answer different questions:

| Signal | What it measures | Main limitation in this project |
| --- | --- | --- |
| Current per-byte loss | How poorly a stored window is predicted now | High loss can mean unfamiliar or noisy content, without any recent forgetting |
| Loss increase after a new-source update | Immediate interference from that actual update | Scoring before and after costs extra forward passes and need not predict long-term retention |
| Increase from a previously recorded low loss | Deterioration relative to sampled history | A selected historical minimum can be optimistic; cached scores become stale |

These are engineering hypotheses inferred from the papers and local results.
None implies that replaying the hardest window is always useful. A selector
could concentrate on rare errors, neglect other stages, or spend more time
ranking memories than the uniform baseline spends learning from them.

The first bounded diagnostic should use disposable copies of complete learned
checkpoints and only previously admitted training windows. It should score a
small declared candidate pool, execute the actual prospective source update
with inherited Adam moments, then measure each candidate again. It should
record both the mean next-byte loss and the existing answer-weighted training
objective where applicable; a long context must not hide deterioration on a
short supervised answer. Candidate ranking must not read development answers
or reserved tests.

Both sides of each comparison must use the same explicitly defined recurrent
initialization. Extra diagnostic forwards must not advance the live learner's
source or speech state, RNG, replay descriptors, optimizer or counters. Full
checkpoint and continuation controls should verify that disabling the
diagnostic preserves the ordinary trajectory. An AdamW-aware diagnostic is an
adaptation of the interference idea, not an exact reproduction of MIR.

Before a selector becomes a learning experiment, report the rank agreement
between cheap cached scores and measured interference, selection stability,
extra forward work, device memory, total live time and speech interruption
latency. Keep stage coverage bounded and include a uniform-selection control
with the same candidate scoring overhead. A further comparison must match
wall-clock budget as well as source exposure: equal update counts alone would
give a more expensive selector extra compute. Preserve numerical failures near
firing thresholds instead of hiding them in an aggregate score.

Only a subsequently declared multi-seed retention result can justify changing
replay policy. This diagnostic cannot by itself establish durable language,
biological development, automatic neuron growth or generational improvement.
