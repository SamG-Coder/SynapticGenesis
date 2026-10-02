# Controlled retention during live learning

`retention-bench` tests the real native live engine while the observed source distribution changes. It keeps generation active and never trains on the generated text. This measures byte-language retention and transfer to separate books, not mastery of facts or human developmental age.

## Protocol

1. Start one model from random weights and learn the approved reading corpus for 6,000 observed updates. Reservoir replay runs every four updates and SI importance is tracked with penalty strength zero.
2. Save its complete weights, Adam moments, neuron state, speech RNG, source reservoir and consolidation history.
3. Reload that same checkpoint for each intervention. Introduce the selected science book and observe exactly 6,000 more chunks, each up to 128 source byte pairs. An explicit `new` curriculum scope prevents old books from entering the new online stream at EOF.
4. Evaluate both separate held-out books before the shift and every 1,000 later updates using the same fixed strict-FP32 reset-state windows. Each domain uses 16 sequences × 128 targets × 32 batches. Sampled windows may overlap. The final test book never enters this experiment.

The interventions are:

| Arm | Extra updates after the shift | Plasticity change |
| --- | --- | --- |
| `none` | None | Base learning rate 0.0003 |
| `current` | Repeat the current observed chunk every four updates | Same rate |
| `reservoir` | Sample an older observed window every four updates | Same rate |
| `reservoir_si` | Same source reservoir | SI penalty 0.001 |
| `reservoir_slow_core` | Same source reservoir | Embedding/block rate ×0.25; full-rate output head |
| `reservoir_low_lr` | Same source reservoir | Whole-model rate ×0.25 |
| `none_low_lr` | None | Whole-model rate ×0.25 |

`current` controls for doing additional optimizer updates without retrieving old examples. The whole-model low-rate arms test whether slowing the entire learner explains gains otherwise attributed to selective protection of older connections. All arms track SI arrays, including the zero-strength controls, so reported runtime is not the minimum cost of disabling SI entirely. Current and reservoir replay receive the same number of updates, but actual replay byte counts can differ at document tails; these counts are published.

The no-replay/current controls intentionally discard the common checkpoint's source reservoir at the branch. Other starting model state remains identical. There is no automatic promotion or model selection inside this command. Negative forgetting means held-out old-domain loss improved after new learning; positive forgetting means it worsened.

## Run

Prepare the [selected extension](data.md) first. Use fresh output directories:

```powershell
.\build\synapticgenesis.exe retention-bench --train-a data/prepared/development-v2-final/through-stage-3.dat --train-b data/prepared/development-v2-final/stage-4.dat --validation-a data/prepared/development-v2-final/13853.txt --validation-b data/prepared/development-v2-final/12228.txt --steps-a 6000 --steps-b 6000 --seed 1337 --fast --out runs/retention-controls-1337
```

Repeat with seeds 2026 and 31415 to reproduce the declared comparison. Native C++/CUDA performs all learning, replay and evaluation. No external/pretrained model supplies targets. Model size defaults to 1,186,304 parameters (signed LIF); dimension and ALIF options are available for separate experiments. The protocol uses one observed sequence per update and graph generation of 96 bytes every 500 observations.

The output contains initial/common checkpoints, a fixed curriculum and per-arm checkpoint, transcript and evaluation curve. `result.json` records source identities, common-checkpoint identity, losses, source/replay counts, plasticity settings and measured latency. Throughput divides new observed pairs by time spent in update/generation ticks; this excludes evaluation, checkpoint/log I/O, source transition and setup. It is not directly comparable to a batched training throughput measurement or a loop timer that includes those costs.

The independent CLI fixture reloads each arm checkpoint in another native process and verifies reported held-out losses, update counts and new-domain cursors. It also rejects train/evaluation equality and protects existing experiments. Source-preparation checks handle whole-book/paragraph separation; neither these checks nor the experiment rule out paraphrase overlap.

The comparison is exploratory: a few seeds and two small historical language domains cannot establish a universal memory policy, long-term retention, biological fidelity or conversational competence. Recommended development settings should remain explicit and reversible as broader content and skill probes are added.
