# Native reading-adaptation diagnostic

The [declared design](plasticity-experiment-design.md) compares fresh and experienced learners under the same new-source policy. `experiments/adaptation_probe.cu` now runs that comparison on disposable model copies, calling the production `Model::forward`, `backward` and `update` methods. No second set of neuron equations or optimizer kernels is introduced.

The executable has three initialization modes. `fresh` uses the existing random initializer and zero Adam history. `carry` reads the original checkpoint through the authoritative native loader and copies weights, both Adam moments and the global optimizer step. `reset` copies exactly the same weights but starts both moments and the step counter at zero. The original live checkpoints retain their curriculum, replay, teacher, lifespan and lineage state; the diagnostic does not overwrite them.

The comparison uses strict FP32, batch 1, context 128 and unweighted byte loss. Every training window resets membrane, continuous trace and associative state. The temporary copies omit replay, teachers and consolidation penalties as declared. This measures adaptation under a controlled policy; it does not measure the full live system with all its memory mechanisms enabled.

## Sources, evaluation and restart

`scripts/adaptation_sources.py` authenticates the selected early-reader edition and creates a fixed schedule for each seed. A schedule begins with `SGADAPT1`, context, window count and a source hash. Each subsequent row records a document index, byte offset and the hash of its 129 source bytes. Native admission validates every row before training and rejects windows crossing a document boundary. The external JSON record also retains SHA-256 identities. Source selection is uniform over documents, then over valid byte starts within the chosen document.

Training and development evaluation cover every in-document next-byte pair once in consecutive 128-target windows, including the exact shorter tail. Each window starts with reset recurrent state. Scores are reported per document and pooled by target count. Development text supplies no parameter updates. Earlier book and binding assessments use the existing production CLI and their established separate source editions. Reserved tests are not scored.

Evaluation must preserve the complete weight and optimizer arrays. Short evaluation windows use temporary execution views that share those arrays. Reported peak explicit model bytes include the temporary allocation before a view begins sharing; CUDA/context/cuBLAS overhead is excluded. Synchronized update timing excludes evaluation, logging, checkpoints and process startup. Shared desktop GPU use prevents interpreting it as an isolated hardware benchmark.

Each endpoint writes an ordinary model checkpoint plus an `.sgadapt` sidecar. The sidecar binds the entire checkpoint file to the initialization mode, original checkpoint, seed, learning rate, starting Adam step, source files, complete window schedule and completed probe updates. It also records the original weight/moment identities and has its own checksum. A resume must match all of these identities. The global optimizer step resumes from the original count plus completed probe updates; recurrence is reset for the next window.

These files are diagnostic copies. They are not replacements for population-owned live checkpoints. Their header identifies them as adaptation outputs, and their sidecar is required for diagnostic restart.

## Commands

```powershell
# Build in a separate directory, preserving the production study executable.
.\build.ps1 -AdaptationProbe
python tests/adaptation_probe.py --out runs/adaptation-validation

# Short integration exercise on all seven starts and both rates for seed 1337.
python scripts/adaptation_experiment.py --smoke --out runs/adaptation-smoke-new
python tests/adaptation_experiment.py --study runs/adaptation-smoke-new

# Declared 42-trajectory comparison. Use a fresh directory.
python scripts/adaptation_experiment.py --out runs/reading-adaptation-panel
python tests/adaptation_experiment.py --study runs/reading-adaptation-panel
```

The orchestration script materializes and hashes all source schedules before invoking a learner. It journals native commands, records each endpoint and assesses earlier skills at the beginning and end. The separate audit authenticates original files, initial carried/reset parameters, every presented window, checkpoint counters and evaluation aggregates. It performs no model computation.

## Verification boundary

The native CLI fixtures pass **30 actions**, including **18 rejected inputs**. Fresh, carried and reset runs resume to identical complete checkpoints. Changing intermediate evaluation frequency also leaves the final checkpoint identical. The checks verify that carried copies retain both moments and the update counter, reset copies have zero moments/counter and identical weights, and all source windows and evaluation target counts are correct. An independent CPU reference agrees with the first fresh fixture's window loss within **9.32e-8**.

A CUDA memory-sanitizer run covers learning, evaluation tails, endpoint persistence and a carried-state start, with **zero errors**. These are mechanism and compatibility checks. The earlier strict learned CPU/native score discrepancy and cold first-Adam-step discrepancy remain unresolved; a single new fixture does not establish general numerical parity.

The [validation report](../reports/adaptation-probe-validation.json) records the complete integration smoke: all **fourteen** declared short trajectories and **42 checkpoints** pass the state/exposure audit. Its 14 learning commands and 112 earlier-skill assessment commands preserve the original models. The full adaptation study had not started when this implementation stage was validated. No learning-capacity or general-language result is established by these checks.

The subsequent [completed 42-trajectory comparison](reading-adaptation.md) passes its execution audit across 210 checkpoints. It finds early development gains followed by overfitting and forgetting under the declared exposure budget. Its results and the implementation-validation evidence remain separate.
