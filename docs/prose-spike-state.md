# Membrane snapshots from the larger founder

Read-only CPU inspection of the first three immutable 105M stage checkpoints
found large stored membrane values. This provides a concrete reason to
investigate activity regulation while preserving the ongoing baseline run.
It does not yet establish why language quality is weak or which change helps.

The associative model uses the same signed trace-neuron equations in learning
and generation. For the last processed byte, let `u` be the membrane before a
spike, `s` the signed spike, and `r` the stored value after subtractive reset:

```text
s = 1[u >= 1] - 1[u <= -1]
r = u - s
surrogate(u) = 0.3 * (max(0, 1 - abs(u - 1)) + max(0, 1 - abs(u + 1)))
```

For this cell, `abs(r) >= 1` implies `abs(u) >= 2`, where that local spike
surrogate is zero. This is a conservative diagnostic: a zero stored reset
does not tell us whether the preceding `u` was zero or exactly at a firing
threshold. The untouched initial state has no preceding byte at all.
The equation and reset behavior are in
[the trace kernel](../src/trace_neuron.cuh) and
[the shared spike surrogate](../src/spike_lm.cu).

| Saved source observations | Lowest layer fraction with proven zero last-step local spike surrogate | Highest layer fraction |
| ---: | ---: | ---: |
| 8,176 | 83.03% | 94.12% |
| 26,110 | 88.57% | 97.39% |
| 95,207 | 88.09% | 98.12% |

For example, layer five (zero-based index 4) at 95,207 observations has a
median absolute stored membrane of 49.26 and a 90th percentile of 202.48.
The firing boundary itself is one. These values describe the saved recurrence;
they are not firing-rate measurements or a model-wide gradient norm.

The [machine-readable report](../reports/prose-105m-state-snapshots.json)
records each layer, the untouched initial snapshot, checkpoint hashes and the
exact native equation files. The reader seeks directly to recurrent state
after the three parameter arrays, loading only the bounded state payload.
It computes SHA-256 before and after inspection and performs no model
forward pass, weight update, generation or GPU operation. Its interpretation
was checked on 20,006 scalar states, including the exact firing and surrogate
boundaries. Native loading remains the authority for payload checksum and
full format validation; this small diagnostic does not replace it.

There are several limits to the inference:

- These are single snapshots at different source positions, not a matched
  observation stream. They cannot establish a time-averaged trend or isolate
  model size from training history.
- A zero local surrogate at one byte does not prove that a neuron never
  learns. Earlier byte positions, recurrent carries, trace gates, output
  projections and associative paths can still contribute gradients.
- Positive and negative spikes carry different signals. High nonzero-spike
  frequency alone is not constant output, and a continuous trace can carry
  information between spikes.
- The current CUDA projections are dense. Reducing spike frequency alone
  would not prove a throughput improvement.

The next diagnostic should use the same source windows for different model
sizes and saved ages, measuring membrane distributions, surrogate support,
sign changes and emitted-signal variation alongside held-out loss. The
existing native trace executable can supply those arrays without another
forward implementation. Any threshold, drive-normalization or reset-policy
experiment must preserve the current baseline and test learning and retention
under matched exposure. The [activity-regulation review](homeostasis-and-awake-replay.md)
explains why a controller needs its own stability and outcome checks; this
snapshot evidence does not select a target firing rate.

```powershell
python scripts/spike_state_snapshot.py --stages 1 2 3 --out reports/prose-105m-state-snapshots.json
```
