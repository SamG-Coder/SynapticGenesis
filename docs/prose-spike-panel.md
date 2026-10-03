# Matched neuron-response diagnostic

The [saved membrane investigation](prose-spike-state.md) found large recurrent
states in the 105M founder. This follow-up fixes the observation text across
model sizes and saved ages. It is a read-only diagnostic, declared before its
new native traces, and does not change the ongoing learning policy.

Use the 2M, 27M and 105M random founders from the
[fixed-exposure comparison](prose-evaluation.md). Trace their untouched initial
checkpoint, stage one at 8,176 source observations, and stage four at 216,289.
For each checkpoint, run the same eight independent 1,024-target-byte windows
through the existing native production forward in strict FP32. The recurrence
starts at zero for each window. The Python runner only orchestrates native
commands, authenticates files and summarizes saved arrays.

Window selection is mechanical: in each source stage's admitted document order,
choose document indices `floor(count / 4)` and `floor(3 * count / 4)`. Take the
centered 1,025 input bytes, giving 1,024 next-byte targets. The selected edition
is `selected-prose-scale-v1`; its prepared manifest and source specification
are authenticated before declaration.

| Source stage | Gutenberg book ID | Starting byte offset |
| ---: | ---: | ---: |
| 1 | 14640 | 15085 |
| 1 | 14766 | 65733 |
| 2 | 43936 | 103346 |
| 2 | 271 | 153103 |
| 3 | 421 | 215243 |
| 3 | 1268 | 563700 |
| 4 | 834 | 290298 |
| 4 | 105 | 234136 |

These are training-source windows. The separate size comparison provides
validation-book losses and fixed raw generations. This diagnostic introduces
no reserved-test scoring and does not treat source-window loss as held-out
generalization.

For every layer, preserve the raw arrays and report:

- Positive, negative and zero spike fractions; the fraction of neurons whose
  signed spike is constant over the observed samples; mean per-neuron marginal
  spike entropy.
- Adjacent signed-spike changes and direct positive-to-negative or
  negative-to-positive transitions. Pooling excludes artificial transitions
  between independent windows.
- Absolute membrane, input-drive and emitted-signal distributions; per-neuron
  temporal variation of the continuous emitted signal.
- The local triangular spike-surrogate coefficient, evaluated in float32
  including cancellation close to zero. Its zero fraction concerns that local
  derivative only, not the full model gradient.
- Mean next-byte loss and an independent stable-softmax reconstruction of
  each target's loss from the native logits.

Marginal entropy is not joint information, language competence or a pruning
criterion. Pooling emission variation includes differences between windows.
Short reset windows do not reproduce the live stream's long recurrence. Dense
CUDA projections still perform dense work regardless of measured spike count.

## Execution and evidence

The runner first requires the complete size-comparison report, matched source,
replay and scheduled-generation counters, and the recorded learned checkpoint
hashes. It can wait on the comparison runner's verified Windows process handle,
requiring that process to exit successfully before any new GPU command. This
sequences it after both the large founder and the smaller controls.

The separately rebuilt trace executable first repeats a preserved admitted 2M
checkpoint's 256-byte trace. All 26 output files must match their published
SHA-256 identities exactly. A mismatch stops the diagnostic and saves the
failure; it is not accepted by relaxing a numerical threshold. The panel then
issues 72 new forwards across nine checkpoints and eight windows, for 73 native
commands including that compatibility check. It never performs a weight update.
Checkpoint hashes are checked before and after each group of windows. Each
completed checkpoint's measurements are journalled before proceeding.

CPU validation has passed the constant-versus-alternating signed-spike cases,
independent-window boundaries, exact firing boundaries, near-zero float32
cancellation, and four malformed-input rejections. Parsing the existing
preserved native trace matched all 26 file identities and reconstructed its
per-target loss with maximum absolute error `6.5316e-7` nats. Source declaration
produced the eight windows above, each 1,025 bytes. The separate native target
built successfully and the ongoing learner's executable hash was unchanged.

The rebuilt executable's compatibility run and all 72 matched forwards remain
pending. The cached-trace check verifies the new reader; it does not establish
that the rebuilt executable or full new orchestration has passed. No activity
controller, reset change, learned threshold or quality improvement is claimed.

After the size comparison is complete:

```powershell
python scripts/prose_spike_panel.py prepare --out runs/prose-spike-panel
python scripts/prose_spike_panel.py run --out runs/prose-spike-panel
```

When queuing behind a live comparison, pass its actual process ID with
`--wait-pid` and its full Python executable path with `--wait-executable` to
the `run` command. The declaration pins the executable, source files, analysis
scripts and window hashes; inputs are rechecked after waiting and on completion.
Keep declared inputs unchanged until the panel has finished. Use a fresh output
directory for a new declaration rather than overwriting an earlier protocol.
