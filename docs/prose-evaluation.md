# Comparing larger models at fixed exposure

The [evaluation specification](../data/prose-evaluation-v1.json) fixes the
development checks for the expanded prose curriculum. It is declared while
the 105M founder continues learning. Its first stage already had the exploratory
reader check and two generations reported in [the founder record](prose-105m-founder.md).
At declaration time, the remaining checkpoints and matching smaller controls
had not been assessed. The [105M assessments](prose-105m-complete.md) have since
completed at all four stages. The [complete three-size result](prose-size-results.md)
now includes both smaller controls and their raw generations.

The 2M, 27M and 105M profiles use the same seed, learning rate, source order,
128-byte learning chunks, stage replay and scheduled speech. Compare their
immutable stage checkpoints at 8,176, 26,110, 95,207 and 216,289 source
observations. These are equal data exposures, not equal compute or wall time.
One seed and one common learning rate do not establish an optimized size ranking.

Each assessment reports:

- Separate byte losses on four validation books: *New National First Reader*,
  *Home Geography for Primary Grades*, *The Velveteen Rabbit*, and
  *Far from the Madding Crowd*. Their unweighted mean gives each book equal
  influence, rather than letting the longest book dominate.
- Separate losses on the first two training readers as retention monitors.
  These scores describe previously observed material; they are not held-out
  generalization results. Their change from stage one shows whether later
  learning degrades that measure of early reading.
- Four raw 256-byte continuations with fixed prompts and sampling settings.
  Two are prose openings and two are questions. A question-shaped prompt does
  not turn book pretraining into instruction training; publish failures too.

The native evaluator uses strict FP32, 16 sequences of 128 bytes in each of
32 batches, and its fixed sampling seed 712367. Thus every assessment scores
the same 65,536 sampled target bytes per book. Recurrence resets for every
window. This measures short-window prediction, not long-context recall or
the completeness of remembered facts. No reserved test book is scored.

The driver binds the selected source specification, prepared books, schedule,
executable and scripts before native work. Assessments require immutable
`stage-N.ckpt` files, verify founder settings and record the checkpoint hash
before and after. All raw samples and native command logs remain in the run
directory. The public declaration itself contains no new quality results.

```powershell
python scripts/prose_evaluation.py declare --out runs/prose-evaluation-declared

# Run after the active GPU learning session has finished.
python scripts/prose_evaluation.py assess --plan runs/prose-evaluation-declared/protocol.json --run runs/prose-105m-founder --stage 1 --out runs/prose-evaluation-declared/105m-stage-1
```

CPU validation authenticated the complete source edition and schedule, checked
the two existing 105M stage headers against the declaration, and compiled the
Python driver. Native execution has since completed all forty commands for the
105M founder: six book evaluations and four generations at each of four stages,
with checkpoint identities unchanged. The matching smaller-model portion has
also completed, giving 120 native assessment commands across twelve checkpoints.
Capacity benchmarks remain separate evidence about speed.

Increasing parameters also increases the amount of learning needed. The
[Chinchilla study](https://arxiv.org/abs/2203.15556) found that model size and
training tokens should grow together for compute-efficient transformer training.
That result motivates measuring both data and capacity here, but its fitted
ratios are not established laws for this byte-level spiking architecture.
Our 4.82-million-word collection is a useful larger experiment, not demonstrated
sufficient training for a broadly capable 105M model.

## Sequential execution

`scripts/prose_size_comparison.py` runs the declared comparison without
overlapping GPU experiments. It first requires the original 105M founder's
complete stage-four checkpoint and final session. It assesses that model, then
trains fresh 2M and 27M founders through the identical source schedule and
assesses all four checkpoints of each. It journals each completed assessment
to `partial.json` before assembling the final comparison.

For an already-running founder, `--wait-pid` can bind its actual native process
handle before any new GPU work. The gate checks the executable path, records
the process creation identity and requires a successful exit. Holding the
handle prevents PID reuse from satisfying the wait. An observation timeout
does not cause a restart. The completed checkpoint and session must then agree
on source observations, source bytes, replay count and scheduled generation.
The native evaluator performs full checkpoint validation when it loads each
model; the small Python header reader is only an exposure diagnostic.

After the original native process has exited, the runner also works without
the optional process gate:

```powershell
python scripts/prose_size_comparison.py --out runs/prose-size-panel
```

CPU checks verified waiting for a successful child process, rejection of a
nonzero exit, rejection of a mismatched executable, and rejection of the
unfinished founder before assessment. The first three existing stage headers
matched the declared exposure boundaries. The original live run has now exited
successfully, its final checkpoint and session counters agree, and all four
105M assessments completed. Full three-size orchestration has now completed
successfully, with all twelve assessments verified for publication.

The final comparison requires all three models to have identical source,
replay and scheduled-speech counters at each stage. It reports each validation
book, the validation mean and each training monitor's change from stage one.
All raw samples remain available. The original founder's continuation saves
every 8,192 observations while the smaller driver saves every 2,048, so total
session time is not treated as a controlled size-speed comparison. Saving does
not change the declared learning policy; the separate capacity experiment
provides the controlled speed measurements.
