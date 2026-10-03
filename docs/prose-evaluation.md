# Comparing larger models at fixed exposure

The [evaluation specification](../data/prose-evaluation-v1.json) fixes the
development checks for the expanded prose curriculum. It is declared while
the 105M founder continues learning. Its first stage already had the exploratory
reader check and two generations reported in [the founder record](prose-105m-founder.md).
The remaining checkpoints and matching smaller controls have not been assessed.

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
Python driver. Native execution of this new driver is pending the active
learning run. Existing capacity benchmarks and the earlier founder assessment
are separate evidence; they do not validate this driver's full execution.

Increasing parameters also increases the amount of learning needed. The
[Chinchilla study](https://arxiv.org/abs/2203.15556) found that model size and
training tokens should grow together for compute-efficient transformer training.
That result motivates measuring both data and capacity here, but its fitted
ratios are not established laws for this byte-level spiking architecture.
Our 4.82-million-word collection is a useful larger experiment, not demonstrated
sufficient training for a broadly capable 105M model.
