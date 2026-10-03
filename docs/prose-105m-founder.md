# First checkpoint of the larger founder

A new C1024/H4096/L8 associative spiking model was initialized randomly and
trained through the ordinary curriculum CLI. It has **104,851,472 parameters
and 32,768 spiking neurons**. It uses no imported weights or pretrained
tokenizer. The [expanded reading edition](prose-scale-corpus.md) is bound into
its four-stage source schedule.

The first stage completed 8,176 source observations: four complete passes
through the four elementary readers, totaling **1,045,096 source target-byte
exposures**. It also completed 2,044 ordinary stage-replay updates with 261,337
replayed target bytes, and generated 1,536 bytes that never became targets.
The durable first-stage checkpoint is
`runs/prose-105m-founder/stage-1.ckpt`; the [report](../reports/prose-105m-founder.json)
records its SHA-256, exact source/executable identities and session counters.

The live session took 156.04 seconds, about 6,698 source bytes/s. This timing
includes intermediate and final checkpoint saves, replay and scheduled speech;
it excludes model initialization and the initial checkpoint save. The separate
[capacity benchmark](bounded-live-memory.md) provides generation throughput
and buffer measurements. Update-tick p95 was 28.83 ms, and ticks including
96-byte speech had p95 112.86 ms; these are histogram upper bounds.

This early model is **not yet coherent or useful for questions**. A development
reader check gave 2.187037 nats/byte over 65,536 sampled target bytes. There
was no matched smaller-model quality comparison. Both raw sample prompts below
used strict FP32, graph execution, seed 42, top-k 40 and temperature 0.8 after
resetting recurrence. Neither assessment changed the checkpoint.

```text
Prompt: The bird
Output begins: The bird moning in
a now hi live son find every lingir to night it.
```

```text
Prompt: Question: What is water?
        Answer:
Continuation begins: I knear birls thart, and white must our out, and
of the childres.
```

The report retains both complete 128-byte outputs and their raw hex. This is
an initial pipeline and learning checkpoint, not evidence of a useful answer,
general capability, or a language-quality benefit from scale. The model has
not completed the 4.82-million-word edition at this checkpoint; stages two
through four contain the remaining books. No reserved tests were scored and
the founder has not been admitted for reproduction.

```powershell
python scripts/prose_founder.py --out runs/prose-105m-founder --updates 8176 --profile 105m

.\build\synapticgenesis.exe live --resume runs/prose-105m-founder/latest.ckpt --curriculum runs/prose-scale-curriculum/curriculum.sg --out runs/prose-105m-founder --updates 216289 --prompt "The bird " --log-every 2048 --save-every 8192
```

Resume preserves dimensions, recurrence, optimizer, RNG, source identities and
replay. The same source driver also supports a fresh 27-million or 2-million
parameter control through its `--profile` argument. Such controls still need
to be run before claiming a learning advantage for the larger model.
