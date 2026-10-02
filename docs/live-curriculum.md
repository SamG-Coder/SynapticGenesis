# Continuous development in the native learner

The live curriculum changes the source distribution while one model keeps learning and speaking. Weights, Adam moments, replay windows and optional synaptic-importance history survive the change. This implements scheduled exposure to harder content; stage numbers are not human ages or evidence of mastery. Simulation lifespan still belongs to the population clock.

## Prepare and run

The preparation script writes cumulative source editions and a schedule alongside the existing independent stage files. If an earlier prepared directory already exists, use a fresh one; the script never overwrites an existing corpus:

```powershell
python scripts/prepare_corpus.py --out data/prepared/foundations-live-v1

.\build\synapticgenesis.exe live --curriculum data/prepared/foundations-live-v1/curriculum.sg --validation data/prepared/foundations-live-v1/validation.dat --out runs/developing-founder --lr 0.0003 --seed 1337 --chunk 128 --replay reservoir --replay-capacity 1024 --replay-every 4 --graph --speak-every 500 --tokens 96 --prompt "The bird " --fast
```

Omitting `--updates` runs to the final scheduled update. To stop earlier, pass a lower total `--updates` count. Resume with `--resume runs/developing-founder/latest.ckpt`, the identical `--curriculum`, output directory and prompt. Do not repeat creation-only settings such as chunk, seed or replay mode. A checkpointed stop can also be requested with the output directory's `STOP` file.

## Schedule format

```text
SGCURRICULUM1
2000 "through-stage-1.dat" 1
4000 "through-stage-2.dat" 1
6000 "through-stage-3.dat" 0.5
```

Each row contains the cumulative **observed-chunk update count** at the end of a stage, a path relative to the schedule, and a multiplier on the base learning rate. Replay optimizer steps are counted separately and do not advance these thresholds. The generated example uses scale 1 for all three stages; the example above illustrates a lower rate in the last stage. No plasticity schedule has yet been established as optimal.

Every later corpus must begin with exactly the previous corpus bytes, followed by a document separator (`0x1e`) and at least one new valid document. Previous document boundaries and indices must be identical. The program checks every edition before training and binds their content hashes plus the schedule bytes into the checkpoint. Editing even a future source invalidates a resume, as does editing the schedule. Raw schedule line endings therefore matter. The current bounded schedule supports up to 4,096 stages and 100 million observed updates; it is not an open-ended online ingestion service.

At a transition the cursor starts at the first newly added document. Once the new material reaches EOF, the normal cursor wraps through the cumulative corpus. Earlier observed windows also remain eligible for reservoir replay. The replay reservoir is bounded, so it does not guarantee a particular quota for every past stage.

Version 2 of the **schedule text** makes that wrap policy explicit:

```text
SGCURRICULUM2
6000 "reading.dat" 1 all
12000 "reading-and-science.dat" 0.25 new
```

The fourth column is `all` (repeat all documents introduced so far) or `new` (repeat only documents added by this stage). On either setting a transition starts at the first newly added document. With `new`, earlier documents stay addressable for replay but do not silently reappear in the online observation stream when the new books reach EOF. This permits a controlled retention comparison. Old version-1 schedules retain their original `all` behavior.

The scope is bound into the existing live v4 checkpoint through the schedule hash. Resume reconstructs the allowed online document range from the same checked editions. Session and transition logs report `online_first_document` and `online_document_count`. The [retention experiment](retention.md) tests replay under this isolation.

The same GPU parameter and optimizer allocations continue through the transition. Recurrent state receives the ordinary new-document reset on the next observed chunk. If SI is enabled, the unfinished document's path is consolidated at the explicit stage boundary; completed documents are not consolidated twice. Stage changes can abandon the unread remainder of the current document, as expected for a fixed exposure budget.

## Saved development history

`SGCURRICULUM3` adds an explicit answer-emphasis multiplier after the repetition scope:

```text
SGCURRICULUM3
6000 "reading.dat" 1 all 1
18000 "reading-and-lessons.dat" 0.25 new 64
```

A multiplier of 1 preserves ordinary learning. A multiplier above 1 requires every document introduced at that stage to contain exactly one literal `\nAnswer: ` field with a nonempty suffix. Its suffix bytes, including any trailing newline, receive that multiplier; all preceding targets keep weight 1. The native loss and gradient are divided by the sum of target weights in each observed or replayed window. No target is fabricated by this mechanism. This is supervised emphasis of selected source answers, not autonomous teacher feedback or reward learning.

The original document's annotation persists when it is replayed or revisited during later stages. Preflight rejects missing or duplicate answer fields. The checked schedule binds the multiplier without a new checkpoint layout, so changing supervision requires a new declared stream. Resume reconstructs the same document ranges and weights before learning. Logs record the emphasized-document count and stage transitions record the new multiplier. Loss logged for an emphasized window is weighted cross-entropy; ordinary held-out book evaluation stays unweighted.

Normalization is local to each training window. If a document spans several chunks, a window consisting entirely of answer bytes has its ordinary mean gradient. The short selected lessons fit in one 128-byte chunk, so their context and answer compete under the declared weight. This policy does not lengthen the gradient horizon or guarantee successful learning. The [teaching experiment](language-probes.md) compares it with uniform weighting at the same observations and replay budget.

Live checkpoint extension v4 stores the schedule identity, current stage and base learning rate in addition to the full existing runtime. `stage-1.ckpt`, `stage-2.ckpt`, etc. archive the state at each completed boundary. `latest.ckpt` supports ordinary restarts, including an exact boundary: the next invocation performs the pending transition before its first new observation.

`metrics.jsonl` records transitions, source hashes, preserved replay count and learning rate. `session.json` records stage, source exposure, replay exposure, generated bytes and latency. Explicit resume `--lr` changes the base learning rate while retaining the current stage multiplier; it does not erase learned history.

An existing single-corpus live v2/v3 checkpoint can be attached with `--resume` if its corpus is the first scheduled edition and its observed-update count still fits stage 1. Legacy v1 checkpoints remain readable but cannot attach directly. Starting a new stream with `--checkpoint` initializes new replay/consolidation history; use `--resume` to preserve it. A curriculum started from a supplied checkpoint uses that checkpoint's saved learning rate unless `--lr` is given.

For a registered population member, use [`population-live`](evolution.md) with its ID and the same curriculum. It owns the checkpoint path, automatically resumes complete live state and preserves the member's birth time and lineage. Its `stage-N.ckpt` archives and logs live under `member/live/`, while the canonical checkpoint remains `member/latest.ckpt`. The population's lifespan rule can terminate an individual's eligibility for further learning even if its curriculum is incomplete.

Validation is isolated from training and speech state. Exact held-out documents included anywhere in the scheduled cumulative corpus are rejected before a run is written. This check and source-preparation paragraph deduplication do not detect paraphrases or all short overlaps. Test material should remain reserved for final comparisons.

## Verified boundaries

The native suite tests LIF and ALIF, with and without SI, with graph generation and replay active. Restarts before, at and after transitions preserve speech, replay RNG, state and optimizer within a declared numerical tolerance. CLI tests repeat these checks across separate native processes, reject changed future sources and exercise read-only inference from v4 checkpoints. These correctness results do not establish better retention, a curriculum advantage or useful conversation.
