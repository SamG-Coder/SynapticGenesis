# Resident conversation and live correction

The scripted correction experiment loads a checkpoint for each question and
learning round. The resident candidate loads it once, keeps one learned
parameter/optimizer workspace on the GPU, and accepts questions and learning
commands in the same process. This removes repeated model loading from the
question path. Speed and numerical equivalence still require GPU validation.

`experiments/conversation_session.cuh` calls the existing `LiveEngine`,
`live_tick`, curriculum extension and `GraphDecoder`. The question view shares
weights and optimizer storage with the learner and owns its transient recurrent
state. Each question starts with fresh context, matching the controlled
correction checks. A learning command advances through already admitted source
material with the original replay and periodic speech policy. Generated answers
are never silently converted into training targets.

The candidate supports ordinary grouped-replay checkpoints and their membrane
policy wrapper. It currently rejects teacher bundles and SI consolidation. This
is a diagnostic conversation transport over selected material; it does not yet
admit arbitrary new typed corrections or preserve a multi-turn chat context.

## Commands

Build without launching model computation:

```powershell
.\build.ps1 -ResidentConversation
.\build\resident-conversation\synaptic-conversation-protocol-test.exe
```

The native executable accepts `run --resume <checkpoint> --curriculum <schedule>
--out <fresh-directory> --script <session-file>`. To introduce reviewed correction
material, also pass its append-only `--extend-curriculum <schedule>`. The original
periodic speech prompt must match the checkpoint; `--prompt` defaults to
`The bird `, including its trailing space. Question generation separately uses
`--answer-bytes`, `--answer-top-k`, `--answer-temperature` and `--answer-seed`.

For the complete 105M prose model and the prepared six-correction edition, a
session file has this form:

```text
SGCONVERSATION1
ask before "What is two plus three?"
learn 216295
save first-exposure
ask after "What is two plus three?"
ask reworded "What does two plus three equal?"
quit
```

`learn` takes an absolute source-update count within the admitted schedule. Here
216295 means six updates after the existing 216289-update prose checkpoint; it
does not mean six updates for an arbitrary checkpoint. The original native
curriculum and state checks still apply.

`--script -` accepts this protocol through stdin. After sending the header, the
caller waits for `ready`, sends a question, reads its answer, and can send a
learning command while the same process remains alive. A malformed ordinary
command is rejected without discarding that process. Explicit saves and a
normal `quit`/EOF write checkpoints; an abrupt process termination may lose work
since the last save. File scripts are fully checked before allocating the model.

Commands, byte-exact answers, update events and timing are saved in the output
directory. JSON strings escape each raw byte as U+00xx, following the existing
native probe convention. The `.bin` files retain the original output bytes.
Question timing separates graph capture, prompt prefill and resident decode.
Decode wall time includes sampling and synchronization; it is not GPU-kernel-only
time. No speedup is claimed from comparing timers with different boundaries.

## Acceptance

The [build and parser evidence](../reports/resident-conversation-build.json)
records a successful CUDA/C++ build and 31 CPU-only protocol rejections. The
[host driver checks](../reports/resident-conversation-host-checks.json) cover raw
byte preservation and failure/reuse/incomplete predecessor handling. Neither
report establishes native numerical equivalence or language quality.

The queued GPU acceptance first runs four small associative fixtures: ordinary
and membrane cost 0.001, each with FP32 and TF32 learning. It requires complete
checkpoint equality against the preserved runtime before and after questions,
at an intermediate learning endpoint, after uninterrupted continuation and
after restart. It also requires byte-identical sampled answers and a real stdin
exchange that waits for each reply before sending the next turn.

If those pass, one resident process repeats the complete 105M six-correction
experiment. All 72 answers and all three corrected checkpoints must match the
existing separately loaded runs exactly. The comparison reports resident
decode and learning timing, while retaining the original language-quality
results. Matching poor answers would establish execution equivalence only.

`scripts/resident_conversation_acceptance.py` waits on the actual correction
driver process and checks its complete result before launching CUDA. Existing
studies keep their original executables and pinned inputs. The candidate remains
on the research branch until its GPU acceptance is available.

The [launch snapshot](../reports/resident-conversation-launch.json) confirms that
the acceptance driver is live and waiting for the correction process, with zero
native commands started at that check. The protocol SHA-256 is
`8bbb09eb45663e06769677e7c1761c1eb4fcee18dfd180d1515d177144c19a25`;
the declared source commit is `752355c`.
