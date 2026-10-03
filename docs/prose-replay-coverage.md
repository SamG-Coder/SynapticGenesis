# What the larger learner actually rehearses

The completed 2M and 105M founders use the same 1,024-slot, stage-balanced
replay policy. Reconstructing their final source stage found **22,751 replay
updates on 755 distinct earlier-source windows**. Those windows contain
96,581 target-byte positions out of 11,399,051 available in the earlier books,
or approximately **0.85%**. Most of that rehearsal therefore revisits the same
small subset of the earlier text.

This is a coverage measurement, not a finding that the model forgot the other
99.15% or a proof that replay size caused the [observed regression](prose-105m-complete.md).
The original source stream learned the full earlier books, and their vocabulary
and patterns can generalize beyond stored replay positions. Distinctness here
means a document/offset/length descriptor, not a semantic fact or a unique string.

The [full evidence](../reports/prose-replay-coverage.json) reconstructs every
final-stage source observation and all 30,271 scheduled replay selections from
the stage-three checkpoint. The selections come from the deterministic host
policy reference, not a native per-event trace. Its complete final replay
payload, source cursor, source/replay counters and random-number states match
the native stage-four checkpoint exactly. Every metadata word except the
learned-payload checksum matches. Both model sizes produce the same coverage.

| Replayed source stage | Available unique source windows | Replay updates during stage four | Distinct windows replayed | Window coverage |
| --- | ---: | ---: | ---: | ---: |
| 1: foundations | 2,044 | 7,607 | 243 | 11.89% |
| 2 | 17,934 | 7,654 | 256 | 1.43% |
| 3 | 69,097 | 7,490 | 256 | 0.37% |
| 4: current material | 121,082 | 7,520 | 1,045 | 0.86% |

The first three replay pools stop receiving new source windows once stage four
starts. Their selected windows are each replayed approximately 29–31 times on
average; one foundation window is replayed 76 times. The first stage contains
duplicate positions because its four source passes enter the reservoir as
separate observations. Its final 256 slots represent 243 distinct positions.

The current-stage pool changes as new material arrives, so its 1,045 distinct
replayed windows exceed the 256 slots present at the end. All 34 earlier books
receive some final-stage replay. Of the 21 newly introduced books, 20 receive
replay during that stage. Book 46, *A Christmas Carol*, receives its complete
source pass but no reconstructed replay selection in this interval. Absence
from the final reservoir alone would not establish that fact; the full interval
reconstruction is needed.

## Why this matters for the next change

Increasing model size left the number of replay descriptors unchanged. The
runtime has access to the complete admitted source library, but its ordinary
replay selection samples only from the stored descriptors. More neurons do not
automatically create broader rehearsal.

A larger-reservoir control is therefore a concrete, relatively simple next
candidate after the current rate and neuron studies. The existing native
founder command already accepts a replay capacity up to 65,536. Comparing a
fresh 1,024-slot founder with a fresh 16,384-slot founder would change coverage
without adding learned scoring passes or increasing the nominal replay cadence.
This is a proposed experiment, not an additional queued run or a demonstrated
improvement.

At four source groups, descriptor/policy storage is `8 * (17 + 5*4 + 3*capacity)`:
24,872 bytes for 1,024 slots and 393,512 bytes for 16,384 slots, an increase of
360 KiB. These are host policy/checkpoint bytes. They do not include the source
library, neural weights, CUDA scratch or process overhead, and they are not a
measured GPU-memory result. Sampling, admission, serialization and runtime cost
still need measurement.

The comparison must keep source exposure, initial model policy and nominal
replay cadence fixed and report actual per-stage counts and target bytes. A
larger reservoir changes random-number consumption and selected examples, so
equal seed does not guarantee identical stage counts or replay targets. Both
retention and new-source learning need assessment. The existing failed
[two-choice replay experiment](replay-selection.md) also shows why better-looking
selection statistics cannot substitute for quality and cost measurements.

An already trained checkpoint cannot recover previously discarded online
experiences merely by allocating more slots. Here the books remain externally
available, so rebuilding a pool from them would be a distinct intervention
with its own provenance and exposure contract. This audit neither resizes a
live pool nor changes the active native runs.

## Verification and implementation

The audit authenticates the source edition, reconstructs all 55 native document
positions from actual curriculum bytes, and binds its input checkpoints to the
original native assessments. The document order is grouped by curriculum
stage; it must not be inferred from the source manifest's flat order alone.
It checks every reconstructed descriptor's document, chunk alignment and short
tail length, and compares replay event totals with the native per-stage deltas.

The shared host reference now accepts an optional replay observer. Running it
with and without that observer reaches the same complete native endpoint for
both sizes. `scripts/experiment_checkpoint.py` adds a bounded ordinary-policy
reader so these policy-only checks can skip the large learned arrays. Existing
callers needing weights retain the original reader. The native model loader
remains responsible for full payload validation.

The [reader/accounting checks](../reports/prose-replay-coverage-validation.json)
match the bounded reader against the earlier full reader on two native 2M
checkpoints, verify duplicate and one-byte-tail accounting, and reject seven
malformed cases. All checkpoint hashes remain unchanged. This stage performs
no CUDA forward, gradient update, alternate replay-policy training or reserved
test scoring.
