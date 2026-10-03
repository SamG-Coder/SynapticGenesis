# A larger selected reading edition

The new `selected-prose-scale-v1` edition contains **55 training books,
4,817,047 word-like units and 26,896,416 UTF-8 bytes**. Four validation books
contain 166,908 word-like units and three reserved test books contain 253,171.
Training text is about 11.1 times the approximately 434,205 words of authored
book text in the earlier four-stage stream. These are word occurrences in
source prose, not vocabulary size, repeated training exposures or independent
semantic facts. No templated binding lessons are included in this edition.

The selection retains the admitted elementary readers, science reader and
seven narrative books, and adds 43 training titles. These include Treasure
Island, The Adventures of Sherlock Holmes, The Wind in the Willows, Black
Beauty, several Jules Verne novels and longer novels with sustained dialogue.
Far from the Madding Crowd is a new validation book; Moby Dick is newly reserved
for testing. Their authors do not occur among this edition's training books.
All five earlier held-out books retain their exact prepared bytes.

The [source list](../data/sources-prose-scale-v1.json) identifies every title,
author, split and stage. Original editions, notices and catalogue pages remain
in the local raw cache. Project Gutenberg's catalogue marks these editions
public domain in the USA; its source terms are distinct from this repository's
MIT code license. The source list is admitted by hash in the selection
registry, and every raw book is pinned to the bytes reviewed. Preparation now
rejects changed pinned source bytes before producing training text.

## Review and preparation

Forty-nine new candidates were inspected through their catalogue and three
deterministic substantial-paragraph samples at approximately 3%, 50% and 94%
of each body. The first 650 normalized characters of each sample were read.
This is a sample review, not a claim to have read every page. Four candidates
were deferred: a different essay collection, a Latin text, Middle English
verse and an edition requiring separate review of a later introduction.
The [review record](../reports/prose-scale-source-review.json) preserves the
observed passages, edition hashes and decisions.

Preparation removes Gutenberg wrappers, specified illustration/page markers
and exact repeated substantial paragraphs. It reserves held-out paragraphs
before processing training books. Some tables of contents, editorial text,
historical spelling and dialect remain. Exact paragraph protection does not
detect paraphrases, short overlap or common plots.

The [audit](../reports/prose-scale-corpus.json) authenticates all 62 source
editions and prepared outputs, confirms unchanged earlier holdouts, verifies
zero exact protected-paragraph overlap with training, and exercises rejection
of an altered reviewed source. Word-like units use the expression
`[A-Za-z]+(?:['-][A-Za-z]+)*`; they differ from byte tokens and whitespace counts.

This is a larger **reading and language** corpus. Historical fiction is not
a modern knowledge base or instruction-following dataset. Later editions
still need factual explanations, arithmetic, practical science and appropriate
question/answer material. Increasing text volume alone does not demonstrate
reasoning, reliable factual answers or a developmental curriculum benefit.

## Complete exposure schedule

The separate schedule counts all within-document next-byte pairs at chunk
128, including short final windows. It gives elementary reading four passes
and each later group one complete pass. Source traversal enters new books at
each stage; ordinary balanced replay retains earlier observations.

| Stage | New training books | Passes | End observation |
|---|---:|---:|---:|
| Elementary reading | 4 | 4 | 8,176 |
| Shorter narratives | 10 | 1 | 26,110 |
| Sustained narratives and explanations | 20 | 1 | 95,207 |
| Longer novels | 21 | 1 | 216,289 |

The complete initial schedule supplies 27,680,129 source target-byte
exposures, plus separately counted replay. It is an exposure plan rather than
an optimized stopping rule. Its full completion and quality need to be
reported from actual checkpoints; preparation does not mean the model has
already learned these books.

```powershell
python scripts/prepare_corpus.py --sources data/sources-prose-scale-v1.json --raw data/raw/gutenberg-scale-v1 --out data/prepared/prose-scale-v1-pinned
python scripts/prose_scale_schedule.py --prepared data/prepared/prose-scale-v1-pinned --out runs/prose-scale-curriculum
python tests/prose_scale_corpus.py --out reports/prose-scale-corpus.json
```

The schedule supports the measured C1024/H4096/L8 associative model with
104,851,472 parameters and 32,768 spiking neurons. Founders still start from
random weights. Larger models remain subject to the same held-out assessment,
retention and reproduction criteria; size alone does not qualify a parent.
