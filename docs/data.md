# General foundation curriculum

The allowlist is [data/sources.json](../data/sources.json). Founders use only these selected texts, with random initialization and fixed byte token IDs. No external model supplies pretrained weights or hidden tokenizer training data.

The initial sources are historical elementary readers, ordered from simple vocabulary through connected stories. They provide a small reproducible starting point for testing development. They do not provide broad current knowledge, a conversation format, or age-specific human cognitive assessment.

## Outputs

- `stage-1.dat`, `stage-2.dat`, `stage-3.dat`: separate training stages, in declared source order.
- `through-stage-N.dat`: cumulative training documents through that stage, preserving all prior bytes and document indices.
- `curriculum.sg`: an editable example live schedule with 2,000 observed-chunk updates per stage and learning-rate scale 1. These counts are experimental exposure budgets, not learned mastery thresholds.
- `train.dat`: all selected training sources, useful for a pooled/shuffled curriculum control.
- `validation.dat`: one separate book, used for checkpoints and development comparisons.
- `test.dat`: one separate book, reserved for final evaluation.
- Individual cleaned source texts, the exact input `source-spec.json`, and `manifest.json`: exact edition URLs, catalogue/raw/clean hashes, selected boundaries, byte/word counts, duplicate removals and stage labels. The source-spec hash binds the saved input bytes, including line endings.

Stage numbers are curriculum labels. A future population registry will separately record developmental mastery, lifetime exposure and parent generation. Finishing a fixed number of steps is not a mastery test.

## Preparation and isolation

The downloader verifies selected titles and records catalogue metadata. The catalogue describes these editions as public domain in the USA; this is source metadata, not a new project-wide license for the books. Original downloaded notices are kept under `data/raw/`. The MIT license covers this repository's original code.

Gutenberg wrapper material, page markers and illustration markers are removed. Where the source manifest declares a first-lesson or final-body boundary, the cleaner requires it to exist before slicing the text. The cleaner rejects decoding corruption and the reserved document separator. No model-generated text is added.

Whole-book partitions avoid adjacent train/validation windows. Exact normalized paragraphs of at least 120 characters are reserved in the order test, validation, training; duplicates are removed from later partitions. Short passages, shared stories and paraphrases can still overlap. The validation and test books cover different reading levels, so their losses cannot be compared as equal-difficulty skill scores.

Preparation refuses a nonempty output directory. Use a new directory and versioned allowlist for a changed corpus. Cached raw downloads can be inspected with `--download-only`; the manifest hashes identify the exact cached edition. A fresh download after the publisher updates an edition may differ, so compare hashes when reproducing published evidence.

Raw books, prepared text and model checkpoints stay outside the Git tree. The preparation script recreates the selected curriculum from its declared sources. A compact manifest in `reports/` records the edition used for local validation.

## Extending the curriculum

Add selected sources with a reason, stage, split and appropriate provenance. Preserve independent evaluation material and use a new corpus version. More advanced stages can cover arithmetic, natural science, geography and everyday procedures, with independently authored/selected skill questions. Source diversity and correctness matter alongside total bytes.

Teacher models should select or annotate approved source examples with provenance. Their generated explanations, if later enabled, must be recorded separately and checked against external targets. Evaluation answers cannot be inherited from the same teachers being evaluated.

## Selected development extension

[sources-development-v2.json](../data/sources-development-v2.json) preserves the foundation selection and adds one training book and one separate evaluation book:

| Source | Role | Reason |
| --- | --- | --- |
| [The Fairy-Land of Science, Arabella B. Buckley](https://www.gutenberg.org/ebooks/5726) | Stage 4 training | Sustained explanatory prose on light, air, water, plants and insects gives a subject/style change after elementary reading. |
| [Home Geography for Primary Grades, C. C. Long](https://www.gutenberg.org/ebooks/12228) | Stage 4 validation only | A different author and book covers observations of surroundings, land, water and weather. It measures transfer to related prose. |

These are historical language sources. The science book includes obsolete explanations, religious framing, transcription artifacts and references to figures. It is not modern scientific ground truth. Neither byte prediction on these books nor a lower loss demonstrates factual understanding. The geography text is never a learning or replay target. The existing Beacon final test book remains reserved.

```powershell
python scripts/prepare_corpus.py --sources data/sources-development-v2.json --out data/prepared/development-v2-final
```

Preparation starts at the first lecture/lesson and removes the trailing ebook attribution. The verified edition has 324,154 additional training bytes and 84,318 additional held-out bytes. All original foundation training/stage and final-test bytes remain identical. The extension has 585,436 training bytes including separators. Individual cleaned held-out files support separate old/new-domain evaluation; their different difficulty levels must not be interpreted as directly comparable skill scores.

## Selected original teaching lessons

[lessons-relations-v1.json](../data/lessons-relations-v1.json) selects literal templates and a small vocabulary for 312 short location lessons. These original MIT-licensed lessons add 24,071 training bytes, generated deterministically by `scripts/prepare_lessons.py`; no pretrained parameters or parental output distributions are imported. They are a separate optional teaching experiment, not an expansion of the book allowlist by an uncontrolled scraper.

The [probe protocol](language-probes.md) specifies the split by ordered object pairs, source-derived answers, context-reversal tests and the important first-object limitation. Optional `--reading` retains the selected old training corpus byte for byte and builds two explicit replay curricula. Preparation rejects held-out probe contexts in that reading corpus after normalizing line endings. It does not detect paraphrases or all forms of semantic overlap. [Reproduction evidence](../reports/teaching-reproduction.json) checks all prepared bytes and both LF/CRLF contamination controls. The published [manifest](../reports/teaching-v1-manifest.json) identifies the combined 609,508-byte training edition and both supervision schedules. Neither the development probes nor the reserved test probes enter learning.
