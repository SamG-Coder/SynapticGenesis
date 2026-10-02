# General foundation curriculum

The allowlist is [data/sources.json](../data/sources.json). Founders use only these selected texts, with random initialization and fixed byte token IDs. No external model supplies pretrained weights or hidden tokenizer training data.

The initial sources are historical elementary readers, ordered from simple vocabulary through connected stories. They provide a small reproducible starting point for testing development. They do not provide broad current knowledge, a conversation format, or age-specific human cognitive assessment.

## Outputs

- `stage-1.dat`, `stage-2.dat`, `stage-3.dat`: separate training stages, in declared source order.
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
