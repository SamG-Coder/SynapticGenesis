# Selected stories for general language learning

The existing development books contain 585,436 training bytes, and the binding lessons repeat a narrow answer format. Generated samples in the architecture screen remain repetitive lesson fragments. This separate selection adds **1,744,439 training bytes**, approximately **321,847 whitespace-delimited words**, from seven authored books. It is prepared for a later general-language experiment; it is not part of the running associative comparison.

The [allowlist](../data/sources-stories-v1.json) fixes the titles, edition IDs, training/evaluation roles, selection reasons and required text boundaries. The [edition manifest](../reports/stories-v1-manifest.json) records catalogue URLs, raw and cleaned hashes, source counts and cumulative stage hashes. The Gutenberg catalogue identifies the selected editions as public domain in the USA. Cached original notices are retained; the repository's MIT code license does not relicense books.

## Selected material

| Training source | Content group | Prepared words |
| --- | --- | ---: |
| [The Great Big Treasury of Beatrix Potter](https://www.gutenberg.org/ebooks/572), Beatrix Potter | Short narrative collections | 28,161 |
| [Mother Goose in Prose](https://www.gutenberg.org/ebooks/5312), L. Frank Baum | Short narrative collections | 44,159 |
| [The Wonderful Wizard of Oz](https://www.gutenberg.org/ebooks/43936), L. Frank Baum | Sustained adventures | 39,296 |
| [Alice's Adventures in Wonderland](https://www.gutenberg.org/ebooks/11), Lewis Carroll | Sustained adventures | 26,436 |
| [The Jungle Book](https://www.gutenberg.org/ebooks/236), Rudyard Kipling | Longer stories | 50,729 |
| [Five Children and It](https://www.gutenberg.org/ebooks/17314), E. Nesbit | Longer stories | 52,464 |
| [The Secret Garden](https://www.gutenberg.org/ebooks/17396), Frances Hodgson Burnett | Longer stories | 80,602 |

[The Velveteen Rabbit](https://www.gutenberg.org/ebooks/11757), by Margery Williams Bianco, supplies 20,687 bytes for development assessment. [The Happy Prince, and Other Tales](https://www.gutenberg.org/ebooks/902), by Oscar Wilde, supplies 88,492 bytes reserved for final evaluation. Neither supplies learning or replay targets.

The three existing evaluation books are also included in the reservation pass: New National First Reader, Home Geography for Primary Grades and The Beacon Second Reader. Their cleaned bytes remain identical. Reserving their paragraphs before cleaning new training stories protects the earlier evaluation material. The aggregate validation file therefore contains three books and the aggregate test file two; a future experiment should report each relevant book separately because their difficulty and length differ.

## Preparation and evidence

The existing corpus preparation module handles the new allowlist. Required start/end boundaries remove front matter, contents pages, publication credits and trailing transcription notes. It now also removes `[Picture: ...]` annotations. Whole books remain distinct documents separated by byte `0x1e`; the native runtime excludes that separator from sampled windows. No pretrained weights, learned tokenizer or generated story corpus is introduced.

The [verification report](../reports/stories-v1-verification.json) confirms:

- All 22 story-corpus text/data/schedule files reproduce byte-for-byte from the cached selected editions.
- All 20 corresponding files in the prior development corpus reproduce unchanged with the updated cleaner.
- Seven training books, three validation books and two reserved test books have the declared roles; cumulative training stages preserve complete earlier documents.
- There is no training overlap with held-out normalized paragraphs of at least 120 characters, and none of the 288 distinct held-out binding contexts occurs in the new training text.
- A deliberately inserted held-out context is detected. Missing Gutenberg wrappers, missing selected body boundaries and truncated bodies are rejected.

These checks cover exact text conditions. Shared tales, paraphrases and shorter overlap can remain. One duplicate paragraph was removed from Mother Goose in Prose. Narrative references to illustrations, dialect, verse, dated social assumptions and transcription quirks remain. The Velveteen Rabbit plain-text edition begins `HERE` after a decorative initial in its HTML edition; that source transcription is recorded and retained. The books are fiction, not current factual answer keys.

The content groups are organizational labels. They do not establish human ages, mastery or a benefit from ordering. [TinyStories research](https://arxiv.org/abs/2305.07759) demonstrates coherent simplified-story generation by small Transformers in its own synthetic-data setting. It motivates measuring text difficulty and diversity; it does not establish that these authored books or this spiking architecture will produce the same result.

## Reproduce and use

```powershell
python scripts/prepare_corpus.py --sources data/sources-stories-v1.json --download-only
python scripts/prepare_corpus.py --sources data/sources-stories-v1.json --out data/prepared/stories-v1
python tests/story_corpus.py --out runs/story-corpus-check
```

Use a fresh prepared output directory. Existing cached editions are reused and authenticated in the manifest; a future publisher revision can yield different hashes. `stage-1.dat`, `stage-2.dat` and `stage-3.dat` contain 391,211, 358,125 and 995,101 bytes respectively. `train.dat` pools their seven books, and cumulative `through-stage-N.dat` files preserve earlier content. The generated 2,000-observation-per-stage schedule is a preparation example, not a validated exposure budget.

An existing live individual can receive these selected sources through the [append-only curriculum mechanism](live-curriculum.md). The [declared narrative continuation](narrative-learning.md) preserves all nine final models from the associative comparison and measures old-reader/skill retention alongside new narrative loss and actual generated samples. It records the additional stage's replay effects and retains the old editions. Corpus acquisition alone does not demonstrate a language-learning improvement. No model was trained or reserved test scored in this preparation stage.
