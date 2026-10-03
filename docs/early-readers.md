# Selected early readers

This separate starter edition contains **16 short English stories**: ten training texts, three development texts and three reserved test texts. Training totals **14,201 UTF-8 bytes and 2,585 whitespace-delimited words**. It supplies simple actions, questions, daily routines and connected dialogue. The [native reading-adaptation study](reading-adaptation.md) now uses its training stories with separate development assessment. Its size is small; preparing or fitting it does not establish useful conversation.

The [completed live-rehearsal study](live-reading-results.md) also uses this edition with preserved recurrent and replay history. More replay partly reduces forgetting, but prolonged repetition still worsens unseen-story loss after early improvement. This provides evidence for testing content breadth and exposure limits before treating the starter edition as a sufficient general-language curriculum.

A separate [reading expansion](reading-breadth.md) now provides 34 additional training stories and eight evaluation works. It excludes every source in this starter edition and protects all earlier selected books. Preparation is verified; learning on the expansion has not yet been tested.

The publisher's [reading levels](https://storybookscanada.ca/about/faq/) describe increasing text length and complexity. Our first group uses level 1, the second level 2 and the third levels 3–4. These are content groups, not human ages or measured model mastery. Training groups contain 70, 554 and 1,961 words respectively. They need an explicit exposure budget and independent skill assessment before being used for stage promotion.

## Selection and provenance

The [allowlist](../data/sources-early-readers-v1.json) records every title, author/adapter/translator, source filename, role, selection reason, reading level, page count and SHA-256 identity. Text comes from the publisher-linked [Storybooks Canada source repository](https://github.com/global-asp/sbc-source), pinned to revision `be28dab38ac55e8eb2185edb6631b11615d02185`. Its [source page](https://storybookscanada.ca/about/source/) identifies that repository as the source text for the website.

| Role | Selected title | Publisher level |
| --- | --- | ---: |
| Train | [What are you doing?](https://storybookscanada.ca/stories/en/0008/) | 1 |
| Train | [I like to read!](https://storybookscanada.ca/stories/en/0087/) | 1 |
| Train | [A very tall man](https://storybookscanada.ca/stories/en/0001/) | 2 |
| Train | [Decision](https://storybookscanada.ca/stories/en/0027/) | 2 |
| Train | [Khalai talks to plants](https://storybookscanada.ca/stories/en/0089/) | 2 |
| Train | [Zama is great!](https://storybookscanada.ca/stories/en/0095/) | 2 |
| Train | [A Tiny Seed: The Story of Wangari Maathai](https://storybookscanada.ca/stories/en/0110/) | 3 |
| Train | [Sakima's song](https://storybookscanada.ca/stories/en/0315/) | 3 |
| Train | [Holidays with grandmother](https://storybookscanada.ca/stories/en/0243/) | 4 |
| Train | [Grandma's bananas](https://storybookscanada.ca/stories/en/0294/) | 4 |
| Development | [The hungry crocodile](https://storybookscanada.ca/stories/en/0156/) | 1 |
| Development | [Tom the banana seller](https://storybookscanada.ca/stories/en/0296/) | 2 |
| Development | [The day I left home for the city](https://storybookscanada.ca/stories/en/0324/) | 3 |
| Reserved test | [Counting animals](https://storybookscanada.ca/stories/en/0327/) | 1 |
| Reserved test | [Goat, Dog, and Cow](https://storybookscanada.ca/stories/en/0004/) | 2 |
| Reserved test | [What Vusi's sister said](https://storybookscanada.ca/stories/en/0291/) | 4 |

Each selected story has its own **CC BY 3.0 or CC BY 4.0** notice. The repository also contains noncommercial editions; the preparer accepts only the exact selected CC BY sources and checks the publisher's versioned license link. Project code remains MIT. Source texts retain their licenses and full credit in the [attribution record](../reports/early-readers-v1-attribution.md), raw cached editions and prepared `ATTRIBUTION.md`. Author, illustrator, translator and adapter credit is preserved even though images and audio are not acquired. No source creator or publisher endorsement is implied. The [3.0](https://creativecommons.org/licenses/by/3.0/) and [4.0](https://creativecommons.org/licenses/by/4.0/) license pages describe attribution and reuse terms.

The [manifest](../reports/early-readers-v1-manifest.json) records the acquired edition identities. Markdown titles, page separators and credit blocks are kept outside model text; complete unedited page bodies are joined with blank lines. The publisher's English body is checked page by page against the source after whitespace, quotation-mark and ellipsis normalization **for comparison only**. The prepared text keeps the source's punctuation. Raw and publisher snapshot hashes are pinned; a changed source or web snapshot requires review as a new edition instead of silent replacement.

## Preparation and validation

`scripts/corpus/storybooks.py` owns this publisher's format and edition validation. `scripts/prepare_early_readers.py` owns the selected splits, protected evaluation material, cumulative groups and attribution output. It reuses the existing HTTP fetch helper. The Gutenberg preparer and native model code are unchanged.

The [completed audit](../reports/early-readers-v1-verification.json) reproduces all **28 output files byte for byte**, including source identities and attribution. An independent extraction checks every prepared page against the pinned Markdown. All training groups preserve complete earlier documents in their cumulative files. Exact paragraph and 20-word sequence checks find no overlap with the six new held-out stories or five existing evaluation books. The 288 reserved/development binding contexts are also absent. These checks do not detect paraphrases or shorter common phrases.

Ten negative controls reject changed source text, the wrong title, a noncommercial edition, the wrong language, missing pages, duplicate credits, changed publisher licensing, changed published text, an existing output destination and injected held-out training content. The last control exercises corpus assembly after source validation; it does not alter the real source files.

```powershell
# Prepare the earlier selected corpus first; its five evaluation books are protected.
python scripts/prepare_corpus.py --sources data/sources-stories-v1.json --out data/prepared/stories-v1

python scripts/prepare_early_readers.py
python tests/early_reader_corpus.py --out runs/early-reader-corpus-check
```

Use fresh output directories; skip the first command when that exact earlier corpus already exists. The preparer emits individual stories, split files, per-group training files and cumulative training editions. It supplies no automatic learning schedule: 70 words in the first group cannot justify an arbitrary age or update count. A later experiment can use the existing append-only curriculum admission with declared exposure and retention checks.

This preparation stage performs no model training and scores no reserved test. These sources are separate from the fixed frozen-self retention study. Some story meaning relies on illustrations, which the model does not receive; fictional events and simplified biographies are not independently verified factual answer keys. Whole-story holdouts still share vocabulary, constructions and some authors with training. The earlier seven-book narrative corpus remains available for broader text exposure.
