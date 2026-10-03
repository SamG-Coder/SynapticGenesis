# Selected reading expansion

This edition adds **34 training stories and 9,827 words**, with four whole works
for development and four reserved for testing. It follows the
[live-rehearsal comparison](live-reading-results.md), where additional replay
reduced some forgetting but prolonged exposure to ten repeated stories still
overfit. The expansion is prepared and audited. The
[declared native comparison](reading-breadth-experiment.md) now has a completed
execution smoke run; its three-seed learning comparison remains pending. Adding
material has not established better answers, durable retention or a useful
developmental schedule.

| Role | Whole works | Words | UTF-8 bytes, including document separators | Within-document next-byte targets |
| --- | ---: | ---: | ---: | ---: |
| Training | 34 | 9,827 | 53,073 | 53,006 |
| Development | 4 | 1,720 | 8,994 | 8,987 |
| Reserved test | 4 | 1,658 | 8,941 | 8,934 |

The new training selection is about 3.8 times the word count of the
[early-reader starter](early-readers.md). Together those two selections contain
44 training stories and 12,412 words. They remain separate immutable editions;
an experiment must explicitly declare whether and how to combine them. The
earlier seven selected narrative books remain available too. This is a modest
increase in short reading material, not a broad general-knowledge corpus.

## Source selection and attribution

Texts come from [Global ASP's African Storybook source repository](https://github.com/global-asp/asp-source),
pinned to `b5c3d5b2266ad936e5f20fa878e839b32b02b4ba`. The repository describes
its Markdown as minimally edited extractions of African Storybook PDFs.
The [pinned English index](https://github.com/global-asp/asp-source/blob/b5c3d5b2266ad936e5f20fa878e839b32b02b4ba/en/README.md)
supplies each source's original reference and versioned license link. Only the
explicit [allowlist](../data/sources-reading-breadth-v1.json) is downloaded by the
preparer. Each selected source has agreeing CC BY labels in its metadata and
index, a CC BY 3.0 or 4.0 link, and exact raw/body hashes. Source/index agreement
does not independently verify the current publisher page or original PDF.

Sixty candidates were considered and all 42 selected bodies were read in full.
Eighteen candidates were excluded. Four had noncommercial index labels and
were not downloaded. Two others, `0047` and `0190`, had source/index license
conflicts. Remaining exclusions include picture-dependent fragments, a related
compressed market story, unsuitable transcription details and one worked time
example that states the wrong interval from 10:45 to 13:00. The allowlist records
every exclusion, not just the accepted titles. An additional conflict in `0044`
was found during the earlier source review, outside the sixty-candidate batch.

Training includes short household and school routines, connected market and
family stories, animal folktales, and longer narratives such as *Together We're
Strong*, *Refiloe and the washed chickens* and *Sizwe's Smile*. Original names
and credits are preserved. The three content groups are editorial organization,
not publisher reading levels, human ages or measured model mastery.

Development holds out *Father and son*, *The Race*, *Dima and Owl* and *Big blue
bus*. The final test reserves *Girl called Norah*, *The Big Juicy Mango*,
*Amazing Daisy* and *The tree that saved the village of Ombalantu*. The two
Anansi stories and the two related travel descriptions are kept within training.
Related rendition `0057` is excluded alongside selected training story `0019`.

The [complete attribution](../reports/reading-breadth-v1-attribution.md) retains
author, illustrator, translator and adapter fields, immutable source links,
original references, individual licenses and review notes. MIT covers original
project code; these texts retain their own licenses. Images and audio are not
acquired, and no publisher or creator endorsement is implied.

## Preparation and verification

The preparer removes Markdown titles, page markers, credits and three explicitly
pinned editorial footers following the credits. Those footers remain in the
attribution record. Otherwise page bodies retain original spelling, punctuation,
internal line breaks and notes, joined by blank lines. UTF-8 decoding is explicit
on Windows so accented credits survive. Document separator byte `0x1e` marks
whole-story boundaries and is excluded from within-document target counts.

The [manifest](../reports/reading-breadth-v1-manifest.json) records source and
output identities. The [audit](../reports/reading-breadth-v1-verification.json)
reproduces **54 output files byte for byte** and independently extracts all page
bodies. It authenticates all **28 earlier selected sources**, including **11
earlier evaluation works**, before allowing the new edition. No new selected
work overlaps another selected or protected source by normalized whole text,
20 consecutive words, or a normalized paragraph of at least 120 characters.
The 288 existing development/test binding contexts are absent from training.
These checks do not detect all paraphrases or shorter shared phrases.

Twenty negative controls exercise changed source/index bytes, duplicate
identities, inconsistent or unsupported licenses, altered boundaries, a floating
revision, a path escape, a changed editorial footer, output reuse, protected
source mutation and actual contamination at the assembly boundary. The latter
includes an earlier held-out story, a new reserved story, duplicate new training
text and a case/punctuation-normalized protected sequence. Rejected assemblies
write no prepared output directory.

The shared page parser gained optional, explicitly pinned editorial-footer
handling. Its [regression audit](../reports/reading-breadth-early-reader-regression.json)
reproduces all **28 earlier starter files unchanged** and passes all ten earlier
negative controls. Native C++/CUDA code and learned checkpoints are unchanged
by this preparation stage.

```powershell
# Requires the exact earlier stories-v1 and early-readers-v1 prepared editions.
# See docs/early-readers.md if these have not already been prepared.
python scripts/prepare_reading_breadth.py
python tests/reading_breadth_corpus.py --out runs/reading-breadth-check-new
python tests/early_reader_corpus.py --out runs/reading-breadth-regression-new
```

Use fresh output directories. The scripts use only Python's standard library
for preparation and audit; model computation remains in the shared native
learner. The edition supplies individual stories, three split files, three
training groups, cumulative group files, a source specification, attribution
and manifest. It supplies no automatic learning schedule.

## Limits and next comparison

The sources mix fiction, figurative language, simplified explanations and a
historical account. Source-specific notes flag these boundaries. They are not
independently checked factual answer keys, arithmetic supervision or instruction
tuning. Some grammar defects and layout line breaks remain. *Dima and Owl*
retains its original story notes, so its development text includes editorial
prose. Whole-work holdouts still share vocabulary, constructions and some
creators with training.

The [declared comparison](reading-breadth-experiment.md) fixes one and two broader
passes and compares the starter and broader selections with the existing replay
controls, shared ancestors, earlier-skill measurements and actual generated
examples. Equal passes, equal new-source targets and equal optimizer updates are
different comparisons. Preparation and smoke verification do not establish a
stopping rule or justify changing age, growth or reproduction policy.
