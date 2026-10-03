# Arithmetic source notation review

The optional arithmetic converter now handles the additional notation found
in the pinned Prealgebra 2e source. The [audit](../reports/prealgebra-notation-audit.json)
authenticates 75 cached modules against their recorded raw SHA-256 and Git blob
identities, then inspects 22,212 formulas. It converts 22,205 nonempty formulas
and rejects seven empty formulas. Complete-module extraction succeeds for 57
of the 75 modules, compared with 33 using the original Astronomy subset.

This is a conversion stage. No Prealgebra edition has been admitted, no
curriculum has been created from it and no model has learned from it.

## Representation

The new `scripts/corpus/prealgebra_math.py` callback handles indexed roots,
over/underscripts, bold and italic styles, invisible alignment and the two
enclosures present in the reviewed source. The shared CNXML reader accepts
the callback explicitly; its default behavior is unchanged.

The mapping follows the [MathML presentation definitions](https://www.w3.org/TR/MathML3/chapter3.html).
Root indices remain separate from their radicands. Carry digits and other
annotations stay above or below their base. Horizontal bars become overlines
or underlines. Hidden alignment digits use `\phantom{...}` and are not
flattened into visible operands. Enclosures use the documented, non-standard
[MathJax TeX enclosure syntax](https://docs.mathjax.org/en/latest/input/tex/extensions/enclose.html).

| Source notation | Serialized example |
| --- | --- |
| Cube root of n | `\sqrt[3]{n}` |
| Repeating decimal digits | `0. \overline{25}` |
| Carry digit 1 above 4 | `\overset{1}{4}` |
| Division bracket over 12 after divisor 3 | `3 \enclose{longdiv}{12}` |
| Crossed-out factor | `\enclose{updiagonalstrike}{5}` |
| Hidden zero used for alignment | `\phantom{0}` |

This is text for later review, not a pixel-exact page reconstruction. Spacing
and visual layout are normalized. Conversion alone does not verify each
calculation or turn a source explanation into a question-answer dataset.

## Remaining source problems

Eighteen modules remain rejected. Their first extraction errors comprise
twelve invalid or overlapping table spans, five modules containing the seven
empty formulas, and one unsupported list structure. The report records every
module identity and rejection. Additional problems may appear after a first
error is resolved; this is not an exhaustive census of malformed CNXML.

For example, the multiplication table in module `m81255` declares ten columns
but supplies eleven cells per row. Some empty formulas may be layout-only
spacers, while others occur inside prose. They need source-specific review
before any correction or omission; the converter does not silently invent a
missing expression or adjust a declared table width.

The source is the official [OpenStax repository at commit bba5f524](https://github.com/openstax/osbooks-prealgebra-bundle/tree/bba5f5244066884797f730149d918e33f2beefd1).
The cached collection and license are authenticated with the same tree and
declare CC BY 4.0 for this pinned edition. This observation does not substitute
for final attribution, passage review, source-selection admission or protected
validation/test splits. It does not make a claim about later editions.

## Verification and scope

The focused test checks twelve explicit notation examples, rejects sixteen
malformed or unreviewed cases, and verifies that the default converter still
rejects all twelve extension examples. Every mutually extractable module
retains exactly the same output bytes. A separate
[Astronomy compatibility audit](../reports/prealgebra-astronomy-compatibility.json)
re-extracts all 184 admitted modules with their recorded hashes, checks 30
reviewed passages, and repeats the existing source/split and table checks.

```powershell
python tests/prealgebra_math.py --out runs/prealgebra-notation-audit.json
python tests/openstax_corpus.py --out runs/astronomy-compatibility-audit.json
```

The first command uses the previously cached official source survey and tree
under `runs/science-source-review`; use a fresh output path. Both commands run
on CPU. They do not alter source files, prepared training editions, checkpoint
state or any queued native experiment.
