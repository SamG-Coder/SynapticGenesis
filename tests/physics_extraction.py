"""Physics notation and structural omissions must not corrupt source meaning."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from corpus.openstax import math_text, MATH
from corpus.physics import extract_physics, physics_extension
import xml.etree.ElementTree as ET


def formula(body):
    return math_text(ET.fromstring(f'<math xmlns="{MATH[1:-1]}">{body}</math>'), physics_extension)


def document(body):
    return ('<document xmlns="http://cnx.rice.edu/cnxml" xmlns:m="http://www.w3.org/1998/Math/MathML">'
            '<title>Physics extraction fixture</title><content>' + body + '</content></document>').encode()


def check():
    valid = {
        '<mstyle mathvariant="bold" mathsize="normal"><mi>v</mi></mstyle>': r'\boldsymbol{v}',
        '<mstyle mathvariant="bold"><mi>ω</mi></mstyle>': r'\boldsymbol{ω}',
        '<mstyle mathsize="normal"><mi>x</mi><mo>+</mo><mn>2</mn></mstyle>': '{x + 2}',
        '<mstyle scriptlevel="+1"><mi>a</mi></mstyle>': '{a}',
        '<mmultiscripts><mi>U</mi><mprescripts/><mn>92</mn><mn>235</mn></mmultiscripts>': '{}_{92}^{235}{U}',
        '<mmultiscripts><mtext>H</mtext><mprescripts/><none/><mn>4</mn></mmultiscripts>': r'{}^{4}{\text{H}}',
        '<mmultiscripts><mi>X</mi><mprescripts/><mi>Z</mi><mi>A</mi></mmultiscripts>': '{}_{Z}^{A}{X}',
        '<mmultiscripts><mi>Y</mi><mprescripts/><mrow><mi>Z</mi><mo>+</mo><mn>1</mn></mrow><mi>A</mi></mmultiscripts>': '{}_{Z + 1}^{A}{Y}',
        '<mmultiscripts><mi>X</mi><mprescripts/><mn>7</mn><none/></mmultiscripts>': '{}_{7}{X}',
        '<mover accent="true"><mi>v</mi><mo>¯</mo></mover>': r'\overline{v}',
    }
    for source, expected in valid.items():
        assert formula(source) == expected, (source, formula(source), expected)
    rejected = 0
    bad = [
        '<mstyle mathvariant="bold-italic"><mi>x</mi></mstyle>',
        '<mstyle mathvariant="bold" mathsize="large"><mi>x</mi></mstyle>',
        '<mstyle scriptlevel="+2"><mi>x</mi></mstyle>',
        '<mstyle mathvariant="bold" dir="rtl"><mi>x</mi></mstyle>',
        '<mstyle mathvariant="bold"/>',
        '<mstyle mathvariant="bold">lost<mi>x</mi></mstyle>',
        '<mstyle mathvariant="bold"><mi>x</mi>lost</mstyle>',
        '<mmultiscripts><mi>U</mi><mprescripts/><mn>92</mn></mmultiscripts>',
        '<mmultiscripts><mi>U</mi><mn>92</mn><mn>235</mn></mmultiscripts>',
        '<mmultiscripts><mi>U</mi><mprescripts/><mn>92</mn><mn>235</mn><mn>1</mn><mn>2</mn></mmultiscripts>',
        '<mmultiscripts><mi>U</mi><mprescripts/><mprescripts/><mn>235</mn></mmultiscripts>',
        '<mmultiscripts><mi>U</mi><mprescripts bad="yes"/><mn>92</mn><mn>235</mn></mmultiscripts>',
        '<mmultiscripts><mi>U</mi><mprescripts>lost</mprescripts><mn>92</mn><mn>235</mn></mmultiscripts>',
        '<mmultiscripts><mi>U</mi><mprescripts/><none bad="yes"/><mn>235</mn></mmultiscripts>',
        '<mmultiscripts><mi>U</mi><mprescripts/><none>lost</none><mn>235</mn></mmultiscripts>',
        '<mmultiscripts><mi>U</mi><mprescripts/><none><mn>1</mn></none><mn>235</mn></mmultiscripts>',
        '<mmultiscripts><mi>U</mi><mprescripts/><none/><none/></mmultiscripts>',
        '<mmultiscripts><mrow/><mprescripts/><mn>92</mn><mn>235</mn></mmultiscripts>',
        '<mmultiscripts><mi>U</mi><mprescripts/><mrow/><mn>235</mn></mmultiscripts>',
        '<mmultiscripts dir="rtl"><mi>U</mi><mprescripts/><mn>92</mn><mn>235</mn></mmultiscripts>',
        '<none/>', '<mprescripts/>',
    ]
    for source in bad:
        try: formula(source)
        except ValueError: rejected += 1
        else: raise AssertionError(('Accepted invalid/unreviewed notation', source))
    title_list = '<list id="steps" list-type="enumerated" start-value="2"><title>Motion steps</title><item>First source item.</item><item>Second source item.</item></list>'
    notes = '<note class="os-teacher extra"><para>OMITTED_TEACHER</para></note>Retained tail sentence.'
    grid = '<table class="section-key-terms"><tgroup cols="1"><tbody><row><entry>OMITTED_GRID_A</entry><entry>OMITTED_GRID_B</entry></row></tbody></tgroup></table>'
    text, stats = extract_physics(document('<para>Retained leading explanation.</para>' + notes + title_list + grid +
                                        '<para>Refer to <link target-id="steps"/> for details.</para>'))
    decoded = text.decode()
    for token in ('Retained leading explanation.', 'Retained tail sentence.', 'Motion steps',
                  '2. First source item.', '3. Second source item.', 'Refer to Motion steps for details.'):
        assert token in decoded, (token, decoded)
    assert 'OMITTED' not in decoded
    assert stats['policy']['titled_lists_preserved'] == 1 and stats['policy']['omitted_vocabulary_grids'] == 1
    external = '<link class="os-embed" url="#ost/api/ex/k12phys-ch01-ex008"/>'
    text, stats = extract_physics(document('<para>Retained explanation before a remote exercise.</para>' + external +
        'Retained text after a remote exercise.<note class="virtual-physics"><para>OMITTED_SIMULATION</para></note>'))
    assert b'the referenced section' not in text and b'OMITTED_SIMULATION' not in text
    assert b'Retained text after a remote exercise.' in text
    assert stats['policy']['omitted_external_exercise_placeholders'] == 1
    assert stats['policy']['omitted_note_virtual-physics'] == 1
    text, stats = extract_physics(document('<para>Retained text around a source class with a Unicode hyphen.</para>' +
                                         external.replace('os-embed', 'os\u2010embed')))
    assert b'the referenced section' not in text
    assert stats['policy']['external_exercise_unicode_hyphen_aliases'] == 1
    # An inconsistent explanatory table still fails; omission is class-scoped.
    invalid_documents = [
        document('<para>Required explanatory table follows.</para>' + grid.replace('section-key-terms', 'data-table')),
        document('<list><item>First source item.</item><title>Late title</title></list>'),
        document('<list><title>One</title><title>Two</title><item>Source item.</item></list>'),
        document('<list><title>Title</title><label>Unexpected label</label><item>Source item.</item></list>'),
        document('<list><title>Title</title>Lost text<item>Source item.</item></list>'),
        b'<!DOCTYPE document [<!ENTITY hidden "text">]>' + document('<para>Unreviewed entity.</para>'),
        document('<para>An unreviewed control character \u0081 must not become a training token.</para>'),
        document(external.replace('/>', '>Unexpected question text</link>')),
        document(external.replace('/>', '><para>Unexpected question child</para></link>')),
        document(external.replace('k12phys-ch01-ex008', '../another-service')),
        document(external.replace('class="os-embed"', 'class="os-embed" document="m1"')),
    ]
    for raw in invalid_documents:
        try: extract_physics(raw)
        except ValueError: rejected += 1
        else: raise AssertionError('Accepted invalid source structure')
    # The same title transform must retain the complete source math in items.
    raw = document('<list><title>Isotope</title><item>A uranium symbol is <m:math><m:mmultiscripts><m:mi>U</m:mi><m:mprescripts/><m:mn>92</m:mn><m:mn>235</m:mn></m:mmultiscripts></m:math>.</item></list>')
    assert b'{}_{92}^{235}{U}' in extract_physics(raw)[0]
    text, stats = extract_physics(document('<para>A physicist\u0092s \u0093quoted words\u0094 and two dash styles \u0096 \u0097 remain readable.</para>'))
    assert 'physicist’s “quoted words” and two dash styles – —' in text.decode()
    assert stats['punctuation_repairs'] == {f'U+{i:04X}':1 for i in (0x92,0x93,0x94,0x96,0x97)}
    return dict(passed=True,valid_formula_fixtures=len(valid),rejected_notation_or_structure_cases=rejected,
                titled_list_reference_and_order_preserved=True,omitted_note_tail_preserved=True,
                malformed_explanatory_table_rejected=True,recorded_punctuation_repairs_verified=True,
                external_exercise_omission_and_unicode_alias_verified=True,training_admission=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out',type=Path)
    args = parser.parse_args()
    result = check()
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_bytes((json.dumps(result,indent=2)+'\n').encode())
    print(json.dumps(result,indent=2))
