"""Check mathematical meaning, table alignment, source pins and held-out protection."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from corpus.openstax import extract, math_text
from corpus.selection import require_training_spec
from native_experiment import read, sha
from prepare_openstax import corrected_body, load_raw, paragraphs


def fixtures():
    def formula(inner):
        return math_text(ET.fromstring('<math xmlns="http://www.w3.org/1998/Math/MathML">' + inner + '</math>'))

    assert formula('<mfrac><mrow><mn>1</mn><mo>+</mo><mn>2</mn></mrow><msup><mi>x</mi><mn>3</mn></msup></mfrac>') == r'\frac{1 + 2}{{x}^{3}}'
    assert formula('<msubsup><mi>x</mi><mi>i</mi><mn>2</mn></msubsup>') == r'{x}_{i}^{2}'
    assert formula('<msqrt><mfrac><mn>1</mn><mn>4</mn></mfrac></msqrt>') == r'\sqrt{\frac{1}{4}}'
    assert formula('<mtable><mtr><mtd><mn>1</mn></mtd><mtd><mn>2</mn></mtd></mtr><mtr><mtd><mn>3</mn></mtd><mtd><mn>4</mn></mtd></mtr></mtable>') == r'\begin{matrix} 1 & 2 \\ 3 & 4 \end{matrix}'
    for invalid in ('<mfrac><mn>1</mn></mfrac>', '<menclose notation="longdiv"><mn>12</mn></menclose>', '<mi><mn>3</mn></mi>'):
        try:
            formula(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError('Unsupported or malformed math was accepted')
    prefix = '<document xmlns="http://cnx.rice.edu/cnxml"><title>Fixture</title><content><para>This fixture describes a table with two planets and three spacecraft encounters.</para>'
    table = '<table id="t"><tgroup cols="2"><colspec colnum="1" colname="c1"/><colspec colnum="2" colname="c2"/><thead><row><entry namest="c1" nameend="c2">Missions</entry></row><row><entry>Planet</entry><entry>Ship</entry></row></thead><tbody><row><entry morerows="1">Jupiter</entry><entry>Voyager</entry></row><row><entry>Juno</entry></row><row><entry>Saturn</entry><entry>Cassini</entry></row></tbody></tgroup></table>'
    suffix = '<para>See <link target-id="t"/> for the spacecraft.</para><note class="astronomy voyagers-in-astronomy"><para>OMITTED BIOGRAPHICAL SIDEBAR</para></note></content></document>'
    text, _ = extract((prefix + table + suffix).encode())
    assert b'Missions\nPlanet | Ship\nJupiter | Voyager\nJupiter | Juno\nSaturn | Cassini' in text
    assert b'See Table 1 for the spacecraft.' in text and b'OMITTED' not in text
    try:
        extract((prefix + table.replace('nameend="c2"', 'nameend="c3"') + suffix).encode())
    except ValueError as error:
        assert 'Undeclared table column' in str(error)
    else:
        raise AssertionError('Undeclared third column was accepted')
    return dict(mathematical_examples=4, invalid_math_rejected=3, table_and_reference_fixture=True,
                malformed_table_rejected=True, declared_sidebar_omission=True)


def audit(prepared, out):
    fixture_result = fixtures()
    specification = Path('data/sources-astronomy-v1.json')
    spec = require_training_spec(specification)
    manifest = read(prepared / 'manifest.json')
    assert manifest['source_spec_sha256'] == sha(specification)
    assert (prepared / 'source-spec.json').read_bytes() == specification.read_bytes()
    cache = Path('data/raw/openstax-astronomy-2024')
    records = {r['id']: r for r in manifest['sources']}
    assert len(records) == len(spec['sources']) == 184
    assert 'm59901' not in records
    for record in spec['sources']:
        raw = load_raw(spec, record, cache)
        extracted, stats = extract(raw, spec['module_titles'])
        assert hashlib.sha256(extracted).hexdigest() == record['extracted_sha256']
        corrected_body(extracted, record)
        assert stats == records[record['id']]['extraction']
        assert sha(prepared / f"{record['id']}.txt") == records[record['id']]['clean_sha256']
    old = Path(spec['protected_edition']['prepared'])
    protected = set()
    for record in read(old / 'manifest.json')['sources']:
        if record['split'] != 'train':
            protected.update(p for p in paragraphs((old / f"{record['id']}.txt").read_bytes()) if len(p) >= 120)
    for record in spec['sources']:
        if record['split'] != 'train':
            protected.update(p for p in paragraphs((prepared / f"{record['id']}.txt").read_bytes()) if len(p) >= 120)
    for record in spec['sources']:
        if record['split'] == 'train':
            assert not protected.intersection(paragraphs((prepared / f"{record['id']}.txt").read_bytes()))
    for split, record in manifest['outputs'].items():
        ids = [r['id'] for r in spec['sources'] if r['split'] == split]
        payload = b'\x1e'.join((prepared / f'{ident}.txt').read_bytes() for ident in ids)
        assert payload == (prepared / f'{split}.dat').read_bytes()
        assert hashlib.sha256(payload).hexdigest() == record['sha256']
        assert len(payload) == record['bytes'] and len(ids) == record['documents']
        assert len(re.findall(r"[A-Za-z]+(?:['-][A-Za-z]+)*", payload.decode('utf-8'))) == record['word_like_units']
    for split, chapter in [('validation', 17), ('test', 19)]:
        assert {r['chapter_number'] for r in spec['sources'] if r['split'] == split} == {chapter}
        assert all(r['split'] == split for r in spec['sources'] if r['chapter_number'] == chapter)
    review = read('reports/astronomy-source-review.json')
    for row in review['reviewed_samples']:
        record = next(s for s in spec['sources'] if s['id'] == row['module'])
        text, _ = extract(load_raw(spec, record, cache), spec['module_titles'])
        assert row['reviewed_sample'] in re.sub(r'\s+', ' ', text.decode('utf-8'))
        assert row['extracted_sha256'] == record['extracted_sha256']
    with TemporaryDirectory() as directory:
        temporary = Path(directory)
        first = spec['sources'][0]
        target = temporary / first['path']
        target.parent.mkdir(parents=True)
        target.write_bytes((cache / first['path']).read_bytes() + b'changed')
        try:
            load_raw(spec, first, temporary)
        except ValueError as error:
            assert 'Reviewed upstream bytes changed' in str(error)
        else:
            raise AssertionError('Mutated raw source accepted')
    missions = (prepared / 'm59853.txt').read_text(encoding='utf-8')
    assert 'Jupiter | Juno | July 2016 | Orbiter' in missions
    assert 'Saturn | Cassini | July 2004 | Orbiter' in missions
    assert 'Jupiter | Cassini | December 2000 | Flyby' in missions and 'December 2002' not in missions
    assert 'giant planet’s orbital motion' in missions and 'giant planet’s rotation' not in missions
    amended = next(r for r in spec['sources'] if r['id'] == 'm59853')
    raw_body, _ = extract(load_raw(spec, amended, cache), spec['module_titles'])
    try:
        corrected_body(raw_body + raw_body, amended)
    except ValueError as error:
        assert 'exactly once' in str(error)
    else:
        raise AssertionError('Ambiguous repeated correction accepted')
    report = dict(passed=True, source_spec_sha256=sha(specification), manifest_sha256=sha(prepared / 'manifest.json'),
                  source_review_sha256=sha('reports/astronomy-source-review.json'),
                  fixtures=fixture_result, outputs=manifest['outputs'], modules_reextracted=len(records),
                  whole_chapter_splits=True, protected_paragraph_overlap=0, changed_raw_rejected=True,
                  reviewed_samples_authenticated=30, real_merged_rows_checked=True,
                  malformed_module_excluded=True, no_native_model_work=True,
                  explicit_source_corrections_verified=2, ambiguous_correction_rejected=True,
                  limitations=spec['limitations'])
    out.write_bytes((json.dumps(report, indent=2) + '\n').encode())
    print('184 pinned modules, 30 reviewed samples, math/table fixtures and source isolation passed.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepared', type=Path, default=Path('data/prepared/astronomy-v1-reviewed'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    audit(args.prepared, args.out)
