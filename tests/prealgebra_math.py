"""Verify optional arithmetic notation against fixtures and pinned source bytes.

This review produces no admitted edition, curriculum or model checkpoint.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from corpus.openstax import MATH, extract, math_text
from corpus.prealgebra_math import prealgebra_extension


COMMIT = 'bba5f5244066884797f730149d918e33f2beefd1'
REPOSITORY = 'https://github.com/openstax/osbooks-prealgebra-bundle'


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def fixtures():
    def formula(text, extension=prealgebra_extension):
        root = ET.fromstring('<math xmlns="http://www.w3.org/1998/Math/MathML">' + text + '</math>')
        return math_text(root, extension)

    # Expected notation is written independently, including its grouping.
    examples = [
        ('<mroot><mi>n</mi><mn>3</mn></mroot>', r'\sqrt[3]{n}'),
        ('<mn>0.</mn><mover accent="true"><mn>25</mn><mo>¯</mo></mover>', r'0. \overline{25}'),
        ('<mover><mn>4</mn><mn mathsize="small">1</mn></mover>', r'\overset{1}{4}'),
        ('<mover><mo>=</mo><mo>?</mo></mover>', r'\overset{?}{=}'),
        ('<munder accentunder="true"><mrow><mo>+</mo><mn>61</mn></mrow><mtext>____</mtext></munder>', r'\underline{+ 61}'),
        ('<munder><mi>x</mi><mtext>coordinate</mtext></munder>', r'\underset{\text{coordinate}}{x}'),
        ('<mstyle fontweight="bold"><mn>327</mn></mstyle>', r'\mathbf{327}'),
        ('<mstyle fontstyle="italic"><mn>51</mn></mstyle>', r'\mathit{51}'),
        ('<mn>3</mn><menclose notation="longdiv"><mn>12</mn></menclose>', r'3 \enclose{longdiv}{12}'),
        ('<mfrac><menclose notation="updiagonalstrike"><mn>5</mn></menclose><mn>10</mn></mfrac>', r'\frac{\enclose{updiagonalstrike}{5}}{10}'),
        ('<mn>5</mn><mphantom><mn>0</mn></mphantom>', r'5 \phantom{0}'),
        ('<mover><mn>54</mn><mrow><mtext>—</mtext></mrow></mover>', r'\overline{54}'),
    ]
    for text, expected in examples:
        assert formula(text) == expected, (text, formula(text), expected)
    bad = [
        '', '<mrow/>', '<mspace width="0.2em"/>',
        '<mroot><mn>3</mn></mroot>',
        '<mover><mn>4</mn></mover>',
        '<munder><mn>4</mn><mn>1</mn><mn>2</mn></munder>',
        '<mover accent="maybe"><mn>4</mn><mn>1</mn></mover>',
        '<munder align="left"><mn>4</mn><mn>1</mn></munder>',
        '<mstyle mathvariant="double-struck"><mi>x</mi></mstyle>',
        '<menclose notation="circle"><mi>x</mi></menclose>',
        '<menclose><mi>x</mi></menclose>',
        '<mphantom><unsupported><mn>0</mn></unsupported></mphantom>',
        '<mphantom>8<mn>0</mn></mphantom>',
        '<mphantom><mn>0</mn>8</mphantom>',
        '<mphantom mathsize="huge"><mn>0</mn></mphantom>',
        '<mstyle fontweight="bold"/>',
    ]
    for text in bad:
        try:
            formula(text)
        except ValueError:
            pass
        else:
            raise AssertionError('Unreviewed or malformed notation accepted: ' + text)
    for text, _ in examples:
        try:
            formula(text, None)
        except ValueError:
            pass
        else:
            raise AssertionError('Default subset silently enabled the extension')
    return dict(explicit_notation_examples=len(examples), rejected_cases=len(bad),
                default_subset_rejections=len(examples),
                phantom_not_flattened=True, nested_cancellation_preserved=True)


def audit(cache, out):
    if out.exists():
        raise ValueError('Use a fresh audit output')
    checked = fixtures()
    survey_path = cache / 'survey.json'
    tree_path = cache / 'osbooks-prealgebra-bundle-tree.json'
    survey = json.loads(survey_path.read_text(encoding='utf-8'))
    tree = json.loads(tree_path.read_text(encoding='utf-8'))
    assert survey['status'] == 'research_only_not_admitted'
    assert not tree['truncated'] and tree['sha'] == COMMIT
    blobs = {r['path']: r for r in tree['tree'] if r['type'] == 'blob'}
    selected = [r for r in survey['records'] if r['book'] == 'prealgebra']
    assert len(selected) == len({r['module'] for r in selected}) == 75

    def pinned(path, relative, expected=None):
        raw = path.read_bytes()
        blob = hashlib.sha1(f'blob {len(raw)}\0'.encode() + raw).hexdigest()
        assert blobs[relative]['sha'] == blob and blobs[relative]['size'] == len(raw)
        if expected:
            assert expected['git_blob_sha1'] == blob and expected['raw_sha256'] == digest(raw)
        return raw

    license_raw = pinned(cache / 'osbooks-prealgebra-bundle-LICENSE', 'LICENSE')
    collection_raw = pinned(cache / 'prealgebra.collection.xml', 'collections/prealgebra-2e.collection.xml')
    ns = {'c': 'http://cnx.rice.edu/collxml', 'm': 'http://cnx.rice.edu/mdml'}
    collection = ET.fromstring(collection_raw)
    assert {n.get('document') for n in collection.findall('.//c:module', ns)} == {r['module'] for r in selected}
    license_url = collection.find('c:metadata/m:license', ns).get('url')
    assert license_url.rstrip('/').endswith('creativecommons.org/licenses/by/4.0')
    assert license_raw.startswith(b'Attribution 4.0 International')
    records, tags, rejections, default_rejections = [], Counter(), Counter(), Counter()
    formula_rejections = Counter()
    titles = {r['module']: r['title'] for r in selected}
    formula_count, baseline_success = 0, 0
    for item in selected:
        ident = item['module']
        relative = f'modules/{ident}/index.cnxml'
        assert item['repository'] == REPOSITORY and item['commit'] == COMMIT and not item['admitted']
        assert item['url'] == f'https://raw.githubusercontent.com/openstax/osbooks-prealgebra-bundle/{COMMIT}/{relative}'
        raw = pinned(cache / 'raw/prealgebra' / f'{ident}.cnxml', relative, item)
        root = ET.fromstring(raw)
        formulas = root.findall('.//' + MATH + 'math')
        rendered, formula_errors = [], []
        for index, formula in enumerate(formulas):
            try:
                rendered.append(math_text(formula, prealgebra_extension))
            except ValueError as error:
                rendered.append(None)
                formula_errors.append(dict(zero_based_index=index, error=str(error)))
                formula_rejections[str(error)] += 1
        assert len(formulas) == item['math_expressions']
        formula_count += len(formulas)
        tags.update(n.tag[len(MATH):] for n in root.iter() if n.tag.startswith(MATH))
        row = dict(module=ident, title=item['title'], chapter=item['chapter'],
                   raw_sha256=digest(raw), git_blob_sha1=item['git_blob_sha1'],
                   formula_count=len(formulas), formula_rejections=formula_errors,
                   formula_text_sha256=digest(json.dumps(rendered, ensure_ascii=False).encode()),
                   training_admitted=False)
        baseline = None
        try:
            baseline, _ = extract(raw, titles)
            baseline_success += 1
        except ValueError as error:
            default_rejections[str(error)] += 1
        try:
            body, stats = extract(raw, titles, math_extension=prealgebra_extension)
            row.update(whole_module_extracted=True, extracted_sha256=digest(body),
                       extracted_bytes=len(body), extraction=stats)
            if baseline is not None:
                assert body == baseline
        except ValueError as error:
            row.update(whole_module_extracted=False, rejection=str(error))
            rejections[str(error)] += 1
            assert baseline is None or formula_errors
        records.append(row)
    report = dict(passed=True, scope='Notation conversion audit; source admission remains pending',
                  upstream_repository=REPOSITORY, upstream_commit=COMMIT,
                  survey_sha256=digest(survey_path.read_bytes()), tree_sha256=digest(tree_path.read_bytes()),
                  license_sha256=digest(license_raw), collection_sha256=digest(collection_raw),
                  fixtures=checked, modules_authenticated=len(records), formulas_inspected=formula_count,
                  formulas_converted=formula_count - sum(formula_rejections.values()),
                  formula_rejections=dict(formula_rejections),
                  mathml_tag_counts=dict(tags), default_modules_extracted=baseline_success,
                  default_module_rejections=dict(default_rejections),
                  extended_modules_extracted=sum(r['whole_module_extracted'] for r in records),
                  remaining_module_rejections=dict(rejections), mutually_extractable_outputs_identical=True,
                  source_files_modified=False, training_admitted=False, models_trained=False,
                  records=records, implementation_sha256={str(p): digest(p.read_bytes()) for p in
                      [Path(__file__), Path('scripts/corpus/openstax.py'), Path('scripts/corpus/prealgebra_math.py')]},
                  limits=['Formula conversion is not a proof that each source calculation is correct.',
                          'Malformed source tables and unsupported CNXML remain rejected.',
                          'Image-only calculations, passage review, attribution and held-out splits need review before admission.',
                          'Spacing and visual layout are normalized. Enclosures use non-standard MathJax TeX syntax.'])
    out.write_bytes((json.dumps(report, ensure_ascii=False, indent=2) + '\n').encode())
    print(f"{report['formulas_converted']}/{formula_count} formulas converted; {report['extended_modules_extracted']}/75 complete modules extractable; no source admitted.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', type=Path, default=Path('runs/science-source-review'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    audit(args.cache, args.out)
