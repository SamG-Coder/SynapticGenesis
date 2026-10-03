"""Audit missing image descriptions and collect source worked examples for review.

Candidates remain outside training admission. No OCR, invented solutions or
automatic table repairs are used to turn incomplete source into targets.
"""
import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from corpus.openstax import CN, extract
from corpus.prealgebra_math import prealgebra_extension
from native_experiment import read
from prose_founder import file_hash, write


def placeholder_description(text):
    return not any(c.isalnum() for c in text)


def worked_candidate(exercise, title):
    """Return a text-complete candidate or the explicit first structural reason."""
    problems, solutions = exercise.findall(CN + 'problem'), exercise.findall(CN + 'solution')
    if len(problems) != 1 or len(solutions) != 1:
        return None, 'expected_one_problem_and_solution'
    if list(exercise.iter(CN + 'media')) or list(exercise.iter(CN + 'image')):
        return None, 'requires_image_review'
    if list(exercise.iter(CN + 'link')):
        return None, 'requires_cross_reference_review'
    root = ET.Element(CN + 'document')
    ET.SubElement(root, CN + 'title').text = title
    content = ET.SubElement(root, CN + 'content')
    selected = copy.deepcopy(exercise)
    selected.tail = None
    content.append(selected)
    try:
        body, stats = extract(ET.tostring(root), math_extension=prealgebra_extension)
    except ValueError as error:
        return None, 'conversion: ' + str(error)
    if b'Question:' not in body or b'Answer:' not in body:
        raise ValueError('Paired source lost its problem or solution marker')
    return (body, stats), None


def inspected_cases(cache, candidates):
    path = Path('data/prealgebra-review-cases-v1.json')
    cases = read(path)
    assert cases['status'] == 'source_review_only_not_training_admission'
    candidate_map = {(r['module'], r['exercise']): r for r in candidates}
    conflicts, samples = [], []
    for row in cases['conflicts']:
        root = ET.fromstring((cache / 'raw/prealgebra' / f"{row['module']}.cnxml").read_bytes())
        exercise = [n for n in root.iter(CN + 'exercise') if n.get('id') == row['exercise']]
        assert len(exercise) == 1
        table = [n for n in exercise[0].iter(CN + 'table') if n.get('id') == row['table']]
        assert len(table) == 1 and row['description_excerpt'] in table[0].get('aria-label', '')
        assert (row['module'], row['exercise']) not in candidate_map
        conflicts.append(dict(**row,
            problem_cnxml=ET.tostring(exercise[0].find(CN + 'problem'), encoding='unicode'),
            table_description=table[0].get('aria-label', ''),
            table_text=' '.join(' '.join(table[0].itertext()).split()),
            candidate_excluded=True))
    for row in cases['candidate_samples']:
        candidate = candidate_map[row['module'], row['exercise']]
        samples.append(dict(**row, text_sha256=candidate['text_sha256'], text=candidate['text']))
    return dict(review_cases_sha256=file_hash(path), source_conflicts=conflicts,
                manually_inspected_candidates=samples, limits=cases['limits'])


def review(cache, out):
    if out.exists():
        raise ValueError('Use a fresh review directory')
    previous_path = Path('reports/prealgebra-notation-audit.json')
    previous = read(previous_path)
    assert previous['passed'] and not previous['training_admitted']
    assert file_hash(cache / 'osbooks-prealgebra-bundle-LICENSE') == previous['license_sha256']
    assert file_hash(cache / 'prealgebra.collection.xml') == previous['collection_sha256']
    rows, candidates, totals = [], [], Counter()
    for source in previous['records']:
        ident = source['module']
        path = cache / 'raw/prealgebra' / f'{ident}.cnxml'
        assert file_hash(path) == source['raw_sha256'], ident
        root = ET.fromstring(path.read_bytes())
        parents = {child: parent for parent in root.iter() for child in parent}
        counts, placeholder_images, reasons = Counter(), [], Counter()

        def ancestors(node):
            while node in parents:
                node = parents[node]
                yield node

        for media in root.iter(CN + 'media'):
            counts['media_elements'] += 1
            if not placeholder_description(media.get('alt', '')):
                continue
            chain = list(ancestors(media))
            table = next((n for n in chain if n.tag == CN + 'table'), None)
            context = next((n for n in chain if n.tag in (CN + 'problem', CN + 'solution', CN + 'figure')), None)
            counts['placeholder_media_descriptions'] += 1
            role = context.tag.removeprefix(CN) if context is not None else 'other'
            counts['placeholder_in_' + role] += 1
            description = table.get('aria-label', '') if table is not None else ''
            if description:
                counts['placeholder_with_table_description'] += 1
            placeholder_images.append(dict(media_id=media.get('id'), alt=media.get('alt', ''),
                context=role, context_id=context.get('id') if context is not None else None,
                table_id=table.get('id') if table is not None else None,
                table_description_present=bool(description),
                images=[n.get('src') for n in media.iter(CN + 'image')]))
        exercise_ids = set()
        for exercise in root.iter(CN + 'exercise'):
            if exercise.find(CN + 'problem') is None or exercise.find(CN + 'solution') is None:
                continue
            key = exercise.get('id')
            assert key and key not in exercise_ids
            exercise_ids.add(key)
            counts['paired_exercises'] += 1
            if not any(n.tag == CN + 'example' for n in ancestors(exercise)):
                counts['pairs_outside_worked_examples'] += 1
                continue
            counts['worked_example_pairs'] += 1
            result, reason = worked_candidate(exercise, source['title'])
            if reason:
                reasons[reason] += 1
                continue
            body, stats = result
            counts['text_worked_candidates'] += 1
            counts['candidate_text_bytes'] += len(body)
            candidates.append(dict(module=ident, exercise=key, chapter=source['chapter'],
                module_title=source['title'], source_sha256=source['raw_sha256'],
                source_url=f"{previous['upstream_repository']}/blob/{previous['upstream_commit']}/modules/{ident}/index.cnxml",
                text=body.decode('utf-8'), text_sha256=hashlib.sha256(body).hexdigest(),
                extraction=stats, training_admitted=False,
                limitation='Source worked example; passage, factual and split review still required.'))
        totals.update(counts)
        rows.append(dict(module=ident, title=source['title'], raw_sha256=source['raw_sha256'],
                         counts=dict(counts), candidate_rejections=dict(reasons),
                         placeholder_media=placeholder_images))
    observations = inspected_cases(cache, candidates)
    out.mkdir(parents=True)
    candidates_path = out / 'candidates.json'
    attribution = ('Review candidates adapted from Prealgebra 2e by Lynn Marecek, MaryAnne Anthony-Smith, '
        'Andrea Honeycutt Mathis, OpenStax at Rice University, and the contributors credited in the source preface. '
        f"Source: {previous['upstream_repository']}/tree/{previous['upstream_commit']} . "
        'This pinned edition declares Creative Commons Attribution 4.0 International: '
        'https://creativecommons.org/licenses/by/4.0/ . Changes: selected image-free worked examples extracted '
        'from CNXML; MathML serialized to text notation; whitespace and local table labels normalized. '
        'No endorsement is implied. The repository MIT license applies to code, not these source passages.')
    write(candidates_path, dict(status='review_only_not_admitted', attribution=attribution, candidates=candidates))
    (out / 'LICENSE-source.txt').write_bytes((cache / 'osbooks-prealgebra-bundle-LICENSE').read_bytes())
    (out / 'original-preface.cnxml').write_bytes((cache / 'prealgebra-preface.cnxml').read_bytes())
    assert file_hash(out / 'original-preface.cnxml') == next(r['raw_sha256'] for r in rows if r['module'] == 'm81241')
    (out / 'ATTRIBUTION.txt').write_text(attribution + '\n', encoding='utf-8')
    report = dict(complete=True, notation_audit_sha256=file_hash(previous_path),
        upstream_repository=previous['upstream_repository'], upstream_commit=previous['upstream_commit'],
        authenticated_modules=len(rows), totals=dict(totals), modules=rows,
        candidates_path=candidates_path.as_posix(), candidates_sha256=file_hash(candidates_path),
        inspected_cases=observations,
        attribution=attribution, training_admitted=False, source_files_modified=False,
        new_native_commands=0, models_trained=False,
        implementation_sha256={p.as_posix(): file_hash(p) for p in
            (Path(__file__), Path('scripts/corpus/openstax.py'), Path('scripts/corpus/prealgebra_math.py'))},
        limits=['A placeholder description is a review flag, not proof that a whole example is unrecoverable.',
                'An ancestor table description may omit or contradict the calculation; its presence is not validation.',
                'Worked-example candidates exclude explicit images and links, but may still depend on surrounding prose.',
                'No OCR, source correction, complete factual review, held-out split or training admission is performed.'])
    write(out / 'result.json', report)
    print(json.dumps(totals, indent=2), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', type=Path, default=Path('runs/science-source-review'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    review(args.cache, args.out)
