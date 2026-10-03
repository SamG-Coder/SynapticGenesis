"""Apply a small reviewed errata set to separate, unadmitted Physics copies."""
import argparse
import difflib
import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET

from corpus.reviewed_edition import authenticate, require, require_clear, reservations
from corpus.selection import SELECTION
from native_experiment import read, sha
from prepare_openstax import corrected_body
from prose_founder import write


SPEC = Path('data/physics-followup-errata-v1.json')
MODULES = {'m54273', 'm54287', 'm54290'}
CORRECTIONS = {'energy-balance-signs', 'thermal-energy-total', 'thermal-energy-glossary',
               'absolute-zero-quantum', 'absolute-zero-summary', 'absolute-zero-glossary',
               'heat-capacity-mass'}


def compiled(spec_path=SPEC):
    spec_path = Path(spec_path)
    plan = read(spec_path)
    require(plan['version'] == 'physics-followup-errata-v1' and
            plan['status'] == 'reviewed_corrections_not_training_admission', 'Unexpected errata review status')
    for key in ('base_source_spec', 'base_preparation_audit'):
        require(sha(plan[key]) == plan[key + '_sha256'], 'Errata base identity changed: ' + key)
    prepared = Path(plan['base_prepared'])
    require(sha(prepared / 'manifest.json') == plan['base_manifest_sha256'], 'Errata base manifest changed')
    edition = authenticate(plan['base_source_spec'], plan['base_preparation_audit'], prepared)
    require(plan['upstream_commit'] == edition.spec['upstream_commit'] and
            plan['upstream_repository'] == edition.spec['upstream_repository'], 'Errata source commit differs')
    require(len(plan['modules']) == len(MODULES) and {r['id'] for r in plan['modules']} == MODULES,
            'Errata module scope changed')
    require(len(plan['corrections']) == len(CORRECTIONS) and
            {r['id'] for r in plan['corrections']} == CORRECTIONS, 'Errata correction scope changed')
    registry, protected = reservations([edition])
    originals = {r['id']: r for r in edition.manifest['sources']}
    inputs = [spec_path, SELECTION, *edition.inputs, *(Path(r['path']) for r in protected)]
    changes = []
    for module in plan['modules']:
        ident = module['id']
        base = originals[ident]
        require(base['split'] == 'train', 'Errata cannot rewrite a protected module')
        raw_path = Path(plan['source_cache']) / f'{ident}.cnxml'
        raw = raw_path.read_bytes()
        require(sha(raw_path) == base['raw_sha256'], 'Original source module changed')
        root = ET.fromstring(raw)
        body = (prepared / f'{ident}.txt').read_bytes()
        require(sha(prepared / f'{ident}.txt') == module['base_sha256'] == base['clean_sha256'] and
                len(body) == module['base_bytes'], 'Prepared errata input changed')
        records = [r for r in plan['corrections'] if r['module'] == ident]
        require([r['id'] for r in records] == module['correction_ids'], 'Module correction order changed')
        for correction in records:
            elements = [e for e in root.iter() if e.get('id') == correction['source_element_id']]
            require(len(elements) == 1 and correction['source_raw_sha256'] == base['raw_sha256'],
                    'Errata source element changed')
            digest = lambda value: hashlib.sha256(value.encode('utf-8')).hexdigest()
            require(digest(''.join(elements[0].itertext())) == correction['source_element_text_sha256'],
                    'Errata source element text changed')
            require(correction['before'] and correction['after'] and
                    correction['before'] != correction['after'] and '\x1e' not in correction['after'] and
                    digest(correction['before']) == correction['before_sha256'] and
                    digest(correction['after']) == correction['after_sha256'], 'Errata passage identity changed')
            require(correction['issue'].strip() and correction['references'] and
                    all(link.startswith('https://openstax.org/books/') for link in correction['references']),
                    'Missing reviewed correction evidence')
        amended = corrected_body(body, dict(extracted_sha256=module['base_sha256'],
            corrected_sha256=module['corrected_sha256'], corrections=records))
        require(len(amended) == module['corrected_bytes'], 'Errata output extent changed')
        require_clear(amended, registry)
        changes.append(dict(id=ident, original=body, corrected=amended, stage=base['stage'],
                            corrections=module['correction_ids']))
        inputs.append(raw_path)
    require(sum(len(c['corrections']) for c in changes) == len(CORRECTIONS), 'Unapplied correction')
    return plan, edition, changes, list(dict.fromkeys(inputs))


def review(out, spec_path=SPEC):
    out = Path(out)
    require(not out.exists(), 'Use a fresh errata review directory')
    plan, edition, changes, inputs = compiled(spec_path)
    inputs.extend(Path(p) for p in ('scripts/review_physics_errata.py', 'scripts/prepare_openstax.py',
        'scripts/corpus/reviewed_edition.py', 'scripts/corpus/paired_selection.py',
        'scripts/corpus/selection.py', 'scripts/native_experiment.py'))
    pins = {str(p): sha(p) for p in inputs}
    (out / 'review-text').mkdir(parents=True)
    (out / 'diffs').mkdir()
    for row in changes:
        (out / 'review-text' / f"{row['id']}.txt").write_bytes(row['corrected'])
        difference = ''.join(difflib.unified_diff(row['original'].decode().splitlines(keepends=True),
            row['corrected'].decode().splitlines(keepends=True),
            fromfile=f"selected-physics-v1/{row['id']}.txt", tofile=f"unadmitted-errata/{row['id']}.txt"))
        (out / 'diffs' / f"{row['id']}.patch").write_text(difference, encoding='utf-8', newline='\n')
    (out / 'errata-spec.json').write_bytes(Path(spec_path).read_bytes())
    (out / 'ATTRIBUTION.txt').write_text(plan['attribution'] + '\n', encoding='utf-8', newline='\n')
    for name in ('LICENSE-source.txt', 'original-preface.cnxml', 'original-collection.xml'):
        (out / name).write_bytes((edition.prepared / name).read_bytes())
    target_delta = sum(len(r['corrected']) - len(r['original']) for r in changes)
    window_delta = sum((len(r['corrected']) - 1 + 127) // 128 -
                       (len(r['original']) - 1 + 127) // 128 for r in changes)
    require(all(sha(p) == value for p, value in pins.items()), 'Review changed an authenticated input')
    outputs = {p.relative_to(out).as_posix(): sha(p) for p in out.rglob('*') if p.is_file()}
    result = dict(complete=True, status='review_copy_not_admitted', specification_sha256=sha(spec_path),
        source_edition=edition.spec['version'], corrections=len(CORRECTIONS), changed_modules=len(changes),
        modules=[dict(id=r['id'], stage=r['stage'], correction_ids=r['corrections']) for r in changes],
        authenticated_inputs=pins, output_sha256=outputs, old_sources_unchanged=True,
        reserved_text_overlap_checked=True, validation_and_test_rewritten=False,
        training_admitted=False, native_commands=0, new_model_updates=0, curriculum_created=False,
        conditional_single_pass_target_byte_delta=target_delta,
        conditional_single_pass_128_byte_observation_delta=window_delta,
        required_next_step=plan['current_curriculum_action'], limits=plan['review_scope'])
    write(out / 'result.json', result)
    print('Prepared seven corrections in three separate review copies; no training admission or model execution.')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', type=Path, default=SPEC)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    review(args.out, args.spec)
