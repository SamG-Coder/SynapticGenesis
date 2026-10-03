"""Check the revised edition against the original modules and reviewed errata."""
import argparse
from collections import Counter
from copy import deepcopy
from pathlib import Path
import re
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from corpus.physics_edits import digest
from corpus.physics_revision import REVISION_PROVENANCE, compiled_revision, reviewed_inputs
from corpus.reviewed_edition import authenticate, require_clear, reservations
from corpus.selection import SELECTION
from native_experiment import read, sha
from prepare_physics import prepare
from prose_founder import write


def audit(spec_path, prepared, cache, report):
    assert not report.exists(), 'Preserve earlier reports'
    spec, manifest = read(spec_path), read(prepared / 'manifest.json')
    plan, base, reviewed, inputs = reviewed_inputs()
    inputs.extend([spec_path, SELECTION, *base.inputs])
    inputs.extend(sorted((cache / 'raw/physics').glob('*.cnxml')))
    implementations = {Path(m.__file__).resolve() for m in tuple(sys.modules.values())
                       if getattr(m, '__file__', None) and
                       Path(m.__file__).resolve().is_relative_to(ROOT / 'scripts')}
    inputs.extend([Path(__file__), *implementations])
    pins = {Path(p).as_posix(): sha(p) for p in inputs}
    old_rows = {r['id']: r for r in base.spec['sources']}
    reviewed_rows = {r['id']: r for r in reviewed}
    assert spec['version'] == manifest['version'] == 'selected-physics-v2'
    assert len(spec['sources']) == len(manifest['sources']) == 99
    assert [r['id'] for r in spec['sources']] == [r['id'] for r in base.spec['sources']]
    assert Counter(r['split'] for r in spec['sources']) == dict(train=90, validation=4, test=5)
    assert spec['chapter_policy'] == base.spec['chapter_policy']
    assert spec['protected_files'] == base.spec['protected_files']
    assert spec['overlap_omissions'] == base.spec['overlap_omissions']
    assert spec['upstream_commit'] == base.spec['upstream_commit']
    assert spec['upstream_repository'] == base.spec['upstream_repository']
    bodies, changes = {}, []
    registry, _ = reservations([base])
    for row in spec['sources']:
        ident = row['id']
        old = (base.prepared / f'{ident}.txt').read_bytes()
        raw = (prepared / f'{ident}.txt').read_bytes()
        expected, corrected = deepcopy(old_rows[ident]), old.decode('utf-8')
        corrections = [r for r in plan['corrections'] if r['module'] == ident]
        for correction in corrections:
            assert corrected.count(correction['before']) == 1
            corrected = corrected.replace(correction['before'], correction['after'], 1)
        assert raw == corrected.encode('utf-8')
        if corrections:
            assert row['split'] == 'train' and row['stage'] == 2
            assert raw == reviewed_rows[ident]['corrected'] and raw != old
            expected['corrected_sha256'] = digest(raw)
            expected['correction_ids'] += [r['id'] for r in corrections]
            # Marker counts are independent of the passage-replacement implementation.
            for marker in ('Question:', 'Answer:', 'Solution'):
                assert raw.count(marker.encode()) == old.count(marker.encode())
            require_clear(raw, registry)
            changes.append(dict(id=ident, old_bytes=len(old), new_bytes=len(raw),
                old_sha256=digest(old), new_sha256=digest(raw),
                correction_ids=[r['id'] for r in corrections]))
        assert row == expected
        bodies[ident] = raw
    assert {r['id'] for r in changes} == {'m54273', 'm54287', 'm54290'}
    assert sum(len(r['correction_ids']) for r in spec['sources']) == 12
    assert sum(r['new_bytes'] - r['old_bytes'] for r in changes) == -187
    for split in ('train', 'validation', 'test'):
        chosen = [bodies[r['id']] for r in spec['sources'] if r['split'] == split]
        joined = b'\x1e'.join(chosen)
        assert joined == (prepared / f'{split}.dat').read_bytes()
        assert manifest['outputs'][split] == dict(bytes=len(joined), documents=len(chosen),
            sha256=digest(joined),
            word_like_units=len(re.findall(r"[A-Za-z]+(?:['-][A-Za-z]+)*", joined.decode())))
        if split != 'train':
            assert joined == (base.prepared / f'{split}.dat').read_bytes()
    for stage in manifest['stages']:
        raw = b'\x1e'.join(bodies[r['id']] for r in spec['sources']
                          if r['split'] == 'train' and r['stage'] == stage['id'])
        assert raw == (prepared / stage['file']).read_bytes()
        assert len(raw) == stage['bytes'] and digest(raw) == stage['sha256']
        if stage['id'] != 2:
            assert raw == (base.prepared / stage['file']).read_bytes()
    for name in ('LICENSE-source.txt', 'original-preface.cnxml', 'original-collection.xml'):
        assert (prepared / name).read_bytes() == (base.prepared / name).read_bytes()

    rejected = []
    with tempfile.TemporaryDirectory(prefix='physics-revision-') as directory:
        temporary = Path(directory)
        provisional = temporary / 'integration-audit.json'
        # This temporary record exercises downstream authentication, not publication.
        write(provisional, dict(passed=True, source_spec_sha256=sha(spec_path),
            prepared_manifest_sha256=sha(prepared / 'manifest.json'), outputs=manifest['outputs']))
        admitted = authenticate(spec_path, provisional, prepared)
        assert len(admitted.provenance) == 17
        copy = temporary / 'prepared'
        shutil.copytree(prepared, copy)
        protected_id = next(r['id'] for r in spec['sources'] if r['split'] == 'validation')
        files = ['source-spec.json', 'manifest.json', 'm54273.txt', protected_id + '.txt',
                 'train.dat', 'validation.dat', 'test.dat', 'stage-2.dat', 'LICENSE-source.txt',
                 'passage-review.json', 'ATTRIBUTION.txt', *REVISION_PROVENANCE.values()]
        for name in files:
            path = copy / name
            original = path.read_bytes()
            try:
                path.write_bytes(original + b' ')
                try:
                    authenticate(spec_path, provisional, copy)
                except ValueError as error:
                    rejected.append(dict(case='prepared:' + name, reason=str(error)))
                else:
                    raise AssertionError('Altered prepared input accepted: ' + name)
            finally:
                path.write_bytes(original)
        edits = {
            'module-order': lambda s: s['sources'].reverse(),
            'heldout-split': lambda s: next(r for r in s['sources'] if r['split'] == 'test').update(split='train'),
            'stage': lambda s: s['sources'][0].update(stage=4),
            'text-hash': lambda s: s['sources'][0].update(corrected_sha256='0' * 64),
            'correction-ids': lambda s: next(r for r in s['sources'] if r['id'] == 'm54273').update(correction_ids=[]),
            'upstream': lambda s: s.update(upstream_commit='0' * 40),
            'attribution': lambda s: s.update(attribution='Missing original source attribution'),
            'review-pin': lambda s: s.update(followup_review_sha256='0' * 64),
            'review-location': lambda s: s.update(followup_review='different-review.json'),
            'protected-inputs': lambda s: s.update(protected_files=[]),
        }
        for name, edit in edits.items():
            altered = deepcopy(spec)
            edit(altered)
            assert altered != spec
            try:
                compiled_revision(altered, cache)
            except ValueError as error:
                assert 'differs from reviewed changes' in str(error)
                rejected.append(dict(case='selection:' + name, reason=str(error)))
            else:
                raise AssertionError('Unreviewed selection accepted: ' + name)
        altered_spec = temporary / 'sources.json'
        write(altered_spec, altered)
        destination = temporary / 'rejected'
        try:
            prepare(altered_spec, cache, destination)
        except ValueError as error:
            assert 'differs from the admitted edition' in str(error) and not destination.exists()
            rejected.append(dict(case='admission-before-output', reason=str(error)))
        else:
            raise AssertionError('Unadmitted input produced output')
    assert all(sha(path) == value for path, value in pins.items())
    result = dict(passed=True, version=spec['version'], source_spec_sha256=sha(spec_path),
        prepared_manifest_sha256=sha(prepared / 'manifest.json'), outputs=manifest['outputs'],
        stages=manifest['stages'], modules=99, changed_modules=changes, unchanged_modules=96,
        split_modules=dict(Counter(r['split'] for r in spec['sources'])),
        all_module_bytes_independently_reconstructed=True, followup_corrections=7,
        total_declared_text_corrections=12, validation_and_test_byte_identical=True,
        unchanged_topic_stages=[1, 3, 4], original_source_notices_byte_identical=True,
        provenance_files=17, source_inputs_unchanged=True, authenticated_inputs=pins,
        prepared_files={p.name: sha(p) for p in sorted(prepared.iterdir()) if p.is_file()},
        rejected_inputs=rejected, native_commands=0, model_learning_verified=False,
        reserved_tests_scored=False, reproduction_admitted=False)
    report.parent.mkdir(parents=True, exist_ok=True)
    write(report, result)
    # Authenticate the actual published record after all independent checks passed.
    authenticate(spec_path, report, prepared)
    print('Physics v2: 99 modules, 7 follow-up corrections, 96 unchanged modules,',
          len(rejected), 'altered-input rejections; no model execution.')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sources', type=Path, default=ROOT / 'data/sources-physics-v2.json')
    parser.add_argument('--prepared', type=Path, default=Path('data/prepared/physics-v2-reviewed'))
    parser.add_argument('--cache', type=Path, default=Path('runs/science-source-review'))
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    audit(args.sources, args.prepared, args.cache, args.report)
