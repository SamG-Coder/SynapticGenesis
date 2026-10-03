"""Authenticate prepared, reviewed textbook editions without reselecting passages."""
from dataclasses import dataclass
from pathlib import Path
import re

from corpus.paired_selection import PairReservations, probe_contexts
from corpus.selection import require_training_spec
from extend_curriculum import documents
from native_experiment import read, sha


def require(condition, message):
    if not condition:
        raise ValueError(message)


@dataclass
class ReviewedEdition:
    spec: dict
    manifest: dict
    prepared: Path
    paired: bool
    inputs: list
    provenance: dict


def authenticate(spec_path, audit_path, prepared, *, paired=False):
    """Bind all document, split, stage and attribution bytes to the published audit."""
    spec_path, audit_path, prepared = map(Path, (spec_path, audit_path, prepared))
    spec = require_training_spec(spec_path)
    audit, manifest = read(audit_path), read(prepared / 'manifest.json')
    require(audit['passed'] and sha(spec_path) == audit['source_spec_sha256']
            == manifest['source_spec_sha256'] and
            sha(prepared / 'manifest.json') == audit['prepared_manifest_sha256'],
            'Reviewed edition differs from its published audit')
    require((prepared / 'source-spec.json').read_bytes() == spec_path.read_bytes(),
            'Prepared source specification changed')
    require(manifest['version'] == spec['version'] and
            manifest['source_repository'] == spec['upstream_repository'] and
            manifest['source_commit'] == spec['upstream_commit'], 'Reviewed edition identity changed')
    expected = {r['id']: r for r in spec['sources']}
    records = {r['id']: r for r in manifest['sources']}
    require(len(expected) == len(spec['sources']) == len(records) == len(manifest['sources'])
            and expected.keys() == records.keys(), 'Reviewed document inventory changed')
    inputs = [spec_path, audit_path, prepared / 'manifest.json', prepared / 'source-spec.json']
    bodies = {}
    for ident, record in records.items():
        require(re.fullmatch(r'[A-Za-z0-9_-]+', ident) is not None and
                {k: v for k, v in record.items() if k not in ('clean_bytes', 'clean_sha256')} == expected[ident],
                'Reviewed document identity changed')
        path = prepared / f'{ident}.txt'
        raw = path.read_bytes()
        require(len(raw) == record['clean_bytes'] and sha(path) == record['clean_sha256']
                and len(documents(raw)) == 1, 'Prepared document changed: ' + ident)
        raw.decode('utf-8')
        bodies[ident] = raw
        inputs.append(path)
    require(set(manifest['outputs']) == {'train', 'validation', 'test'} and
            manifest['outputs'] == audit['outputs'], 'Prepared split inventory changed')
    for split, row in manifest['outputs'].items():
        selected = [bodies[r['id']] for r in spec['sources'] if r['split'] == split]
        path = prepared / f'{split}.dat'
        require(b'\x1e'.join(selected) == path.read_bytes() and sha(path) == row['sha256']
                and len(selected) == row['documents'], 'Prepared split changed: ' + split)
        inputs.append(path)
    require([s['id'] for s in spec['stages']] == [1, 2, 3, 4] and
            len(manifest['stages']) == 4, 'Reviewed topic order changed')
    for stage, intended in zip(manifest['stages'], spec['stages']):
        require({k: stage[k] for k in ('id', 'name')} == intended and
                stage['file'] == f"stage-{stage['id']}.dat", 'Prepared topic identity changed')
        path = prepared / stage['file']
        selected = [bodies[r['id']] for r in spec['sources']
                    if r['split'] == 'train' and r['stage'] == stage['id']]
        require(selected and b'\x1e'.join(selected) == path.read_bytes() and
                sha(path) == stage['sha256'], 'Prepared topic changed: ' + stage['file'])
        inputs.append(path)

    provenance = {'source-spec.json': spec_path, 'manifest.json': prepared / 'manifest.json',
                  'preparation-audit.json': audit_path}
    for key, filename in (('passage_review', 'passage-review.json'), ('source_edits', 'source-edits.json'),
                          ('source_review_spec', 'source-review-spec.json'),
                          ('corrected_review', 'corrected-review.json')):
        if key not in spec:
            continue
        path = Path(spec[key])
        require(sha(path) == spec[key + '_sha256'], 'Reviewed input changed: ' + key)
        if key in ('passage_review', 'source_edits'):
            require((prepared / filename).read_bytes() == path.read_bytes(),
                    'Prepared review copy changed: ' + filename)
            inputs.append(prepared / filename)
        provenance[filename] = path
        inputs.append(path)
    if spec['version'] == 'selected-physics-v2':
        from corpus.physics_revision import REVISION_PROVENANCE
        for key, filename in REVISION_PROVENANCE.items():
            path = Path(spec[key])
            require(sha(path) == spec[key + '_sha256'] and
                    (prepared / filename).read_bytes() == path.read_bytes(),
                    'Prepared revision provenance changed: ' + key)
            provenance[filename] = path
            inputs.extend((path, prepared / filename))
    auxiliary_spec = read(spec['source_review_spec']) if 'source_review_spec' in spec else spec
    auxiliary = {r['name']: r for r in auxiliary_spec['auxiliary_files']}
    for name, filename in (('license', 'LICENSE-source.txt'), ('preface', 'original-preface.cnxml'),
                           ('collection', 'original-collection.xml')):
        path = prepared / filename
        record = auxiliary[name]
        require(sha(path) == record.get('sha256', record.get('raw_sha256')),
                'Prepared source attribution changed: ' + name)
        provenance[filename] = path
        inputs.append(path)
    path = prepared / 'ATTRIBUTION.txt'
    require(path.read_bytes() == (spec['attribution'] + '\n').encode('utf-8'),
            'Prepared attribution text changed')
    provenance[path.name] = path
    inputs.append(path)
    return ReviewedEdition(spec, manifest, prepared, paired, inputs, provenance)


def reservations(editions):
    """Reserve both editions' holdouts plus every previously protected input."""
    registry, protected = PairReservations(), {}
    for edition in editions:
        for row in edition.spec['protected_files']:
            previous = protected.setdefault(row['path'], row)
            require(previous == row, 'Conflicting protected input identity')
        for split in ('validation', 'test'):
            path = edition.prepared / f'{split}.dat'
            for raw in documents(path.read_bytes()):
                registry.add(raw.decode('utf-8'), path.as_posix(), paired=edition.paired)
            row = dict(path=path.as_posix(), sha256=sha(path), kind='documents')
            previous = protected.setdefault(row['path'], row)
            require(previous == row, 'Conflicting protected split identity')
    for row in protected.values():
        path = Path(row['path'])
        require(sha(path) == row['sha256'], 'Protected input changed: ' + str(path))
        if row['kind'] == 'probe':
            for context in probe_contexts(path.read_bytes()):
                registry.contexts.setdefault(context, row['path'])
        elif row['kind'] == 'documents':
            for raw in documents(path.read_bytes()):
                registry.add(raw.decode('utf-8'), row['path'])
        else:
            raise ValueError('Unknown protected input type')
    return registry, list(protected.values())


def require_clear(raw, registry, *, paired=False):
    for doc in documents(raw):
        conflict = registry.conflict(doc.decode('utf-8'), paired=paired)
        require(conflict is None, 'Reserved content in proposed curriculum: ' + str(conflict))
