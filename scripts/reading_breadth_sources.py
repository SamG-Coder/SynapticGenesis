"""Authenticate selected editions and prepare common assessments for breadth trials."""
import json
from pathlib import Path

from adaptation_sources import prepare_schedule
from native_experiment import sha, write


EDITIONS = {'starter': 'early-readers-v1', 'expansion': 'reading-breadth-v1'}


def read_utf8(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def verify_sources():
    result = {}
    for name, version in EDITIONS.items():
        root = Path('data/prepared') / version
        specification = Path('data') / ('sources-' + version + '.json')
        published = Path('reports') / (version + '-manifest.json')
        manifest = read_utf8(root / 'manifest.json')
        spec = read_utf8(specification)
        assert sha(root / 'manifest.json') == sha(published)
        assert sha(specification) == manifest['source_spec_sha256']
        assert specification.read_bytes() == (root / 'source-spec.json').read_bytes()
        assert len(spec['sources']) == len(manifest['sources'])
        for selected, recorded in zip(spec['sources'], manifest['sources']):
            assert all(recorded.get(key) == value for key, value in selected.items())
            path = root / (selected['id'] + '.txt')
            assert sha(path) == selected['body_sha256'] and path.stat().st_size == recorded['clean_bytes']
        for split, recorded in manifest['outputs'].items():
            sources = [s for s in spec['sources'] if s['split'] == split]
            raw = b'\x1e'.join((root / (s['id'] + '.txt')).read_bytes() for s in sources)
            path = root / (split + '.dat')
            assert path.read_bytes() == raw and sha(path) == recorded['sha256']
            assert path.stat().st_size == recorded['bytes'] and len(sources) == recorded['documents']
        result[name] = dict(directory=root.resolve().as_posix(), manifest=manifest,
                            manifest_sha256=sha(root / 'manifest.json'), specification_sha256=sha(specification))
    assert len(result['starter']['manifest']['sources']) == 16
    assert len(result['expansion']['manifest']['sources']) == 42
    return result


def prepare_assessment(out, editions):
    out.mkdir()
    groups = {}
    for split in ('train', 'validation'):
        documents, records = [], []
        for name, edition in editions.items():
            first = len(documents)
            for source in edition['manifest']['sources']:
                if source['split'] != split:
                    continue
                raw = (Path(edition['directory']) / (source['id'] + '.txt')).read_bytes()
                documents.append(raw)
                records.append(dict(edition=name, id=source['id'], body_sha256=source['body_sha256']))
            groups[name + '_' + split] = dict(split=split, first=first, end=len(documents),
                                              pairs=sum(len(d)-1 for d in documents[first:]))
        (out / (split + '.dat')).write_bytes(b'\x1e'.join(documents))
        write(out / (split + '-documents.json'), records)
    schedule = prepare_schedule(out / 'train.dat', 1337, 1, out / 'evaluation.sg')
    result = dict(groups=groups, files={p.name: sha(p) for p in out.iterdir() if p.is_file()},
                  evaluation_schedule_sha256=schedule['schedule_sha256'])
    write(out / 'manifest.json', result)
    return result


def grouped_reading(reading, assessment):
    values = {}
    for name, group in assessment['groups'].items():
        docs = reading[group['split']]['documents'][group['first']:group['end']]
        pairs = sum(r['pairs'] for r in docs)
        assert pairs == group['pairs'] and pairs > 0
        values[name] = sum(r['loss'] * r['pairs'] for r in docs) / pairs
    return values


def source_exposure(raw, observations):
    """Describe exact sequential targets without simulating model learning."""
    docs = raw.split(b'\x1e')
    windows = [(i, start, min(128, len(doc)-1-start))
               for i, doc in enumerate(docs) for start in range(0, len(doc)-1, 128)]
    selected = [windows[i % len(windows)] for i in range(observations)]
    return dict(documents=len(docs), observations_per_pass=len(windows),
                targets_per_pass=sum(len(d)-1 for d in docs), presented_targets=sum(w[2] for w in selected),
                unique_targets=sum(w[2] for w in set(selected)), complete_passes=observations // len(windows),
                pass_fraction=observations / len(windows))
