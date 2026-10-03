"""Append the admitted astronomy text to the complete prose curriculum."""
import argparse
from pathlib import Path
import re
import shutil

from corpus.selection import require_training_spec
from extend_curriculum import documents, prepare as extend, read_schedule
from native_experiment import read, sha, verified_book_manifest
from prose_founder import write


def protect_training(training, holdouts):
    """Reject exact documents and normalized substantial held-out paragraphs."""
    protected_documents, protected_paragraphs = set(), set()
    for raw in holdouts:
        protected_documents.update(documents(raw))
        for doc in documents(raw):
            protected_paragraphs.update(p for p in paragraphs(doc) if len(p) >= 120)
    for raw in training:
        for doc in documents(raw):
            if doc in protected_documents:
                raise ValueError('Training includes an exact held-out document')
            if protected_paragraphs.intersection(paragraphs(doc)):
                raise ValueError('Training includes a substantial held-out paragraph')


def paragraphs(raw):
    return {re.sub(r'\s+', ' ', p).strip()
            for p in re.split(r'\n\s*\n', raw.decode('utf-8'))}


def authenticate_astronomy(prepared):
    source = Path('data/sources-astronomy-v1.json')
    spec = require_training_spec(source)
    manifest = read(prepared / 'manifest.json')
    audit = read('reports/astronomy-corpus.json')
    if (not audit['passed'] or sha(prepared / 'manifest.json') != audit['manifest_sha256']
            or sha(source) != audit['source_spec_sha256']
            or sha(source) != manifest['source_spec_sha256']
            or (prepared / 'source-spec.json').read_bytes() != source.read_bytes()):
        raise ValueError('Astronomy edition differs from the reviewed preparation')
    records = {r['id']: r for r in manifest['sources']}
    for record in spec['sources']:
        body = prepared / f"{record['id']}.txt"
        if sha(body) != records[record['id']]['clean_sha256']:
            raise ValueError('Prepared astronomy document changed: ' + record['id'])
    for split, row in manifest['outputs'].items():
        payload = b'\x1e'.join((prepared / f"{r['id']}.txt").read_bytes()
                                for r in spec['sources'] if r['split'] == split)
        path = prepared / f'{split}.dat'
        if payload != path.read_bytes() or sha(path) != row['sha256']:
            raise ValueError('Prepared astronomy split changed: ' + split)
    for stage in manifest['stages']:
        payload = b'\x1e'.join((prepared / f"{r['id']}.txt").read_bytes()
                                for r in spec['sources'] if r['split'] == 'train' and r['stage'] == stage['id'])
        path = prepared / stage['file']
        if path.read_bytes() != payload or sha(path) != stage['sha256']:
            raise ValueError('Prepared astronomy stage changed: ' + str(stage['id']))
    auxiliaries = {r['name']: r for r in spec['auxiliary_files']}
    if (sha(prepared / 'LICENSE-source.txt') != auxiliaries['license']['raw_sha256']
            or sha(prepared / 'original-preface.cnxml') != auxiliaries['preface']['raw_sha256']
            or (prepared / 'ATTRIBUTION.txt').read_text(encoding='utf-8') != spec['attribution'] + '\n'):
        raise ValueError('Astronomy source attribution changed')
    return spec, manifest


def prepare(curriculum, prepared, out):
    spec, manifest = authenticate_astronomy(prepared)
    source = Path('data/sources-prose-scale-v1.json')
    require_training_spec(source)
    prose = Path(spec['protected_edition']['prepared'])
    verified_book_manifest(prose, source)
    previous_plan = read(curriculum.parent / 'protocol.json')
    if (sha(curriculum) != previous_plan['schedule_sha256']
            or sha(prose / 'manifest.json') != previous_plan['prepared_manifest_sha256']
            or sha(source) != previous_plan['source_spec_sha256']):
        raise ValueError('Prose schedule differs from its declared exposure')
    _, earlier = read_schedule(curriculum)
    if len(earlier) != 4 or len(previous_plan['stages']) != 4:
        raise ValueError('Expected four existing prose stages')
    for stage, planned in zip(earlier, previous_plan['stages']):
        if (sha(stage['source']) != planned['source_sha256']
                or stage['end_update'] != planned['end_update']
                or (float(stage['rate']), stage['scope'], float(stage['answer'])) != (1, 'new', 1)):
            raise ValueError('Existing prose stage changed')
    holdouts = [folder / f'{split}.dat' for folder in (prose, prepared) for split in ('validation', 'test')]
    science = [prepared / stage['file'] for stage in manifest['stages']]
    protect_training([earlier[-1]['content'], *(p.read_bytes() for p in science)],
                     [p.read_bytes() for p in holdouts])
    chunk = previous_plan['chunk']
    if chunk != 128 or [s['id'] for s in manifest['stages']] != [1, 2, 3, 4]:
        raise ValueError('Unexpected chunk or astronomy stage order')
    out.mkdir(parents=True, exist_ok=False)
    end, schedule, rows = earlier[-1]['end_update'], curriculum, []
    for stage, addition in zip(manifest['stages'], science):
        added = documents(addition.read_bytes())
        observations = sum((len(doc) - 1 + chunk - 1) // chunk for doc in added)
        pairs = sum(len(doc) - 1 for doc in added)
        end += observations
        directory = out / f"after-astronomy-{stage['id']}"
        edition = extend(schedule, addition, directory, end, rate_scale=1,
                         scope='new', answer_scale=1, holdouts=holdouts)
        rows.append(dict(curriculum_stage=4 + stage['id'], astronomy_stage=stage['id'],
                         name=stage['name'], documents=len(added), passes=1,
                         source_pairs=pairs, observations=observations, end_update=end,
                         selected_source=addition.as_posix(), selected_sha256=sha(addition),
                         schedule_sha256=edition['schedule_sha256']))
        schedule = directory / 'curriculum.sg'
    for name in ('LICENSE-source.txt', 'ATTRIBUTION.txt', 'original-preface.cnxml'):
        shutil.copyfile(prepared / name, schedule.parent / name)
    _, extended = read_schedule(schedule)
    assert len(extended) == len(earlier) + len(rows)
    for a, b in zip(earlier, extended):
        assert all(a[k] == b[k] for k in ('end_update', 'content', 'rate', 'scope', 'answer'))
    inputs = [curriculum, curriculum.parent / 'protocol.json', source,
              prose / 'manifest.json', prepared / 'manifest.json',
              Path('data/sources-astronomy-v1.json'), Path('reports/astronomy-corpus.json'),
              Path(__file__), Path('scripts/extend_curriculum.py'),
              Path('scripts/corpus/selection.py'), *science, *holdouts]
    protocol = dict(status='prepared_before_any_astronomy_learning',
        parent_schedule=curriculum.as_posix(), parent_schedule_sha256=sha(curriculum),
        curriculum=schedule.as_posix(), curriculum_sha256=sha(schedule), chunk=chunk,
        previous_stages=4, previous_source_observations=earlier[-1]['end_update'], stages=rows,
        added_observations=sum(r['observations'] for r in rows),
        added_source_pairs=sum(r['source_pairs'] for r in rows), end_update=end,
        training_word_occurrences=manifest['outputs']['train']['word_like_units'],
        authenticated_inputs={p.as_posix(): sha(p) for p in inputs},
        existing_stages_unchanged=True, exact_document_and_paragraph_holdout_checks=True,
        no_native_model_work=True, checkpoint_selected=False, reproduction_admitted=False,
        policy='Append one complete pass of each of four selected astronomy stages; preserve all '
               'existing prose stages. Learning rate and ordinary source-byte loss remain unchanged. '
               'Use native resume plus append-only curriculum extension to retain learned state.',
        limits='Topic ordering is not a measured developmental age or mastery schedule. Exact '
               'document and substantial-paragraph checks do not exclude paraphrases or short overlap. '
               'Native admission, new-content learning, retention, and live generation are not yet tested '
               'with this extension. This preparation does not select a model or queue learning.')
    write(out / 'protocol.json', protocol)
    print('Prepared', len(rows), 'new stages:', protocol['added_observations'],
          'source observations,', protocol['added_source_pairs'], 'next-byte targets.', flush=True)
    return protocol


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--curriculum', type=Path, default=Path('runs/prose-scale-curriculum/curriculum.sg'))
    parser.add_argument('--prepared', type=Path, default=Path('data/prepared/astronomy-v1-reviewed'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.curriculum, args.prepared, args.out)
