"""Append reviewed arithmetic and Physics to the unchanged complete prose history."""
import argparse
from pathlib import Path
import shutil

from corpus.reviewed_edition import authenticate, require, require_clear, reservations
from corpus.selection import SELECTION, require_training_spec
from extend_curriculum import documents, prepare as extend, read_schedule
from native_experiment import read, sha, verified_book_manifest
from prose_founder import write


def prose_history(curriculum):
    source, audit_path = Path('data/sources-prose-scale-v1.json'), Path('reports/prose-scale-corpus.json')
    prepared = Path('data/prepared/prose-scale-v1-pinned')
    require_training_spec(source)
    manifest = verified_book_manifest(prepared, source)
    audit, protocol = read(audit_path), read(curriculum.parent / 'protocol.json')
    require(audit['passed'] and sha(source) == audit['source_spec_sha256'] and
            sha(prepared / 'manifest.json') == audit['source_manifest_sha256'] and
            protocol == audit['schedule'] and sha(curriculum) == protocol['schedule_sha256'],
            'Prose schedule differs from the audited exposure')
    _, stages = read_schedule(curriculum)
    require(len(stages) == 4 and protocol['chunk'] == 128, 'Expected four complete prose stages')
    for stage, declared in zip(stages, protocol['stages']):
        require(sha(stage['source']) == declared['source_sha256'] and
                stage['end_update'] == declared['end_update'] and
                (float(stage['rate']), stage['scope'], float(stage['answer'])) == (1, 'new', 1),
                'Previous prose stage changed')
    inputs = [source, audit_path, prepared / 'manifest.json', prepared / 'source-spec.json',
              curriculum, curriculum.parent / 'protocol.json', *[s['source'] for s in stages]]
    inputs.extend(prepared / f"{r['id']}.txt" for r in manifest['sources'])
    inputs.extend(prepared / f'{split}.dat' for split in manifest['outputs'])
    return stages, protocol, inputs


def prepare(curriculum, arithmetic, physics, out):
    require(not out.exists(), 'Use a fresh continuation directory')
    earlier, parent, inputs = prose_history(curriculum)
    editions = [
        authenticate('data/sources-prealgebra-v1.json', 'reports/prealgebra-worked-selection.json',
                     arithmetic, paired=True),
        authenticate('data/sources-physics-v1.json', 'reports/physics-selection.json', physics),
    ]
    registry, protected = reservations(editions)
    require_clear(earlier[-1]['content'], registry)
    seen = set(documents(earlier[-1]['content']))
    for edition in editions:
        raw = (edition.prepared / 'train.dat').read_bytes()
        require_clear(raw, registry, paired=edition.paired)
        added = documents(raw)
        require(len(set(added)) == len(added) and not seen.intersection(added),
                'New curriculum repeats an exact training document')
        seen.update(added)
        inputs.extend(edition.inputs)
    inputs.extend(Path(r['path']) for r in protected)
    inputs.extend([SELECTION, Path(__file__), Path('scripts/corpus/reviewed_edition.py'),
                   Path('scripts/corpus/paired_selection.py'), Path('scripts/corpus/selection.py'),
                   Path('scripts/extend_curriculum.py'), Path('scripts/native_experiment.py'),
                   Path('scripts/prose_founder.py'), Path('src/stage_replay.cuh')])
    arithmetic_topics = []
    for stage in editions[0].manifest['stages']:
        added = documents((arithmetic / stage['file']).read_bytes())
        arithmetic_topics.append(dict(topic=stage['id'], name=stage['name'], documents=len(added),
            observations=sum((len(d) - 1 + parent['chunk'] - 1) // parent['chunk'] for d in added)))
    require(b'\x1e'.join((arithmetic / s['file']).read_bytes() for s in editions[0].manifest['stages']) ==
            (arithmetic / 'train.dat').read_bytes(), 'Arithmetic topic order differs from its training stream')
    identities = {p.as_posix(): sha(p) for p in inputs}
    # All authentication, overlap and duplicate checks finish before any output.
    out.mkdir(parents=True)
    schedule, end, rows = curriculum, earlier[-1]['end_update'], []
    holdouts = [Path(r['path']) for r in protected if r['kind'] == 'documents']
    chunk = parent['chunk']
    for label, edition in zip(('arithmetic', 'physics'), editions):
        topics = [dict(id=1, name='Reviewed arithmetic worked examples', file='train.dat',
                       source_topics=[1, 2, 3, 4])] if label == 'arithmetic' else [
            dict(stage, source_topics=[stage['id']]) for stage in edition.manifest['stages']]
        for stage in topics:
            source = edition.prepared / stage['file']
            added = documents(source.read_bytes())
            pairs = sum(len(doc) - 1 for doc in added)
            observations = sum((len(doc) - 1 + chunk - 1) // chunk for doc in added)
            end += observations
            directory = out / f"after-{label}-{stage['id']}"
            extension = extend(schedule, source, directory, end, rate_scale=1, scope='new',
                               answer_scale=1, holdouts=holdouts)
            rows.append(dict(curriculum_stage=len(earlier) + len(rows) + 1,
                edition=edition.spec['version'], source_topics=stage['source_topics'], name=stage['name'],
                documents=len(added), passes=1, source_pairs=pairs, observations=observations,
                end_update=end, selected_source=source.as_posix(), selected_sha256=sha(source),
                curriculum=(directory / 'curriculum.sg').as_posix(),
                curriculum_sha256=extension['schedule_sha256']))
            schedule = directory / 'curriculum.sg'
        # Each handoff has its own source notices, including the earlier addition.
        for included in editions[:1 if label == 'arithmetic' else 2]:
            destination = schedule.parent / 'provenance' / included.spec['version']
            destination.mkdir(parents=True)
            for name, source in included.provenance.items():
                shutil.copyfile(source, destination / name)
    _, final = read_schedule(schedule)
    require(all(all(a[k] == b[k] for k in ('content', 'end_update', 'rate', 'scope', 'answer'))
                for a, b in zip(earlier, final)), 'Existing prose history changed')
    require_clear(final[-1]['content'], registry)
    require(all(sha(path) == digest for path, digest in identities.items()),
            'A source changed during continuation preparation')
    outputs = {p.relative_to(out).as_posix(): sha(p) for p in sorted(out.rglob('*')) if p.is_file()}
    # Conditional occupancy follows StageReplayView::validate, not a learned-model result.
    prior_windows = [r['passes'] * r['observations_per_pass'] for r in parent['stages']]
    physics_windows = [r['observations'] for r in rows[1:]]
    occupancy = {}
    for name, counts in (
        ('grouped_arithmetic', prior_windows + [rows[0]['observations']] + physics_windows),
        ('separate_arithmetic_topics', prior_windows + [r['observations'] for r in arithmetic_topics] + physics_windows),
    ):
        quotas = [16384 // len(counts) + int(i < 16384 % len(counts)) for i in range(len(counts))]
        stored = [min(count, quota) for count, quota in zip(counts, quotas)]
        occupancy[name] = dict(groups=len(counts), quotas=quotas, stored_windows=stored,
                              retained_prose_windows=sum(stored[:4]), unused_slots=16384 - sum(stored))
    result = dict(version='arithmetic-physics-continuation-v2',
        status='prepared_before_learning', parent_schedule=curriculum.as_posix(),
        parent_schedule_sha256=sha(curriculum), previous_stages=len(earlier),
        previous_observations=earlier[-1]['end_update'], chunk=chunk,
        curriculum=schedule.as_posix(), curriculum_sha256=sha(schedule), stages=rows,
        added_observations=sum(r['observations'] for r in rows),
        added_source_pairs=sum(r['source_pairs'] for r in rows), end_update=end,
        added_training_documents=sum(r['documents'] for r in rows),
        added_training_word_occurrences=sum(e.manifest['outputs']['train']['word_like_units'] for e in editions),
        final_unique_documents=len(documents(final[-1]['content'])), final_training_bytes=len(final[-1]['content']),
        arithmetic_topics_in_source_order=arithmetic_topics,
        conditional_replay_occupancy=dict(capacity=16384, cases=occupancy,
            assumption='Completed prose founder and one pass of every addition under existing stage replay. '
                       'Computed slot counts only; no RNG, replay trajectory or quality simulation. '
                       'This comparison does not select a checkpoint or its replay policy.'),
        protected_inputs=protected, authenticated_inputs=identities, generated_files=outputs,
        checkpoints_selected=False, learning_queued=False, native_commands=0, reproduction_admitted=False,
        policy='Preserve all four prose stages; introduce every arithmetic lesson once in one native '
               'stage, retaining its four-topic source order, followed by four Physics stages. '
               'Rate scale 1, new-document '
               'scope, ordinary answer weight 1. Later native continuation must resume full state.',
        limits=['Five new native stages are not biological ages, measured prerequisites or mastery gates.',
                'No learning, quality, retention, native admission or optimal exposure is established.',
                'Stage-balanced replay will redistribute its fixed budget across nine groups; this '
                'proposal does not claim an unchanged distribution or specify a new replay policy.',
                'Exact whole-text, question, substantial-paragraph and authored-context checks do not '
                'exclude paraphrases, short overlap or changed-number templates.',
                'Astronomy remains a separate unconsumed curriculum proposal; it is not included here.',
                'Source licences and original attribution accompany both textbook handoffs.'])
    write(out / 'protocol.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--curriculum', type=Path, default=Path('runs/prose-scale-curriculum/curriculum.sg'))
    parser.add_argument('--arithmetic', type=Path, default=Path('data/prepared/prealgebra-worked-v1-pinned'))
    parser.add_argument('--physics', type=Path, default=Path('data/prepared/physics-v1-reviewed'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.curriculum, args.arithmetic, args.physics, args.out)
    print('Prepared', len(result['stages']), 'new stages:', result['added_observations'],
          'source observations,', result['added_source_pairs'], 'next-byte targets.')
