"""Independently enumerate the continuation and reject altered learning inputs."""
import argparse
from collections import Counter
from functools import partial
from pathlib import Path
import shlex
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from corpus.paired_selection import PairReservations
from corpus.reviewed_edition import require_clear
from development_schedule import PHYSICS_AUDIT, PHYSICS_SOURCES, prepare as prepare_schedule
from native_experiment import read, sha
from prose_founder import write


def audit(edition, out, report, *, physics_sources=PHYSICS_SOURCES,
          physics=Path('data/prepared/physics-v1-reviewed'), physics_audit=PHYSICS_AUDIT):
    prepare = partial(prepare_schedule, physics_sources=physics_sources, physics_audit=physics_audit)
    source_paths = {'prealgebra': Path('data/sources-prealgebra-v1.json'), 'physics': physics_sources}
    revision = read(physics_sources)['version'] == 'selected-physics-v2'
    out.mkdir(parents=True, exist_ok=False)
    plan = read(edition / 'protocol.json')
    assert plan['native_commands'] == 0 and not plan['learning_queued'] and not plan['checkpoints_selected']
    assert all(sha(p) == h for p, h in plan['authenticated_inputs'].items())
    assert all(sha(edition / p) == h for p, h in plan['generated_files'].items())
    assert {p.relative_to(edition).as_posix() for p in edition.rglob('*') if p.is_file()} == {
        'protocol.json', *plan['generated_files']}
    original = Path(plan['parent_schedule'])
    original_rows = [shlex.split(line) for line in original.read_text().splitlines()[1:]]
    assert len(original_rows) == 4 and int(original_rows[-1][0]) == 216289
    end, total_pairs, total_windows, total_docs = 216289, 0, 0, 0
    combined = (original.parent / original_rows[-1][1]).read_bytes()
    corpus_counts, expected_ids, stage_rows = {}, [], []
    for label, prepared in [('prealgebra', Path('data/prepared/prealgebra-worked-v1-pinned')),
                            ('physics', physics)]:
        spec = read(source_paths[label])
        selected = [r for r in spec['sources'] if r['split'] == 'train']
        seen = []
        topic_groups = [[1, 2, 3, 4]] if label == 'prealgebra' else [[n] for n in range(1, 5)]
        for topics in topic_groups:
            chosen = [r['id'] for r in selected if r['stage'] in topics]
            bodies = [(prepared / f'{ident}.txt').read_bytes() for ident in chosen]
            combined += b'\x1e' + b'\x1e'.join(bodies)
            pairs = windows = partial_windows = 0
            for raw in bodies:
                # Enumerate real positions rather than reuse the preparer's ceiling formula.
                cursor = 0
                while cursor < len(raw) - 1:
                    target_end = min(cursor + 128, len(raw) - 1)
                    targets = raw[cursor + 1:target_end + 1]
                    assert targets and len(targets) <= 128
                    pairs += len(targets)
                    windows += 1
                    partial_windows += len(targets) < 128
                    cursor += len(targets)
                assert cursor == len(raw) - 1
            end += windows
            recorded = plan['stages'][len(stage_rows)]
            assert (recorded['edition'], recorded['source_topics'], recorded['documents'], recorded['passes'],
                    recorded['source_pairs'], recorded['observations'], recorded['end_update']) == (
                spec['version'], topics, len(chosen), 1, pairs, windows, end)
            schedule = Path(recorded['curriculum'])
            assert sha(schedule) == recorded['curriculum_sha256']
            rows = [shlex.split(line) for line in schedule.read_text().splitlines()[1:]]
            assert len(rows) == 5 + len(stage_rows)
            for fields, previous in zip(rows[:4], original_rows):
                assert fields[0] == previous[0] and fields[2:] == previous[2:]
                assert (schedule.parent / fields[1]).read_bytes() == (original.parent / previous[1]).read_bytes()
            for fields, old_stage in zip(rows[4:-1], stage_rows):
                assert int(fields[0]) == old_stage['end_update'] and fields[2:] == ['1', 'new', '1']
                assert sha(schedule.parent / fields[1]) == old_stage['cumulative_source_sha256']
            assert rows[-1][0] == str(end) and rows[-1][2:] == ['1', 'new', '1']
            assert (schedule.parent / rows[-1][1]).read_bytes() == combined
            stage_rows.append(dict(edition=spec['version'], source_topics=topics, documents=len(chosen),
                source_pairs=pairs, observations=windows, final_partial_windows=partial_windows,
                end_update=end, cumulative_source_sha256=sha(schedule.parent / rows[-1][1])))
            total_pairs += pairs
            total_windows += windows
            total_docs += len(chosen)
            seen.extend(chosen)
        assert Counter(seen) == Counter(r['id'] for r in selected)
        expected_ids.extend((spec['version'], ident) for ident in seen)
        corpus_counts[spec['version']] = len(seen)
    assert len(set(expected_ids)) == total_docs == 258
    assert (total_windows, total_pairs, end) == (14057, 1783364 if revision else 1783551, 230346)
    assert (plan['added_observations'], plan['added_source_pairs'], plan['end_update']) == (
        total_windows, total_pairs, end)
    assert (plan['final_unique_documents'], plan['final_training_bytes']) == (313, len(combined))
    assert len(combined) == (28680296 if revision else 28680483)
    assert [r['observations'] for r in plan['arithmetic_topics_in_source_order']] == [237, 269, 275, 28]
    conditional = plan['conditional_replay_occupancy']
    assert conditional['capacity'] == 16384
    grouped = conditional['cases']['grouped_arithmetic']
    separated = conditional['cases']['separate_arithmetic_topics']
    assert (grouped['groups'], grouped['retained_prose_windows'], grouped['unused_slots']) == (9, 7284, 1011)
    assert (separated['groups'], separated['retained_prose_windows'], separated['unused_slots']) == (12, 5464, 4651)
    assert grouped['stored_windows'] == [1821] * 4 + [809] + [1820] * 4
    # Source notices accompany each complete textbook handoff, including prior additions.
    attribution_files = 0
    for label, prepared, handoff in [
        ('prealgebra', Path('data/prepared/prealgebra-worked-v1-pinned'), 'after-arithmetic-1'),
        ('physics', physics, 'after-physics-4'),
        ('prealgebra', Path('data/prepared/prealgebra-worked-v1-pinned'), 'after-physics-4'),
    ]:
        spec = read(source_paths[label])
        destination = edition / handoff / 'provenance' / spec['version']
        for name in ('LICENSE-source.txt', 'original-preface.cnxml', 'original-collection.xml',
                     'ATTRIBUTION.txt', 'source-spec.json', 'passage-review.json', 'manifest.json'):
            assert (destination / name).read_bytes() == (prepared / name).read_bytes()
            attribution_files += 1
        if label == 'physics' and revision:
            for name in ('base-source-spec.json', 'base-preparation-audit.json', 'base-manifest.json',
                         'followup-errata.json', 'followup-review.json', 'followup-checks.json'):
                assert (destination / name).read_bytes() == (prepared / name).read_bytes()
                attribution_files += 1

    # Check preflight against changed real editions without touching any archived inputs.
    copies = {}
    for label, folder in [('arithmetic', Path('data/prepared/prealgebra-worked-v1-pinned')),
                          ('physics', physics)]:
        copies[label] = out / label
        shutil.copytree(folder, copies[label])
    rejected = []
    for label, directory in copies.items():
        record = next(r for r in read(directory / 'manifest.json')['sources'] if r['split'] == 'train')
        names = [record['id'] + '.txt', 'train.dat', 'stage-1.dat', 'LICENSE-source.txt',
                 'passage-review.json', 'ATTRIBUTION.txt']
        if label == 'physics' and revision:
            names.extend(('base-source-spec.json', 'base-preparation-audit.json', 'base-manifest.json',
                          'followup-errata.json', 'followup-review.json', 'followup-checks.json'))
        for name in names:
            path = directory / name
            unchanged = path.read_bytes()
            path.write_bytes(unchanged + b'altered')
            destination = out / f'rejected-{label}-{name}'
            try:
                prepare(original, copies['arithmetic'], copies['physics'], destination)
            except ValueError as error:
                assert 'changed' in str(error) and not destination.exists(), str(error)
                rejected.append(label + ':' + name)
            else:
                raise AssertionError('An altered prepared input was accepted')
            finally:
                path.write_bytes(unchanged)

    # Independent negative controls for the exact overlap boundary.
    registry = PairReservations()
    paragraph = 'The reserved worked solution explains the complete sequence of mathematical steps. ' * 3
    heldout = 'Reserved lesson\n\nQuestion: Find 12 - 7.\n\nAnswer: 5\n\n' + paragraph
    registry.add(heldout, 'reserved-arithmetic', paired=True)
    cases = [('whole-text', heldout.replace(' ', '   '), False),
             ('same-question-new-answer', 'Changed title\n\nQuestion: Find 12 - 7.\n\nAnswer: five', True),
             ('answer-paragraph', 'New title\n\n' + paragraph.replace(' ', '  ') + '\n\nNew ending', False)]
    registry.contexts['The reserved object is inside the cupboard.'] = 'reserved-probe'
    cases.append(('probe-context', 'An example: The reserved object is inside the cupboard. Explain.', False))
    for name, text, paired in cases:
        try:
            require_clear(text.encode(), registry, paired=paired)
        except ValueError as error:
            assert 'Reserved content' in str(error)
            rejected.append(name)
        else:
            raise AssertionError('Reserved content was accepted: ' + name)
    for distinct in ('Find 12 + 7.', 'Find 1.2 - 7.'):
        require_clear(('Other\n\nQuestion: ' + distinct + '\n\nAnswer: Example').encode(), registry, paired=True)
    altered_parent = out / 'changed-parent'
    altered_parent.mkdir()
    parent_raw = original.read_bytes()
    (altered_parent / 'curriculum.sg').write_bytes(parent_raw.replace(b'1 new 1', b'0.5 new 1', 1))
    shutil.copyfile(original.parent / 'protocol.json', altered_parent / 'protocol.json')
    destination = out / 'rejected-parent'
    try:
        prepare(altered_parent / 'curriculum.sg', copies['arithmetic'], copies['physics'], destination)
    except ValueError as error:
        assert 'audited exposure' in str(error) and not destination.exists()
        rejected.append('changed-parent-rate')
    else:
        raise AssertionError('Changed parent policy was accepted')
    result = dict(passed=True, protocol=plan, protocol_sha256=sha(edition / 'protocol.json'),
        audit_script_sha256=sha(Path(__file__)), stages=stage_rows, corpus_documents=corpus_counts,
        every_selected_document_once=True, four_original_stages_byte_identical=True,
        complete_target_coverage=True, copied_attribution_files_verified=attribution_files,
        altered_or_reserved_inputs_rejected=rejected, reserved_tests_scored=False,
        native_commands=0, native_admission_verified=False, model_learning_verified=False,
        reproduction_admitted=False)
    write(report, result)
    print('Nine stages, 258 new documents, complete byte coverage and', len(rejected), 'rejections passed.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--edition', type=Path, default=Path('runs/prose-arithmetic-physics-curriculum-v2'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--physics-sources', type=Path, default=PHYSICS_SOURCES)
    parser.add_argument('--physics', type=Path, default=Path('data/prepared/physics-v1-reviewed'))
    parser.add_argument('--physics-audit', type=Path, default=PHYSICS_AUDIT)
    args = parser.parse_args()
    audit(args.edition, args.out, args.report, physics_sources=args.physics_sources,
          physics=args.physics, physics_audit=args.physics_audit)
