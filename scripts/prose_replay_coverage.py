"""Reconstruct actual replay coverage from immutable native prose checkpoints."""
import argparse
from collections import Counter
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tests'))
from stage_replay_reference import ReplayReference
from corpus.selection import require_training_spec
from experiment_checkpoint import policy_checkpoint
from native_experiment import read, verified_book_manifest
from prose_founder import file_hash, write


def unique_counts(episodes, lengths, first, end, chunk):
    distinct = set(episodes)
    for document, offset, length in distinct:
        if not (first <= document < end and offset % chunk == 0 and
                0 <= offset < lengths[document] - 1 and
                length == min(chunk, lengths[document] - 1 - offset)):
            raise ValueError('Replay descriptor is not an observed-source chunk')
    return dict(distinct_windows=len(distinct), distinct_target_pairs=sum(e[2] for e in distinct),
                covered_documents=sorted({e[0] for e in distinct}))


def audit(profiles, out):
    source_path = Path('data/sources-prose-scale-v1.json')
    source = require_training_spec(source_path)
    prepared = Path('data/prepared/prose-scale-v1-pinned')
    verified_book_manifest(prepared, source_path)
    schedule_path = Path('runs/prose-scale-curriculum/curriculum.sg')
    schedule = read(schedule_path.parent / 'protocol.json')
    if file_hash(schedule_path) != schedule['schedule_sha256']:
        raise ValueError('Source schedule changed')
    books = [b for stage in (1, 2, 3, 4) for b in source['sources']
             if b['split'] == 'train' and b['stage'] == stage]
    source_file = Path(schedule['stages'][-1]['cumulative_source'])
    docs = source_file.read_bytes().split(b'\x1e')
    if len(docs) != len(books) or any(raw != (prepared / f'{book["id"]}.txt').read_bytes()
                                     for raw, book in zip(docs, books)):
        raise ValueError('Book identities do not reconstruct the actual curriculum document order')
    lengths = list(map(len, docs))
    inputs = [source_path, prepared / 'manifest.json', schedule_path, schedule_path.parent / 'protocol.json', source_file]
    pins = {p.as_posix(): file_hash(p) for p in inputs}
    rows = []
    for profile in profiles:
        directory = Path('runs/prose-105m-founder') if profile == '105m' else Path(f'runs/prose-size-panel/founder-{profile}')
        founder_path = directory / 'protocol.json'
        founder = read(founder_path)
        if (founder['profile'] != profile or not founder['random_initialization'] or founder['imported_weights'] or
                founder['source_schedule'] != schedule or founder['seed'] != 1337):
            raise ValueError('Founder provenance or source schedule differs')
        pins[founder_path.as_posix()] = file_hash(founder_path)
        paths = [directory / f'stage-{stage}.ckpt' for stage in (3, 4)]
        for stage, path in zip((3, 4), paths):
            assessed = Path(f'runs/prose-size-panel/{profile}-stage-{stage}/result.json')
            assessment = read(assessed)
            identity = file_hash(path)
            if identity != assessment['checkpoint_sha256']:
                raise ValueError('Checkpoint differs from native assessment')
            pins[path.as_posix()] = identity
            pins[assessed.as_posix()] = file_hash(assessed)
        initial, before = policy_checkpoint(paths[0])
        completed, after = policy_checkpoint(paths[1])
        if (initial[24], completed[24], initial[6], before[2], before[3]) != (95207, 216289, 128, 4, 1024):
            raise ValueError('Unexpected stage exposure or replay policy')
        reference = ReplayReference(paths[0], before[2], before[14], completed[12], lengths)
        selected = Counter()
        event_count = 0

        def observe(observation, episode):
            nonlocal event_count
            if observation % before[2] or not initial[24] < observation <= completed[24]:
                raise ValueError('Replay occurred outside its declared cadence')
            selected[episode] += 1
            event_count += 1

        reference.run_until(completed[24], on_replay=observe)
        exact = reference.matches(paths[1])
        # Existing callers without the observer must retain the same state.
        unobserved = ReplayReference(paths[0], before[2], before[14], completed[12], lengths)
        unobserved.run_until(completed[24])
        unobserved.matches(paths[1])
        groups, first = [], 0
        for index, group in enumerate(reference.groups):
            historical_updates = before[20 + 5 * index] if index < before[16] else 0
            historical_pairs = before[21 + 5 * index] if index < before[16] else 0
            events = {e: count for e, count in selected.items() if first <= e[0] < group.end}
            if sum(events.values()) != group.updates - historical_updates or sum(
                    e[2] * count for e, count in events.items()) != group.pairs - historical_pairs:
                raise ValueError('Recorded replay events differ from native stage counter deltas')
            available_windows = sum((lengths[d] - 2) // 128 + 1 for d in range(first, group.end))
            available_pairs = sum(lengths[d] - 1 for d in range(first, group.end))
            pool = unique_counts(group.items, lengths, first, group.end, 128)
            replayed = unique_counts(events, lengths, first, group.end, 128)
            book_rows = []
            for d in range(first, group.end):
                per_book = {e: n for e, n in events.items() if e[0] == d}
                book_rows.append(dict(book=books[d]['id'], document=d,
                    stored_slots=sum(e[0] == d for e in group.items),
                    stored_distinct_windows=len({e for e in group.items if e[0] == d}),
                    final_stage_replay_updates=sum(per_book.values()),
                    final_stage_distinct_replayed_windows=len(per_book)))
            groups.append(dict(stage=index + 1, documents=group.end - first,
                available_unique_source_windows=available_windows, available_source_target_pairs=available_pairs,
                final_stored_slots=len(group.items), final_stored=pool,
                final_stage_replay_updates=sum(events.values()),
                final_stage_replay_pairs=sum(e[2] * n for e, n in events.items()),
                final_stage_replayed=replayed,
                replayed_window_fraction=replayed['distinct_windows'] / available_windows,
                replayed_target_fraction=replayed['distinct_target_pairs'] / available_pairs,
                mean_presentations_per_replayed_window=sum(events.values()) / len(events),
                max_presentations_of_one_window=max(events.values()), books=book_rows))
            first = group.end
        if event_count != completed[24] // 4 - initial[24] // 4 or event_count != after[6] - before[6]:
            raise ValueError('Reconstructed total replay count differs')
        row = dict(profile=profile, parameters=completed[14], seed=initial[10],
            starting_observations=initial[24], ending_observations=completed[24],
            reconstructed_replay_events=event_count, **exact,
            observer_does_not_change_policy=True, groups=groups)
        rows.append(row)
        print(profile, 'exact replay reconstruction:', event_count, 'events; distinct windows by group:',
              [g['final_stage_replayed']['distinct_windows'] for g in groups], flush=True)
    if any(row['groups'] != rows[0]['groups'] for row in rows[1:]):
        raise ValueError('Matched model sizes received different replay coverage')
    if any(file_hash(name) != digest for name, digest in pins.items()):
        raise ValueError('Replay evidence changed during inspection')
    code = [Path(__file__), Path('tests/stage_replay_reference.py'), Path('scripts/experiment_checkpoint.py'),
            Path('src/stage_replay.cuh'), Path('src/live.cuh'), Path('src/live_replay.cuh')]
    write(out, dict(complete=True, kind='Exploratory CPU reconstruction of actual last-stage replay',
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        inputs_sha256=pins, code_sha256={p.as_posix(): file_hash(p) for p in code}, rows=rows,
        native_commands=0, model_updates=0, checkpoint_files_unchanged=True, reserved_tests_scored=False,
        actual_document_order_verified=True, equal_replay_coverage_across_profiles=True,
        limits='Exact source/replay/RNG reconstruction, not a simulation of neuron dynamics or language quality. '
               'Distinctness counts document/offset/length descriptors, not semantic facts or unique strings. '
               'Final stored windows differ from all windows replayed over time; current-stage pools change. '
               'Absent final slots do not imply a book was never learned or replayed. All new books were '
               'observed through the native source stream. Greater coverage may or may not improve retention. '
               'This report does not test a larger reservoir, full-history sampling or a new replay cadence.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--profiles', nargs='+', choices=('2m', '27m', '105m'), default=['2m', '105m'])
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or len(args.profiles) != len(set(args.profiles)):
        parser.error('Use distinct profiles and a fresh output path')
    audit(args.profiles, args.out)
