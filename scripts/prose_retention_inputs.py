"""Admission, fixed exposure and bounded checkpoint evidence for the LR study."""
import hashlib
from pathlib import Path
import struct

from corpus.selection import SELECTION, require_training_spec, selection
from native_experiment import read, verified_book_manifest
from prose_founder import file_hash
from prose_size_comparison import counters


ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / 'data/prose-retention-lr-v1.json'
COUNTERS = ('online_updates', 'global_updates', 'observed_pairs', 'generated_bytes',
            'replay_updates', 'replay_pairs')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def authenticate(pins):
    for name, expected in pins.items():
        require(file_hash(name) == expected, f'Experiment input changed: {name}')


def state(path):
    """Bounded diagnostic read; native loading checks the complete payload."""
    path = Path(path)
    with path.open('rb') as stream:
        meta = struct.unpack('<32Q', stream.read(256))
        hp = struct.unpack('<8f', stream.read(32))
        require(meta[0] == 0x314d4c53434e5042 and meta[1] == 6 and meta[17] == 5,
                'Expected an associative ordinary stage-replay checkpoint')
        require(17 <= meta[31] <= 16384, 'Unexpected replay payload size')
        start = 288 + 12 * meta[14] + 4 * meta[18]
        require(path.stat().st_size == start + 8 * meta[31], 'Unexpected checkpoint extent')
        stream.seek(start)
        extra_raw = stream.read(8 * meta[31])
        extra = struct.unpack(f'<{meta[31]}Q', extra_raw)
    require(extra[1] == 3 and extra[9] == 0, 'Expected uniform grouped replay without consolidation')
    return dict(counters=counters(path), hyperparameters=list(hp), cell=meta[1], seed=meta[10],
                batch=meta[5], chunk=meta[6], fast=meta[16],
                cursor_and_rng=list(meta[19:22]) + [meta[23], meta[25]],
                speech_policy=list(meta[26:30]), graph=extra[8], replay_every=extra[2],
                replay_capacity=extra[3], replay_payload_sha256=hashlib.sha256(extra_raw).hexdigest())


def expected_exposure(spec, exposure, source, prepared):
    """Count each real next-byte window, including every short document tail."""
    selected = [r['id'] for r in source['sources'] if r['split'] == 'train' and r['stage'] == 4]
    windows = []
    for ident in selected:
        length = (prepared / f'{ident}.txt').stat().st_size
        windows.extend(min(exposure['chunk'], length - 1 - offset)
                       for offset in range(0, length - 1, exposure['chunk']))
    parent = exposure['stages'][2]['end_update']
    require(parent == spec['starting_observations'], 'Starting exposure differs')
    require(len(windows) == exposure['stages'][3]['observations_per_pass'], 'Stage window count differs')
    prior_pairs = sum(s['source_pairs_per_pass'] * s['passes'] for s in exposure['stages'][:3])
    require(prior_pairs + sum(windows) == exposure['source_pairs'], 'Source byte exposure differs')
    require(spec['endpoints'] == [parent + len(windows) // 2, exposure['online_updates']],
            'Expected halfway and complete final-stage endpoints')
    expected = {}
    for end in [parent, *spec['endpoints']]:
        expected[str(end)] = dict(online_updates=end, global_updates=end + end // 4,
            observed_pairs=prior_pairs + sum(windows[:end - parent]),
            generated_bytes=(end // 500) * 96, replay_updates=end // 4)
    return selected, expected


def admit(spec):
    """Admit only the reviewed prose parent, schedule and bounded diagnostic use."""
    require(spec['version'] == 'prose-retention-lr-v1' and spec['profile'] == '105m' and
            spec['seed'] == 1337 and spec['arms'] == [
                {'name': 'resumed-control', 'learning_rate': .0003},
                {'name': 'quarter-rate', 'learning_rate': .000075}], 'Unexpected v1 comparison arms')
    evaluation = read(spec['evaluation_spec'])
    source_path = Path(evaluation['source_spec'])
    source = require_training_spec(source_path)
    prepared, schedule = Path(evaluation['prepared']), Path(spec['schedule'])
    manifest = verified_book_manifest(prepared, source_path)
    exposure_path = schedule.parent / 'protocol.json'
    exposure = read(exposure_path)
    require(Path(evaluation['schedule']) == schedule, 'Assessment and learning schedules differ')
    require(file_hash(schedule) == exposure['schedule_sha256'], 'Source schedule changed')
    require(file_hash(prepared / 'manifest.json') == exposure['prepared_manifest_sha256'], 'Edition changed')
    require(file_hash(source_path) == exposure['source_spec_sha256'], 'Source selection changed')
    for stage in exposure['stages']:
        require(file_hash(stage['cumulative_source']) == stage['source_sha256'], 'Scheduled text changed')
    candidates = [r for r in selection().get('prose_diagnostic_bases', [])
                  if r['sha256'] == spec['checkpoint_sha256']]
    require(len(candidates) == 1, 'Parent is not uniquely admitted for this prose diagnostic')
    admitted = candidates[0]
    for key, value in [('profile', spec['profile']), ('seed', spec['seed']),
                       ('online_updates', spec['starting_observations']),
                       ('source_edition', source['version']), ('schedule_sha256', file_hash(schedule))]:
        require(admitted[key] == value, f'Parent admission differs: {key}')
    require(Path(admitted['checkpoint']) == Path(spec['checkpoint']) and
            Path(admitted['schedule']) == schedule and
            Path(admitted['published_evidence']) == Path(spec['original_result']), 'Parent admission path differs')
    require(file_hash(spec['original_result']) == admitted['published_evidence_sha256'],
            'Published parent evidence changed')
    published = read(spec['original_result'])
    require(published['complete_prose_learning'] and published['random_initialization'] and
            not published['imported_weights'] and not published['reserved_tests_scored'],
            'Original run provenance differs')
    parent_rows = [r for r in published['assessments'] if r['stage'] == 3]
    require(len(parent_rows) == 1, 'Published stage-three assessment missing')
    parent_row = parent_rows[0]
    require(parent_row['checkpoint_sha256'] == spec['checkpoint_sha256'], 'Published parent identity differs')
    require(file_hash(spec['checkpoint']) == spec['checkpoint_sha256'], 'Parent checkpoint changed')
    original = published['completion']
    require(file_hash(original['checkpoint']) == original['checkpoint_sha256'], 'Original final checkpoint changed')
    selected, expected = expected_exposure(spec, exposure, source, prepared)
    observed = state(spec['checkpoint'])
    require([observed['counters'][k] for k in ('channels', 'hidden', 'layers')] ==
            evaluation['profiles'][spec['profile']], 'Parent model shape differs')
    require(observed['seed'] == spec['seed'] == evaluation['seed'] and
            observed['batch'] == 1 and observed['chunk'] == 128 and observed['fast'] == 1 and
            observed['graph'] == 1 and observed['replay_every'] == 4 and observed['replay_capacity'] == 1024,
            'Parent live policy differs')
    require(observed['speech_policy'][:3] == [500, 96, 40], 'Parent speech policy differs')
    initial_rate = struct.unpack('<f', struct.pack('<f', spec['arms'][0]['learning_rate']))[0]
    require(observed['hyperparameters'][0] == observed['hyperparameters'][7] == initial_rate,
            'Control rate differs from parent rate')
    require(observed['hyperparameters'][6] == 1, 'Parent core-rate multiplier differs')
    require(all(observed['counters'][k] == v for k, v in expected[str(spec['starting_observations'])].items()),
            'Parent source exposure differs')
    splits = {r['id']: r['split'] for r in manifest['sources']}
    require(all(splits[n] == 'validation' for n in evaluation['validation_books']) and
            all(splits[n] == 'train' for n in evaluation['training_retention_books']), 'Assessment split differs')
    additional = [selected[0], selected[-1]]
    require(len(set(evaluation['validation_books'] + evaluation['training_retention_books'] + additional)) == 8,
            'Diagnostic books are not distinct')
    paths = [SPEC, SELECTION, Path(spec['evaluation_spec']), Path(spec['original_result']),
             Path(spec['checkpoint']), Path(original['checkpoint']), source_path,
             prepared / 'manifest.json', prepared / 'source-spec.json', schedule, exposure_path,
             Path('build/synapticgenesis.exe'), Path('runs/prose-size-panel/protocol.json'),
             Path('runs/prose-spike-panel/protocol.json'),
             *(prepared / f'{n}.txt' for n in splits),
             *(Path(s['cumulative_source']) for s in exposure['stages'])]
    return dict(specification=spec, evaluation=evaluation, exposure=exposure,
                admission=admitted, expected_exposure=expected, parent_state=observed,
                parent_assessment=parent_row, original_completion=original,
                additional_books=additional, authenticated_inputs={p.as_posix(): file_hash(p) for p in paths})


def matched(left, right):
    require(all(left['counters'][k] == right['counters'][k] for k in COUNTERS), 'Exposure counters differ')
    require(left['replay_payload_sha256'] == right['replay_payload_sha256'], 'Replay state differs')
    require(left['cursor_and_rng'] == right['cursor_and_rng'], 'Source cursor or speech RNG differs')
    for name in ('seed', 'batch', 'chunk', 'fast', 'speech_policy', 'graph', 'replay_every', 'replay_capacity'):
        require(left[name] == right[name], f'Live policy differs: {name}')
    require(left['hyperparameters'][1:7] == right['hyperparameters'][1:7], 'Other optimizer policy differs')


def predecessors(comparison_path, spike_path):
    comparison, spike = read(comparison_path), read(spike_path)
    require(comparison['complete'] and comparison['matched_source_replay_and_speech_counters'] and
            not comparison['reserved_tests_scored'] and len(comparison['rows']) == 12,
            'Matched size comparison is incomplete')
    require({(r['profile'], r['stage']) for r in comparison['rows']} ==
            {(p, s) for p in ('2m', '27m', '105m') for s in (1, 2, 3, 4)}, 'Size endpoints differ')
    require(spike['complete'] and spike['compatibility_trace_byte_identical'] and
            spike['all_checkpoints_unchanged'] and spike['learning_updates'] == 0 and
            not spike['reserved_tests_scored'] and spike['native_commands'] == 73,
            'Matched neuron panel is incomplete')
    require({(r['profile'], r['stage']) for r in spike['records']} ==
            {(p, s) for p in ('2m', '27m', '105m') for s in (0, 1, 4)} and
            len(spike['records']) == 9, 'Neuron panel endpoints differ')
    require(spike['comparison_sha256'] == file_hash(comparison_path), 'Neuron panel comparison differs')
    return {p.as_posix(): file_hash(p) for p in (comparison_path, spike_path)}
