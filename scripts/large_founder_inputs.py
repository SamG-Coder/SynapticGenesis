"""Admission and bounded checkpoint facts for the saved 411M prose study."""
import hashlib
from pathlib import Path
import shutil
import struct

from corpus.selection import require_training_spec
from experiment_checkpoint import policy_checkpoint
from native_experiment import read, verified_book_manifest
from prose_founder import SHAPES, file_hash
from prose_retention_inputs import authenticate, require
from prose_size_comparison import counters

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / 'data/prose-411m-v1.json'


def first_stage_pairs(source, observations):
    documents = Path(source).read_bytes().split(b'\x1e')
    require(observations > 0 and all(len(d) >= 2 for d in documents), 'Invalid preflight source windows')
    windows = [min(128, len(d) - 1 - offset) for d in documents for offset in range(0, len(d) - 1, 128)]
    return sum(windows[i % len(windows)] for i in range(observations))


def facts(path):
    meta, extra = policy_checkpoint(Path(path))
    with Path(path).open('rb') as stream:
        stream.seek(256)
        hp = struct.unpack('<8f', stream.read(32))
    return dict(counters=counters(Path(path)), seed=meta[10], batch=meta[5], chunk=meta[6],
                fast=meta[16], hyperparameters=list(hp), graph=extra[8],
                replay_every=extra[2], replay_capacity=extra[3],
                cursor_and_rng=list(meta[19:22]) + [meta[23], meta[25]],
                speech_policy=list(meta[26:30]),
                replay_payload_sha256=hashlib.sha256(struct.pack(f'<{len(extra)}Q', *extra)).hexdigest())


def validate(spec):
    require(spec['version'] == 'prose-411m-v1' and spec['profile'] == '411m' and
            tuple(spec['shape']) == SHAPES['411m'] == (2048, 8192, 8), 'Unexpected 411M shape')
    require((spec['parameters'], spec['spiking_neurons'], spec['seed'], spec['source_observations']) ==
            (411028496, 65536, 1337, 216289), 'Unexpected 411M identity or exposure')
    require((spec['learning_rate'], spec['core_scale'], spec['replay_capacity'], spec['replay_every'],
             spec['save_every']) == (.0003, 1., 16384, 4, 8192), 'Unexpected 411M learning policy')
    require(spec['native_preflight'] == dict(fresh_source_observations=128,
            resumed_source_observations=129, evaluation_batch=16, evaluation_context=128,
            evaluation_batches=2, sample_bytes=64), 'Unexpected native preflight')


def admit():
    spec = read(SPEC)
    validate(spec)
    evaluation_path = Path(spec['evaluation_spec'])
    evaluation = read(evaluation_path)
    source = require_training_spec(evaluation['source_spec'])
    prepared = Path(evaluation['prepared'])
    manifest = verified_book_manifest(prepared, evaluation['source_spec'])
    schedule = Path(evaluation['schedule'])
    exposure_path = schedule.parent / 'protocol.json'
    exposure = read(exposure_path)
    require(file_hash(schedule) == exposure['schedule_sha256'] and
            file_hash(prepared / 'manifest.json') == exposure['prepared_manifest_sha256'] and
            file_hash(evaluation['source_spec']) == exposure['source_spec_sha256'], 'Prose edition changed')
    require([s['end_update'] for s in exposure['stages']] == evaluation['stage_endpoints'] ==
            [8176, 26110, 95207, 216289], 'Curriculum endpoints differ')
    splits = {r['id']: r['split'] for r in manifest['sources']}
    require(all(splits[n] == 'validation' for n in evaluation['validation_books']) and
            all(splits[n] == 'train' for n in evaluation['training_retention_books']), 'Assessment split differs')
    require(evaluation['evaluation'] == dict(batch=16, context=128, batches=32, target_bytes_per_book=65536,
            math='strict FP32', sampling=evaluation['evaluation']['sampling']), 'Assessment geometry differs')
    for stage in exposure['stages']:
        require(file_hash(stage['cumulative_source']) == stage['source_sha256'], 'Scheduled text changed')
    for field in ('runtime', 'capacity_evidence', 'baseline_result', 'early_width_result'):
        require(file_hash(spec[field]) == spec[field + '_sha256'], f'{field} identity differs')
    capacity = read(spec['capacity_evidence'])
    require(capacity['complete'] and capacity['rows'][1]['parameters'] == spec['parameters'], 'Missing 411M capacity result')
    baseline, width = read(spec['baseline_result']), read(spec['early_width_result'])
    require(baseline['complete'] and baseline['same_candidate_replay_history_across_sizes'] and
            not baseline['reserved_tests_scored'], 'Incomplete replay baseline')
    require(width['complete'] and len(width['rows']) == 15 and not width['reserved_tests_scored'], 'Incomplete width study')
    files = [SPEC, ROOT / 'data/training-selection.json', evaluation_path, Path(evaluation['source_spec']),
             prepared / 'manifest.json', prepared / 'source-spec.json', schedule, exposure_path,
             *(prepared / f'{n}.txt' for n in splits), *(Path(s['cumulative_source']) for s in exposure['stages']),
             *(Path(spec[k]) for k in ('runtime', 'capacity_evidence', 'baseline_result', 'early_width_result')),
             *(ROOT / 'scripts' / n for n in ('large_founder_inputs.py', 'prose_large_founder.py',
                 'prose_founder.py', 'checkpoint_assessment.py', 'native_experiment.py', 'experiment_checkpoint.py',
                 'prose_size_comparison.py', 'prose_retention_inputs.py', 'corpus/selection.py', 'audit_binding.py')),
             ROOT / 'tests/large_prose_founder.py',
             *sorted((ROOT / 'src').glob('*.cu')), *sorted((ROOT / 'src').glob('*.cuh'))]
    return dict(specification=spec, evaluation=evaluation, exposure=exposure,
                authenticated_inputs={p.as_posix(): file_hash(p) for p in files})


def storage_budget(out):
    # Four immutable stages, initial, latest and two transactional replacement slots.
    maximum_checkpoint_bytes = 288 + 12 * 411028496 + 4 * 139264 + 8 * (17 + 5 * 4 + 3 * 16384)
    required = 8 * maximum_checkpoint_bytes + 10 * 1024**3
    free = shutil.disk_usage(Path(out).resolve().anchor).free
    require(free >= required, 'Insufficient free disk for preserved 411M checkpoints and replacement reserve')
    return dict(maximum_checkpoint_bytes=maximum_checkpoint_bytes, checkpoint_slots_reserved=8,
                additional_reserve_bytes=10 * 1024**3, required_free_bytes=required,
                available_bytes=free, periodic_latest_replaced=True)
