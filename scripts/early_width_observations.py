"""Independent CPU checks of sampled native updates and their exposure labels."""
from pathlib import Path
import json
import math
import struct

import numpy as np

from native_experiment import read
from prose_founder import file_hash
from prose_projection_snapshot import layout
from prose_retention_inputs import require


SPEECH_PROMPT = b'The bird '
SPEECH_EVERY = 500
SPEECH_GENERATED_BYTES = 96


def speech_records(raw, end):
    """The probe saves prompt + completion; native counters count completion."""
    records = end // SPEECH_EVERY
    width = len(SPEECH_PROMPT) + SPEECH_GENERATED_BYTES
    require(len(raw) == records * width, 'Speech output extent differs')
    frames = [raw[at:at + width] for at in range(0, len(raw), width)]
    require(all(frame.startswith(SPEECH_PROMPT) for frame in frames), 'Speech prompt differs')
    return frames


def match_guard_speech(control_directory, observed_directory, end):
    """Compare every speech byte and production marker without decoding text."""
    frames = speech_records((Path(observed_directory) / 'speech.bin').read_bytes(), end)
    expected = bytearray(b'\n[session starts at online update 0]\n')
    for index, frame in enumerate(frames, 1):
        at = index * SPEECH_EVERY
        expected.extend(f'\n[online update {at}; global update {at + at // 4}]\n'.encode('ascii'))
        expected.extend(frame)
        expected.extend(b'\n')
    require((Path(control_directory) / 'transcript.txt').read_bytes() == expected,
            'Native guard speech transcript differs')
    return dict(speech_records=len(frames), prompt_bytes_per_record=len(SPEECH_PROMPT),
                generated_bytes_per_record=SPEECH_GENERATED_BYTES, raw_transcript_byte_identical=True)


def source_windows(data_path, end, chunk=128):
    docs = Path(data_path).read_bytes().split(b'\x1e')
    require(all(len(d) >= 2 for d in docs), 'Invalid source document')
    windows = [(i, at, min(chunk, len(d) - 1 - at)) for i, d in enumerate(docs)
               for at in range(0, len(d) - 1, chunk)]
    return [windows[i % len(windows)] for i in range(end)]


def moment(values):
    values = np.asarray(values, dtype=np.float64)
    require(values.size > 0 and np.isfinite(values).all(), 'Invalid sampled values')
    return dict(count=int(values.size), mean_absolute=float(np.abs(values).mean()),
                rms=float(np.sqrt(np.mean(values * values))), maximum_absolute=float(np.abs(values).max()))


def compare_moment(actual, values):
    expected = moment(values)
    require(actual.keys() == expected.keys(), 'Moment fields differ')
    require(actual['count'] == expected['count'], 'Moment sample count differs')
    for key in ('mean_absolute', 'rms', 'maximum_absolute'):
        require(math.isclose(actual[key], expected[key], rel_tol=3e-13, abs_tol=1e-15),
                f'Sampled {key} differs from raw coordinates')


def audit(directory, case, seed, end, points, source_path):
    directory = Path(directory)
    result, plan = read(directory / 'result.json'), read(directory / 'coordinates.json')
    require(result['complete'] and result['shared_production_live_tick'] and
            not result['generated_text_targets'] and not result['reserved_tests_scored'], 'Probe is incomplete')
    for key in ('channels', 'hidden', 'layers', 'core_scale'):
        require(result[key] == case[key], f'Probe {key} differs')
    require(result['seed'] == seed and result['source_updates'] == end, 'Probe seed/exposure differs')
    windows = source_windows(source_path, end)
    require(result['source_pairs'] == sum(w[2] for w in windows) and
            result['global_updates'] == end + end // 4 and result['replay_updates'] == end // 4 and
            result['generated_bytes'] == end // 500 * 96, 'Probe counters differ')
    rows = [json.loads(line) for line in (directory / 'observations.jsonl').read_text().splitlines()]
    require([r['source_update'] for r in rows] == points and result['observations'] == len(points),
            'Observation points differ')
    blocks, count = layout(case['channels'], case['hidden'], case['layers'])
    c = case['channels']
    fields = [(0, 256 * c, (256, c)), *(field for block in blocks for field in block),
              (count - c - 256 * c - 256, count - 256 * c - 256, (c,)),
              (count - 256 * c - 256, count - 256, (256, c)), (count - 256, count, (256,))]
    require(result['parameters'] == count and len(plan['roles']) == len(fields) and
            plan['sample_limit_per_tensor'] == 64, 'Coordinate model layout differs')
    position = 0
    for role, (first, last, _) in zip(plan['roles'], fields):
        n = min(64, last - first)
        offsets = [first if n == 1 else first + i * (last - first - 1) // (n - 1) for i in range(n)]
        require((role['tensor_begin'], role['tensor_size'], role['sample_begin'], role['sample_size']) ==
                (first, last - first, position, n), 'Sampled role extent differs')
        require(plan['offsets'][position:position + n] == offsets, 'Sampled coordinates differ')
        position += n
    require(position == len(plan['offsets']), 'Unexpected extra sampled coordinates')

    def values(name):
        data = np.fromfile(directory / name, dtype='<f4')
        require(data.size == position and np.isfinite(data).all(), 'Malformed raw coordinate file')
        return data.astype(np.float64)

    initial = values('initial-coordinates.f32')
    with (directory / 'initial.ckpt').open('rb') as stream:
        header = struct.unpack('<32Q', stream.read(256))
        require(header[14] == count and header[24] == 0, 'Initial checkpoint differs')
        for i, offset in enumerate(plan['offsets']):
            stream.seek(288 + 4 * offset)
            require(struct.unpack('<f', stream.read(4))[0] == initial[i], 'Initial sampled coordinate differs')
    for row in rows:
        at = row['source_update']
        require((row['document'], row['byte_offset'], row['target_bytes']) == windows[at - 1], 'Source window label differs')
        require(row['source_global_update'] == at + (at - 1) // 4 and
                row['global_updates_after_tick'] == at + at // 4, 'Source/replay update boundary differs')
        require(math.isfinite(row['source_loss']) and math.isfinite(row['source_gradient_norm_before_clip']),
                'Nonfinite native source metric')
        norm = row['source_gradient_norm_before_clip']
        require(norm >= 0 and math.isclose(row['source_clip_factor'], min(1., 1. / norm) if norm else 1.,
                                          rel_tol=1e-7, abs_tol=1e-8), 'Clip factor differs')
        require(row['generated_bytes_this_tick'] == (96 if at % 500 == 0 else 0), 'Speech boundary differs')
        before, after, gradient = [values(f'source-{at}-{name}.f32') for name in ('before', 'after', 'clipped-gradient')]
        require(np.linalg.norm(gradient) <= min(norm, 1.) + 2e-6, 'Sampled clipped gradient exceeds the full norm')
        require(len(row['roles']) == len(plan['roles']), 'Missing role measurements')
        for measured, role in zip(row['roles'], plan['roles']):
            require(measured['name'] == role['name'], 'Measured role order differs')
            selected = slice(role['sample_begin'], role['sample_begin'] + role['sample_size'])
            for name, data in [('weights_before', before), ('weights_after', after),
                               ('source_update_delta', after - before), ('displacement_from_initial', after - initial),
                               ('clipped_gradient', gradient)]:
                compare_moment(measured[name], data[selected])
        require(len(row['pre_update_forward_layers']) == case['layers'], 'Missing layer activity')
        for layer in row['pre_update_forward_layers']:
            require(layer['neuron_positions'] == row['target_bytes'] * case['hidden'], 'Activity size differs')
            require(all(0 <= layer[k] <= 1 for k in ('spike_event_fraction', 'zero_local_spike_surrogate_fraction')),
                    'Invalid activity fraction')
        if at == end and end % 4:
            with (directory / 'latest.ckpt').open('rb') as stream:
                for i, offset in enumerate(plan['offsets']):
                    stream.seek(288 + 4 * offset)
                    require(struct.unpack('<f', stream.read(4))[0] == after[i], 'Final sampled coordinate differs')
    require(result['source_gradient_norm_before_clip']['count'] == end and
            0 <= result['source_updates_clipped'] <= end, 'Source gradient census differs')
    files = [directory / 'coordinates.json', directory / 'observations.jsonl',
             directory / 'result.json', directory / 'initial-coordinates.f32', directory / 'speech.bin',
             *(directory.glob('source-*.f32'))]
    frames = speech_records((directory / 'speech.bin').read_bytes(), end)
    require(len(frames) * SPEECH_GENERATED_BYTES == result['generated_bytes'], 'Speech completion counter differs')
    return dict(passed=True, source_update_points=points, sampled_coordinates=position,
                raw_coordinate_files=1 + 3 * len(points), native_source_updates=end,
                speech_records=len(frames), prompt_bytes_per_record=len(SPEECH_PROMPT),
                generated_bytes=result['generated_bytes'],
                inputs_sha256={p.as_posix(): file_hash(p) for p in files},
                limit='Independent reconstruction of sampled parameter statistics and source/update labels; '
                      'activity statistics are native observations, not independently re-forwarded here.')
