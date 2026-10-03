"""Bounded evidence reader for ordinary and membrane-wrapped replay studies.

The native loader remains the checksum and model-loading authority. This
reader handles only associative, batch-one, grouped replay without SI/teachers.
"""
import hashlib
import math
from pathlib import Path
import struct


def require(condition, message):
    if not condition:
        raise ValueError(message)


def f32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


def word_float(word):
    require(0 <= word <= 0xffffffff, 'Policy float has high bits set')
    return struct.unpack('<f', struct.pack('<I', word))[0]


def unpack(stream, layout):
    raw = stream.read(struct.calcsize(layout))
    require(len(raw) == struct.calcsize(layout), 'Truncated checkpoint header')
    return struct.unpack(layout, raw)


def read_state(path):
    path = Path(path)
    with path.open('rb') as stream:
        meta = unpack(stream, '<32Q')
        hp = unpack(stream, '<8f')
        c, h, layers = meta[2:5]
        require(meta[0] == 0x314d4c53434e5042 and meta[1] == 6 and meta[5] == 1 and
                8 <= c <= 2048 and 8 <= h <= 8192 and 1 <= layers <= 32 and 1 <= meta[6] <= 4096,
                'Unexpected associative study checkpoint')
        parameters = 513 * c + 256 + layers * (3 * c * h + 103 * h + 35 * c + 98)
        require(meta[14] == parameters and meta[18] == layers * (2 * h + 1024), 'Model layout differs')
        require(all(math.isfinite(v) for v in hp), 'Nonfinite checkpoint policy')
        prefix, cost, band = 288, 0., 1.5
        if meta[17] == 7:
            words = unpack(stream, '<8Q')
            require(words[:3] == (0x314d454d504753, 1, 5) and not any(words[5:]),
                    'Unexpected membrane policy wrapper')
            cost, band = word_float(words[3]), word_float(words[4])
            require(math.isfinite(cost) and 0 < cost <= 100 and math.isfinite(band) and 1 < band < 2,
                    'Invalid membrane policy values')
            prefix = 352
        else:
            require(meta[17] == 5, 'Expected ordinary grouped replay or its membrane wrapper')
        require(22 <= meta[31] <= 17 + 5 * 4096 + 3 * 65536, 'Unbounded replay payload')
        replay_at = prefix + 12 * parameters + 4 * meta[18]
        require(path.stat().st_size == replay_at + 8 * meta[31], 'Unexpected checkpoint extent')
        stream.seek(replay_at)
        raw = stream.read(8 * meta[31])
        extra = struct.unpack(f'<{meta[31]}Q', raw)
    require(extra[1] == 3 and extra[9] == 0 and extra[16] == extra[15] + 1 and
            1 <= extra[16] <= min(4096, extra[3]) and extra[3] <= 65536,
            'Expected ordinary grouped replay without consolidation')
    require(17 + 5 * extra[16] <= len(extra), 'Truncated replay group metadata')
    groups = [list(extra[17 + 5 * i:22 + 5 * i]) for i in range(extra[16])]
    stored = sum(g[2] for g in groups)
    require(stored <= extra[3] and all(g[2] <= g[1] for g in groups) and
            len(extra) == 17 + 5 * len(groups) + 3 * stored, 'Replay records differ')
    return dict(meta=list(meta), hyperparameters=list(hp), prefix_bytes=prefix,
        membrane_cost=cost, membrane_band=band, seed=meta[10], parameters=parameters,
        shape=[c, h, layers], spiking_neurons=h * layers, graph=extra[8],
        replay_every=extra[2], replay_capacity=extra[3], replay_groups=groups,
        replay_payload_sha256=hashlib.sha256(raw).hexdigest(),
        cursor_and_rng=list(meta[19:22]) + [meta[23], meta[25]], speech_policy=list(meta[26:30]),
        counters=dict(online_updates=meta[24], global_updates=meta[7], observed_pairs=meta[22],
            generated_bytes=meta[30], replay_updates=extra[6], replay_pairs=extra[7],
            curriculum_stage=extra[15] + 1))


def initial_fingerprint(path, state):
    require(all(state['counters'][k] == 0 for k in ('online_updates', 'global_updates', 'observed_pairs',
            'generated_bytes', 'replay_updates', 'replay_pairs')), 'Founder already has learning history')
    meta, hp = list(state['meta']), list(state['hyperparameters'])
    meta[15], meta[17], hp[4] = 0, 5, 0  # checksum, wrapper and explicit activity-cost policy
    digest = hashlib.sha256()
    remaining = 12 * state['parameters'] + 4 * meta[18]
    with Path(path).open('rb') as stream:
        stream.seek(state['prefix_bytes'])
        while remaining:
            block = stream.read(min(8 * 1024**2, remaining))
            require(block, 'Truncated neural state')
            digest.update(block)
            remaining -= len(block)
    return dict(normalized_meta=meta, normalized_hyperparameters=hp,
                neural_arrays_sha256=digest.hexdigest(), replay_payload_sha256=state['replay_payload_sha256'])


def matched_exposure(left, right):
    for key in ('counters', 'seed', 'shape', 'cursor_and_rng', 'speech_policy', 'graph',
                'replay_every', 'replay_capacity', 'replay_groups', 'replay_payload_sha256'):
        require(left[key] == right[key], 'Matched learning differs: ' + key)
    require(all(left['hyperparameters'][i] == right['hyperparameters'][i] for i in (0, 1, 2, 3, 5, 6, 7)),
            'An optimizer setting other than the declared objective changed')
