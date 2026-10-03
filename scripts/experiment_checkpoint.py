"""Read checkpoint evidence for experiments; the native loader validates files.

This lightweight diagnostic view does not replace native checksum/layout checks.
"""
import hashlib
import struct


def checkpoint(path):
    raw = path.read_bytes()
    meta = struct.unpack_from('<32Q', raw)
    end = 288 + 12*meta[14] + 4*meta[18]
    extra = struct.unpack_from(f'<{meta[31]}Q', raw, end)
    return meta, extra, raw[288:end], hashlib.sha256(raw).hexdigest()


def distribution(path, boundaries):
    meta, extra, _, _ = checkpoint(path)
    start = 17 + 5*extra[16] if meta[17] in (5, 6) else 16
    counts = [0]*len(boundaries)
    for doc in extra[start::3]:
        counts[next(i for i, end in enumerate(boundaries) if doc < end)] += 1
    return counts


def teaching(path):
    """Diagnostic policy view; native loading remains the validation authority."""
    raw = path.read_bytes()
    meta = struct.unpack_from('<32Q', raw)
    if meta[17] != 6:
        return None
    offset = 288 + 12*meta[14] + 4*meta[18] + 8*meta[31]
    return struct.unpack_from('<32Q', raw, offset)


def state_record(path):
    """Comparable exposure/state evidence for grouped-replay experiments."""
    meta, extra, _, identity = checkpoint(path)
    assert meta[17] in (5, 6) and extra[1] == 3
    groups = [dict(zip(('document_end', 'seen_windows', 'stored_windows', 'replay_updates', 'replay_pairs'),
                       extra[17 + 5*i:22 + 5*i])) for i in range(extra[16])]
    result = dict(checkpoint_sha256=identity, parameters=meta[14], cell=meta[1],
                  online_updates=meta[24], global_updates=meta[7], observed_pairs=meta[22],
                  generated_bytes=meta[30], replay_updates=extra[6], replay_pairs=extra[7],
                  curriculum_stage=extra[15]+1, replay_groups=groups)
    policy = teaching(path)
    if policy:
        result['teaching'] = dict(active=bool(policy[2]), teachers=policy[3], bundle_hash=str(policy[4]),
                                  eligible_documents=policy[6], updates=policy[9], pairs=policy[10],
                                  words=list(policy))
    return result
