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
    start = 17 + 5*extra[16] if meta[17] == 5 else 16
    counts = [0]*len(boundaries)
    for doc in extra[start::3]:
        counts[next(i for i, end in enumerate(boundaries) if doc < end)] += 1
    return counts
