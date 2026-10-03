"""Reject the new file layout in readers scoped to preserved legacy studies."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from experiment_checkpoint import checkpoint, policy_checkpoint, teaching


def run(fixtures, out):
    if out.exists():
        raise ValueError('Use a fresh report')
    legacy = sorted(fixtures.glob('*-legacy.ckpt'))
    wrapped = sorted(fixtures.glob('*-membrane.ckpt'))
    assert len(legacy) == 40 and len(wrapped) == 36
    identities = {}
    for path in legacy:
        raw = path.read_bytes()
        meta = struct.unpack_from('<32Q', raw)
        actual, extra, arrays, digest = checkpoint(path)
        end = 288 + 12 * meta[14] + 4 * meta[18]
        assert actual == meta and arrays == raw[288:end]
        assert extra == struct.unpack_from(f'<{meta[31]}Q', raw, end)
        assert digest == hashlib.sha256(raw).hexdigest()
        teacher = teaching(path)
        if meta[17] == 6:
            assert teacher == struct.unpack_from('<32Q', raw, end + 8 * meta[31])
        else:
            assert teacher is None
        if meta[17] == 5 and not extra[9]:
            assert policy_checkpoint(path) == (meta, extra)
        identities[path.name] = digest
    rejected = 0
    for path in wrapped:
        for reader in (checkpoint, teaching, policy_checkpoint):
            try:
                reader(path)
            except ValueError:
                rejected += 1
            else:
                raise AssertionError(f'{reader.__name__} misread {path.name}')
        identities[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    result = dict(passed=True, cuda_executed=False, legacy_files=40, wrapped_files=36,
                  incompatible_reader_calls_rejected=rejected, fixture_sha256=identities)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(f'PASS: 40 legacy files; {rejected} wrapped-format reader calls rejected')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fixtures', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    run(args.fixtures, args.out)
