"""Verify projection statistics and byte layout against the existing CPU oracle."""
import argparse
import math
from pathlib import Path
import struct
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from prose_founder import file_hash, write
from prose_projection_snapshot import inspect, layout, projection_rows, statistics
from probes_cli import Reference


def audit(out):
    matrix = np.array([[3, 4], [0, 5]], dtype=np.float32)
    gain = np.array([-2, .5], dtype=np.float32)
    expected = np.array([math.sqrt(40), 2.5])
    assert np.array_equal(projection_rows(matrix, gain), expected)
    assert statistics(np.array([0, 3, 4.]))['rms'] == math.sqrt(25 / 3)
    rejected = 0
    for matrix, gain in [(np.array([1., 2.]), np.ones(2)),
                         (np.ones((2, 3)), np.ones(2)),
                         (np.ones((2, 2)), np.array([1., np.nan])),
                         (np.full((2, 2), np.inf), np.ones(2))]:
        try:
            projection_rows(matrix, gain)
        except ValueError:
            rejected += 1
        else:
            raise AssertionError('Bad projection accepted')
    for value in (np.array([]), np.array([np.nan]), np.ones((2, 2))):
        try:
            statistics(value)
        except ValueError:
            rejected += 1
        else:
            raise AssertionError('Bad statistic accepted')
    checked = []
    for name in ('initial.ckpt', 'stage-4.ckpt'):
        path = Path('runs/prose-size-panel/founder-2m') / name
        identity = file_hash(path)
        reference = Reference(path)
        with path.open('rb') as stream:
            meta = struct.unpack('<32Q', stream.read(256))
        fields, count = layout(reference.c, reference.h, reference.l)
        assert count == meta[14]
        weights = np.memmap(path, dtype='<f4', mode='r', offset=288, shape=(count,))
        for expected_block, block in zip(reference.blocks, fields):
            for expected, (start, end, shape) in zip(expected_block, block):
                assert np.array_equal(weights[start:end].reshape(shape), expected.numpy())
        weights._mmap.close()
        result = inspect(path, identity)
        for expected, actual in zip(reference.blocks, result['layers']):
            # Independent scalar dot products, not the diagnostic's reduction.
            rows = (expected[1].double() * expected[0].double()).numpy()
            norms = np.array([math.sqrt(math.fsum(float(x) * float(x) for x in row)) for row in rows])
            assert abs(actual['effective_input_row_l2']['p50'] - float(np.median(norms))) < 2e-15
        assert file_hash(path) == identity
        checked.append(dict(checkpoint=str(path), sha256=identity, layers=reference.l,
                            tensor_roles_per_layer=14, all_weight_views_exact=True))
    write(out, dict(passed=True, native_commands=0, model_updates=0, cases=checked,
        rejected_inputs=rejected, signed_gain_scalar_fixture=True, independent_dot_products=True,
        checkpoint_files_unchanged=True,
        code_sha256={str(p): file_hash(p) for p in [Path(__file__), Path('scripts/prose_projection_snapshot.py'),
                                                   Path('tests/probes_cli.py')]},
        limit='CPU layout/statistics verification on existing native files; no new model forward or performance claim.'))
    print('PASS: two native checkpoints, 112 exact tensor views, independent scalar norms, seven bad inputs.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('Use a fresh output path')
    audit(args.out)
