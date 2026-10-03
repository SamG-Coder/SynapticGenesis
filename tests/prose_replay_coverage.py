"""Check bounded checkpoint policy reads and distinct replay-position accounting."""
import argparse
from pathlib import Path
import struct
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from experiment_checkpoint import checkpoint, policy_checkpoint
from prose_founder import file_hash, write
from prose_replay_coverage import unique_counts


def audit(out, report):
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    for stage in (3, 4):
        path = Path(f'runs/prose-size-panel/founder-2m/stage-{stage}.ckpt')
        before = file_hash(path)
        expected_meta, expected_extra, _, _ = checkpoint(path)
        actual_meta, actual_extra = policy_checkpoint(path)
        assert actual_meta == expected_meta and actual_extra == expected_extra
        assert before == file_hash(path)
        rows.append(dict(checkpoint=path.as_posix(), sha256=before,
            checkpoint_bytes=path.stat().st_size, metadata_bytes_read=256 + 8 * actual_meta[31],
            complete_header_and_replay_payload_match=True))
    lengths = [258, 9]
    events = [(0, 0, 128), (0, 0, 128), (0, 128, 128), (0, 256, 1), (1, 0, 8)]
    assert unique_counts(events, lengths, 0, 2, 128) == dict(
        distinct_windows=4, distinct_target_pairs=265, covered_documents=[0, 1])
    rejected = []
    for label, episode in [('wrong-document', (2, 0, 8)), ('unaligned', (0, 1, 128)),
                           ('past-end', (0, 384, 1)), ('wrong-tail', (0, 256, 2))]:
        try:
            unique_counts([episode], lengths, 0, 2, 128)
        except ValueError:
            rejected.append(label)
        else:
            raise AssertionError('Malformed descriptor accepted')
    for label in ('version', 'unbounded-extra', 'truncated-payload'):
        meta = list(actual_meta)
        if label == 'version':
            meta[17] = 6
        if label == 'unbounded-extra':
            meta[31] = 1 << 63
        path = out / f'{label}.ckpt'
        path.write_bytes(struct.pack('<32Q', *meta))
        try:
            policy_checkpoint(path)
        except ValueError:
            rejected.append(label)
        else:
            raise AssertionError('Invalid policy header accepted')
    write(report, dict(passed=True, cases=rows, duplicate_and_short_tail_accounting=True,
        rejected=rejected, native_commands=0, model_updates=0,
        code_sha256={p.as_posix(): file_hash(p) for p in [Path(__file__), Path('scripts/prose_replay_coverage.py'),
            Path('scripts/experiment_checkpoint.py'), Path('tests/stage_replay_reference.py')]},
        limit='CPU reader/accounting checks. The companion coverage report verifies the full final-stage '
              'source, replay and RNG trajectory against native endpoints. No alternative replay policy was trained.'))
    print('PASS: bounded reads match both preserved headers/policies; duplicate/tail counts and seven rejections.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        parser.error('Use a fresh report path')
    audit(args.out, args.report)
