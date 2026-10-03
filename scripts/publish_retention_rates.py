"""Authenticate the completed rate comparison before publishing its exact bytes."""
import argparse
from pathlib import Path

from native_experiment import read
from prose_founder import file_hash, write
from retention_evidence import verify_prefix


def publish(root, report, audit):
    if report.exists() or audit.exists():
        raise ValueError('Use fresh publication outputs')
    result_path = root / 'result.json'
    result = read(result_path)
    assert result['complete'] and not (root / 'failure.json').exists()
    plan = read(root / 'protocol.json')
    assert result['protocol'] == plan
    for name, expected in plan['authenticated_inputs'].items():
        assert file_hash(name) == expected, name
    rows, sessions = result['assessments'], result['sessions']
    assert len(rows) == 5 and len(sessions) == 4
    commands, hashes = verify_prefix(root, plan, rows, sessions)
    assert len(commands) == len(read(root / 'commands/commands.json')) == result['native_commands'] == 64
    assert sum(c[1] == 'live' for c in commands) == 4
    identity = read(root / 'control-identity.json')
    control = rows[2]
    assert identity['passed'] and identity['full_checkpoint_byte_identical']
    assert (identity['original_sha256'] == identity['resumed_sha256'] == control['checkpoint_sha256']
            == file_hash(plan['original_completion']['checkpoint']))
    for row, original in [(rows[0], plan['parent_assessment']),
                          (control, read('reports/prose-105m-complete.json')['assessments'][-1])]:
        assert [b for b in row['books'] if b['role'] != 'new_stage_training'] == original['books']
        assert row['samples'] == original['samples']
    matches = []
    for index in (1, 2):
        original, candidate = rows[index], rows[index + 2]
        for key in original['state']:
            if key != 'hyperparameters':
                assert original['state'][key] == candidate['state'][key], key
        assert original['state']['hyperparameters'][1:7] == candidate['state']['hyperparameters'][1:7]
        matches.append(dict(online_updates=original['state']['counters']['online_updates'],
            identical_source_replay_speech_counters=True, identical_replay_cursor_and_rng=True))
    expected = []
    parent, base = rows[0], {b['book']: b for b in rows[0]['books']}
    for row in rows[1:]:
        end = row['state']['counters']['online_updates']
        original = next(r for r in rows if r.get('arm') == 'resumed-control'
                        and r['state']['counters']['online_updates'] == end)
        books = {b['book']: b for b in original['books']}
        expected.append(dict(arm=row['arm'], online_updates=end,
            validation_mean_nats_per_byte=row['validation_mean_nats_per_byte'],
            validation_change_from_parent=row['validation_mean_nats_per_byte'] - parent['validation_mean_nats_per_byte'],
            validation_difference_from_control=row['validation_mean_nats_per_byte'] - original['validation_mean_nats_per_byte'],
            books=[dict(book=b['book'], role=b['role'], loss_nats_per_byte=b['loss_nats_per_byte'],
                change_from_parent=b['loss_nats_per_byte'] - base[b['book']]['loss_nats_per_byte'],
                difference_from_control=b['loss_nats_per_byte'] - books[b['book']]['loss_nats_per_byte']) for b in row['books']]))
    assert expected == result['differences']
    assert all(result[k] for k in ('parent_assessment_exact', 'control_checkpoint_byte_identical',
                                  'matched_source_replay_and_speech_counters', 'matched_replay_payloads'))
    assert not result['reserved_tests_scored'] and not result['reproduction_admitted']
    report.write_bytes(result_path.read_bytes())
    write(audit, dict(passed=True, result_sha256=file_hash(result_path), published_sha256=file_hash(report),
        exact_result_copy=True, protocol_sha256=file_hash(root / 'protocol.json'),
        verified_native_commands=64, learning_commands=4, book_evaluations=40, generations=20,
        raw_artifact_sha256=hashes, matched_endpoints=matches, raw_differences_recomputed=True,
        original_control_byte_identical=True, publication_native_commands=0, new_model_updates=0,
        reserved_tests_scored=False, reproduction_admitted=False,
        implementation_sha256={p.as_posix(): file_hash(p) for p in (Path(__file__), Path('scripts/retention_evidence.py'))},
        limit='Artifact verification of one exploratory two-rate run, not an extra seed, '
              'fresh numerical oracle, language benchmark or automatic policy promotion.'))
    print('Published both rates: all 64 native commands and both matched endpoints verified.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('runs/prose-retention-lr'))
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    args = parser.parse_args()
    publish(args.root, args.report, args.audit)
