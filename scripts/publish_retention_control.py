"""Publish the completed restart control while its paired rate trial continues."""
import argparse
from pathlib import Path

from native_experiment import read
from prose_founder import file_hash, write
from retention_evidence import verify_prefix


def publish(root, out):
    if out.exists():
        raise ValueError('Use a fresh publication path')
    plan = read(root / 'protocol.json')
    spec, evaluation = plan['specification'], plan['evaluation']
    partial = read(root / 'partial.json')
    identity = read(root / 'control-identity.json')
    assert identity['passed'] and identity['full_checkpoint_byte_identical']
    for name, expected in plan['authenticated_inputs'].items():
        assert file_hash(name) == expected, name
    labels = ['parent', *(f'resumed-control-{end}' for end in spec['endpoints'])]
    rows = [r for r in partial['assessments'] if r['label'] in labels]
    sessions = [s for s in partial['sessions'] if s['arm'] == 'resumed-control']
    assert [r['label'] for r in rows] == labels and len(sessions) == 2
    original = plan['original_completion']
    final = Path(rows[-1]['checkpoint'])
    assert file_hash(original['checkpoint']) == file_hash(final) == original['checkpoint_sha256']
    assert identity['original_sha256'] == identity['resumed_sha256'] == original['checkpoint_sha256']
    commands, raw_hashes = verify_prefix(root, plan, rows, sessions)
    assert len(commands) == 38
    for row, original_row in [(rows[0], plan['parent_assessment']),
                               (rows[-1], read('reports/prose-105m-complete.json')['assessments'][-1])]:
        assert [b for b in row['books'] if b['role'] != 'new_stage_training'] == original_row['books']
        assert row['samples'] == original_row['samples']
        assert row['validation_mean_nats_per_byte'] == original_row['validation_mean_nats_per_byte']
    write(out, dict(passed=True, scope='Completed original-rate restart control only',
        protocol_sha256=file_hash(root / 'protocol.json'), experiment_source_commit=plan['source_commit'],
        original_completion=original, control_identity=identity, assessments=rows, sessions=sessions,
        native_commands_verified=38, native_learning_commands=2, native_assessment_commands=36,
        completed_command_prefix=commands[:38], raw_artifact_sha256=raw_hashes,
        parent_and_final_original_assessments_exact=True, checkpoint_files_unchanged=True,
        publication_script_sha256=file_hash(Path(__file__)), publication_native_commands=0,
        quarter_rate_results_included=False, reproduction_admitted=False, reserved_tests_scored=False,
        limit='Exact native restart and recorded-output evidence on one 105M run. '
              'The paired lower-rate outcome and improved learning remain unproven in this snapshot.'))
    print('Published exact 105M restart control: 38 completed native commands verified.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('runs/prose-retention-lr'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    publish(args.root, args.out)
