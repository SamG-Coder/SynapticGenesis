"""Reject altered assessment/comparison evidence without touching native outputs."""
import argparse
import copy
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import checkpoint_assessment as assessment
from checkpoint_assessment import verify_saved
from native_experiment import read
from prose_founder import file_hash, write
from publish_replay_capacity import verify_comparisons


def check(out):
    result = read('runs/prose-replay-capacity/result.json')
    original = next(r for r in result['rows'] if r['profile'] == '2m' and r['stage'] == 4)
    directory = Path('runs/prose-replay-capacity/2m-stage-4')
    spec = read(result['protocol']['assessment_plan'])['specification']
    update = original['exposure_counters']['global_updates']
    commands, hashes = verify_saved(directory, original, spec, update)
    assert len(commands) == 10
    baseline = read(result['protocol']['specification']['baseline_comparison'])
    assert len(verify_comparisons(result['rows'], baseline)) == 8
    rejected = []

    def reject(label, action):
        try:
            action()
        except (ValueError, AssertionError):
            rejected.append(label)
        else:
            raise AssertionError('Altered evidence accepted: ' + label)

    for name in ('book_loss', 'book_role', 'missing_sample', 'extra_sample', 'sample_hex',
                 'displayed_sample', 'sample_length', 'validation_mean', 'checkpoint_path'):
        row = copy.deepcopy(original)
        if name == 'book_loss': row['books'][0]['loss_nats_per_byte'] += .1
        elif name == 'book_role': row['books'][0]['role'] = 'training_retention'
        elif name == 'missing_sample': row['samples'].pop()
        elif name == 'extra_sample': row['samples'].append(row['samples'][0])
        elif name == 'sample_hex': row['samples'][0]['output_hex'] = '00'
        elif name == 'displayed_sample': row['samples'][0]['output_utf8'] += 'invented'
        elif name == 'sample_length': row['samples'][0]['generated_bytes'] -= 1
        elif name == 'validation_mean': row['validation_mean_nats_per_byte'] += .1
        elif name == 'checkpoint_path': row['checkpoint'] = 'runs/another-checkpoint.ckpt'
        reject(name, lambda: verify_saved(directory, row, spec, update))
    reject('checkpoint_update', lambda: verify_saved(directory, original, spec, update + 1))
    changed_commands = copy.deepcopy(commands)
    changed_commands[0][changed_commands[0].index('--batches') + 1] = '1'
    real_read = assessment.read

    def changed_read(path):
        return changed_commands if Path(path) == directory / 'commands.json' else real_read(path)
    with patch.object(assessment, 'read', side_effect=changed_read):
        reject('native_command_argument', lambda: verify_saved(directory, original, spec, update))
    for name in ('comparison_delta', 'replay_pair_delta', 'replay_update_budget'):
        rows = copy.deepcopy(result['rows'])
        if name == 'comparison_delta': rows[-1]['comparison']['validation_difference_from_1024'] += .1
        elif name == 'replay_pair_delta': rows[-1]['comparison']['replay_pair_difference'] = 0
        elif name == 'replay_update_budget': rows[-1]['exposure_counters']['replay_updates'] += 1
        reject(name, lambda: verify_comparisons(rows, baseline))
    assert all(file_hash(path) == expected for path, expected in hashes.items())
    write(out, dict(passed=True, actual_saved_assessment_commands_verified=10, paired_comparisons_recomputed=8,
        rejected_evidence_cases=rejected, raw_outputs_unchanged=True, new_native_commands=0,
        implementation_sha256={p.as_posix(): file_hash(p) for p in (Path(__file__),
            ROOT / 'scripts/checkpoint_assessment.py', ROOT / 'scripts/publish_replay_capacity.py')},
        limits='CPU evidence-rejection checks. The companion publication audits all completed candidate and baseline assessments.'))
    print('Passed actual assessment and eight comparisons; rejected', len(rejected), 'altered evidence cases.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists(), 'Use a fresh test report'
    check(args.out)
