"""Verify initialization and matched exposure in the native architecture screen."""
import argparse
import json
from pathlib import Path
import torch
import sys

from probes_cli import Reference
sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
from experiment_checkpoint import checkpoint
from native_experiment import sha


def check(root, early=None):
    comparison = json.loads((root/'comparison.json').read_text())
    protocol = comparison['protocol']
    endpoints = protocol['online_endpoints']
    expected_count = len(protocol['seeds']) * len(protocol['arms']) * len(endpoints)
    assert len(comparison['runs']) == expected_count
    assert len({(r['seed'], r['arm'], r['online_updates']) for r in comparison['runs']}) == expected_count
    for seed in protocol['seeds']:
        baseline = Reference(root/f'{seed}-selective/initial.ckpt')
        memory = Reference(root/f'{seed}-associative/initial.ckpt')
        assert baseline.cell == 5 and memory.cell == 6
        for name in ('embedding', 'gain', 'head', 'bias'):
            assert torch.equal(getattr(baseline, name), getattr(memory, name)), name
        for ordinary, associative in zip(baseline.blocks, memory.blocks):
            assert len(ordinary) == 10 and len(associative) == 14
            assert all(torch.equal(a, b) for a, b in zip(ordinary, associative[:10]))
            assert torch.count_nonzero(associative[12]) == torch.count_nonzero(associative[13]) == 0
        for endpoint in endpoints:
            rows = [r for r in comparison['runs'] if r['seed'] == seed and r['online_updates'] == endpoint]
            assert len(rows) == 3
            for stage in ('common_session', 'session'):
                for key in ('online_updates', 'global_updates', 'observed_pairs', 'replay_updates',
                            'replay_pairs', 'generated_bytes', 'replay_items'):
                    assert len({row[stage][key] for row in rows}) == 1, (seed, endpoint, stage, key)
            assert all(r['decode']['generated_bytes_identical'] and r['decode']['state_logits_max_error'] == 0 for r in rows)
    matched_history = []
    if early is not None:
        old = json.loads((early / 'comparison.json').read_text())
        assert old['protocol']['online_endpoints'] == [34000] and 34000 in endpoints
        assert old['protocol']['seeds'] == protocol['seeds']
        assert old['protocol']['prepared_manifest_sha256'] == protocol['prepared_manifest_sha256']
        for row in old['runs']:
            directory = f'{row["seed"]}-{row["arm"]}'
            for update, field in [(10000, 'prerequisite_checkpoint_sha256'), (34000, 'checkpoint_sha256')]:
                name = f'checkpoint-{update}.ckpt'
                source, current = early / directory / name, root / directory / name
                assert sha(source) == row[field]
                _, _, prior_state, _ = checkpoint(source)
                _, _, current_state, _ = checkpoint(current)
                assert prior_state == current_state, (directory, update)
                matched_history.append(dict(seed=row['seed'], arm=row['arm'], online_updates=update,
                                            learned_and_recurrent_payload_identical=True))
    result = dict(passed=True, seeds=protocol['seeds'], native_commands=comparison['native_commands'],
                  model_count=len(protocol['seeds']) * len(protocol['arms']),
                  endpoint_count=len(endpoints), checkpoint_rows=expected_count,
                  shared_initial_parameters_exact=True,
                  exposure_and_replay_counts_match=True, graph_generation_exact=True,
                  smoke_only=protocol['status'] == 'smoke_only', learning_quality_claim=False)
    if early is not None:
        result['early_history'] = matched_history
        result['early_history_comparison_scope'] = ('Weights, Adam arrays and recurrent payload bytes only. '
            'Complete files have different declared schedule hashes. No trained early checkpoint was loaded.')
    (root/'execution-check.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--early-reference',type=Path)
    a = p.parse_args()
    check(a.root, a.early_reference)
