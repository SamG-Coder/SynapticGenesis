"""Verify initialization and matched exposure in the native architecture screen."""
import argparse
import json
from pathlib import Path
import torch

from probes_cli import Reference


def check(root):
    comparison = json.loads((root/'comparison.json').read_text())
    protocol = comparison['protocol']
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
        rows = [r for r in comparison['runs'] if r['seed'] == seed]
        assert len(rows) == 3
        for stage in ('common_session', 'session'):
            for key in ('online_updates', 'global_updates', 'observed_pairs', 'replay_updates',
                        'replay_pairs', 'generated_bytes', 'replay_items'):
                assert len({row[stage][key] for row in rows}) == 1, (seed, stage, key)
        assert all(r['decode']['generated_bytes_identical'] and r['decode']['state_logits_max_error'] == 0 for r in rows)
    result = dict(passed=True, seeds=protocol['seeds'], native_commands=comparison['native_commands'],
                  model_count=len(comparison['runs']), shared_initial_parameters_exact=True,
                  exposure_and_replay_counts_match=True, graph_generation_exact=True,
                  smoke_only=protocol['status'] == 'smoke_only', learning_quality_claim=False)
    (root/'execution-check.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    check(p.parse_args().root)
