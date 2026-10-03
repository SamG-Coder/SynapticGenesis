"""CPU-check one fixed development reversal group in every final paired model.

This checks evaluator numerics at the trained model dimensions. It is neither a
new quality benchmark nor a way to select a checkpoint or tune the model.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

from probes_cli import Reference
sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
from prepare_binding import examples
from prepare_lessons import write_probes


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(root):
    data = json.loads((root/'comparison.json').read_text())
    spec_path = Path(__file__).parents[1]/'data/lessons-binding-v2.json'
    rows = examples(json.loads(spec_path.read_text()))[2]['development']
    with TemporaryDirectory() as temporary:
        probe = Path(temporary)/'development.sgprobe'
        write_probes(probe, rows, 'SGPROBE2')
        assert sha(probe) == data['protocol']['probe_sha256']['development.sgprobe']
    fixture = rows[:4]
    assert len({row['pair'] for row in fixture}) == 1
    final_update = data['protocol']['online_endpoints'][-1]
    records = []
    for seed in data['protocol']['seeds']:
        for arm in data['protocol']['arms']:
            directory = root/f'{seed}-{arm}'
            checkpoint = directory/f'checkpoint-{final_update}.ckpt'
            identity = sha(checkpoint)
            declared = next(row for row in data['runs'] if row['seed'] == seed and
                            row['arm'] == arm and row['online_updates'] == final_update)
            assert identity == declared['checkpoint_sha256']
            native = json.loads((directory/f'development-{final_update}.json').read_text())
            ref = Reference(checkpoint)
            error = 0
            for row, actual in zip(fixture, native['results'][:4]):
                assert row['id'] == actual['id'] and row['correct'] == actual['gold']
                for context, field in [(row['context'], 'candidate_nll'), ('', 'context_erased_nll')]:
                    expected = [ref.score(context+row['query'], row[f'choice{i}']) for i in range(2)]
                    error = max(error, *(abs(a-b) for a,b in zip(expected, actual[field])))
                assert ref.greedy(row['context']+row['query'], 4) == actual['greedy']
            assert error < 3e-5, (seed, arm, error)
            assert sha(checkpoint) == identity
            records.append(dict(seed=seed, arm=arm, items=4, oracle_max_score_error=error,
                                greedy_answers_identical=True, checkpoint_sha256=identity))
            print(f'{seed} {arm}: CPU score error {error:.8g}, greedy answers identical', flush=True)
    result = dict(passed=True, fixed_development_group=fixture[0]['pair'],
                  all_final_models_checked=True, parameters=declared['parameters'],
                  full_and_context_erased_scores_checked=True, checkpoint_files_unchanged=True,
                  reserved_test_not_evaluated=True, records=records,
                  interpretation='Numerical sanity check of one fixed group per model, not an independent quality benchmark.')
    (root/'learned-oracle.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    check(p.parse_args().root)
