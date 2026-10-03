"""Post-hoc CPU audit of every original development question in every final model.

Reports disagreements without relaxing the earlier fixed-group score tolerance.
No model update, checkpoint selection or reserved-test evaluation occurs.
"""
import argparse
from functools import lru_cache
import json
from pathlib import Path
import torch

from binding_learned_oracle import development_rows, sha
from probes_cli import Reference


def audit(root):
    data = json.loads((root/'comparison.json').read_text())
    p = data['protocol']
    rows = development_rows(p)
    end = p['online_endpoints'][-1]
    records = []
    for seed in p['seeds']:
        for arm in p['arms']:
            directory = root/f'{seed}-{arm}'
            checkpoint = directory/f'checkpoint-{end}.ckpt'
            identity = sha(checkpoint)
            declared = next(r for r in data['runs'] if (r['seed'],r['arm'],r['online_updates']) == (seed,arm,end))
            assert identity == declared['checkpoint_sha256']
            native = json.loads((directory/f'development-{end}.json').read_text())['results']
            assert len(native) == len(rows)
            ref = Reference(checkpoint)
            score = lru_cache(maxsize=None)(ref.score)
            errors, choices, greedy_differences, groups = [], [], [], {}
            for row, actual in zip(rows,native):
                assert (row['id'],row['correct']) == (actual['id'],actual['gold'])
                values = {}
                for context,field in [(row['context'],'candidate_nll'),('','context_erased_nll')]:
                    expected = [score(context+row['query'],row[f'choice{j}']) for j in range(2)]
                    error = max(abs(a-b) for a,b in zip(expected,actual[field]))
                    errors.append(error)
                    values[field] = expected
                full = values['candidate_nll']
                prediction = -1 if abs(full[0]-full[1]) <= 1e-8 else int(full[1]<full[0])
                answer = ref.greedy(row['context']+row['query'],4)
                if prediction != actual['prediction']:
                    choices.append(dict(id=row['id'],native=actual['prediction'],cpu=prediction))
                if answer != actual['greedy']:
                    greedy_differences.append(dict(id=row['id'],native=actual['greedy'],cpu=answer))
                groups.setdefault(row['pair'],[]).append((prediction==row['correct'],answer==row[f'choice{row["correct"]}']))
            assert len(groups) == 144 and all(len(g)==4 for g in groups.values())
            assert sha(checkpoint) == identity
            record = dict(seed=seed,arm=arm,checkpoint_sha256=identity,items=len(rows),groups=len(groups),
                cpu_joint_accuracy=sum(all(v[0] for v in g) for g in groups.values())/len(groups),
                cpu_greedy_joint_accuracy=sum(all(v[1] for v in g) for g in groups.values())/len(groups),
                native_joint_accuracy=declared['development']['joint_accuracy'],
                native_greedy_joint_accuracy=declared['development']['greedy_exact_joint_accuracy'],
                max_score_error=max(errors),scored_pairs_above_3e5_tolerance=sum(e>=3e-5 for e in errors),
                candidate_choice_disagreements=choices,greedy_answer_disagreements=greedy_differences)
            records.append(record)
            print(f'{seed} {arm}: CPU joint={record["cpu_joint_accuracy"]:.6f}; '
                  f'choice disagreements={len(choices)}, greedy disagreements={len(greedy_differences)}',flush=True)
    result = dict(completed=True,post_hoc_diagnostic=True,torch_version=torch.__version__,
        reference_sha256=sha(Path(__file__).with_name('probes_cli.py')),driver_sha256=sha(Path(__file__)),
        original_development_probe_sha256=p['probe_sha256']['development.sgprobe'],
        original_score_tolerance_unchanged=3e-5,checkpoint_files_unchanged=True,reserved_test_not_evaluated=True,
        scope='Every final model on all 576 original development items; counts disagreements, not a new held-out benchmark.',
        records=records)
    (root/'cpu-development-audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    audit(p.parse_args().root)
