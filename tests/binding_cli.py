"""Independent binding labels, shortcut controls, native grouped scores and parsing."""
import argparse
import importlib.util
import itertools
import json
from pathlib import Path
import re
import subprocess
import sys

from probes_cli import Reference
sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
from prepare_binding import examples, prepare
from prepare_lessons import write_probes


def check(exe, out):
    exe = exe.resolve()
    out.mkdir(parents=True, exist_ok=False)
    spec_file = Path(__file__).parents[1] / 'data/lessons-binding-v2.json'
    spec = json.loads(spec_file.read_text())
    _, _, partitions, splits = examples(spec)
    max_error, calls = 0, 0
    for name, rows in partitions.items():
        first_correct, last_correct = [], []
        for row in rows:
            facts = re.findall(r'The (\w+) is in the (\w+)\.', row['context'])
            if not facts:
                facts = [(obj, loc) for loc, obj in re.findall(r'In the (\w+) is the (\w+)\.', row['context'])]
            assert len(facts) == 2 and frozenset(obj for obj,_ in facts) in splits[name]
            queried = re.search(r'(?:Where is the|Find the) (\w+)', row['query']).group(1)
            gold = dict(facts)[queried] + '.'
            assert row[f"choice{row['correct']}"] == gold
            first_correct.append(facts[0][1] + '.' == gold)
            last_correct.append(facts[-1][1] + '.' == gold)
        assert sum(first_correct) == sum(last_correct) == len(rows)//2
        assert not any(all(first_correct[i:i+4]) or all(last_correct[i:i+4]) for i in range(0,len(rows),4))
    fixture = partitions['development'][:4]
    path = out / 'binding.sgprobe'
    write_probes(path,fixture,'SGPROBE2')

    def run(*args, reject=False):
        nonlocal calls
        calls += 1
        p = subprocess.run([str(exe), *map(str,args)],capture_output=True)
        (out / f'command-{calls}.log').write_bytes(p.stdout+p.stderr)
        assert (p.returncode != 0) == reject, (args,p.stderr)

    checkpoints = [Path('build/test-results/resume.ckpt'),Path('build/adaptive-test-results/resume.ckpt'),
                   Path('build/trace-test-results/resume.ckpt')]
    for index, checkpoint in enumerate(checkpoints):
        result = out / f'scores-{index}.json'
        run('language-probes','--checkpoint',checkpoint,'--probes',path,'--output',result)
        values = json.loads(result.read_text())
        ref = Reference(checkpoint)
        assert values['groups']==1 and values['group_size']==4 and values['context_erased_joint_accuracy']==0
        assert 'paired_accuracy' not in values
        correct, exact = [], []
        for row,actual in zip(fixture,values['results']):
            wanted=[ref.score(row['context']+row['query'],row[f'choice{j}']) for j in range(2)]
            max_error=max(max_error,max(abs(a-b) for a,b in zip(wanted,actual['candidate_nll'])))
            assert max_error < 3e-5
            generated=ref.greedy(row['context']+row['query'],4)
            assert generated==actual['greedy']
            correct.append(actual['prediction']==row['correct'])
            exact.append(generated==row[f"choice{row['correct']}"])
        assert values['joint_accuracy']==int(all(correct)) and values['greedy_exact_joint_accuracy']==int(all(exact))
    for kind in ('wrong-label','duplicate-query','duplicate-context','unequal-choices','short-group'):
        bad=[dict(x) for x in fixture]
        if kind=='wrong-label': bad[1]['correct']=bad[0]['correct']
        if kind=='duplicate-query': bad[1]['query']=bad[0]['query']
        if kind=='duplicate-context': bad[2]['context']=bad[0]['context']
        if kind=='unequal-choices': bad[1]['choice0']='bad.'
        if kind=='short-group': bad=bad[:2]
        dest=out/f'{kind}.sgprobe';write_probes(dest,bad,'SGPROBE2')
        output=out/f'{kind}.json'
        run('language-probes','--checkpoint',checkpoints[0],'--probes',dest,'--output',output,reject=True)
        assert not output.exists()
    prepare(spec_file,out/'prepared-a')
    prepare(spec_file,out/'prepared-b')
    assert all(p.read_bytes()==(out/'prepared-b'/p.name).read_bytes() for p in (out/'prepared-a').iterdir())
    report=dict(passed=True,native_commands=calls,oracle_max_score_error=max_error,
                independent_fact_query_labels_checked=True,copy_first_or_last_joint_accuracy=0,
                full_grid_required=True,group_metric_verified=True,prepared_bytes_reproduce=True,
                test_labels_validated_but_no_model_evaluated_on_test=True)
    (out/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--exe',type=Path,default=Path('build/synapticgenesis.exe'))
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();check(a.exe,a.out)
