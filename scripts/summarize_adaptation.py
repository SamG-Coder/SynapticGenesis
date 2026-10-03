"""Separate fitting, generalization and retention in a completed adaptation study."""
import argparse
from pathlib import Path
import statistics

from native_experiment import read, sha, write


def area(points, split):
    total=0.
    for a,b in zip(points,points[1:]):
        total+=(b['updates']-a['updates'])*(a[split]['loss']+b[split]['loss'])/2
    return total/points[-1]['updates']


def summarize(root):
    data,check=read(root/'comparison.json'),read(root/'execution-check.json')
    assert check['passed'] and check['comparison_sha256']==sha(root/'comparison.json')
    p=data['protocol']; runs=[]
    for row in data['runs']:
        first,last=row['evaluations'][0],row['evaluations'][-1]
        measured={}
        for split in ('train','validation'):
            measured[split]=dict(initial_loss=first[split]['loss'],final_loss=last[split]['loss'],
                loss_reduction=first[split]['loss']-last[split]['loss'],
                normalized_curve_area=area(row['evaluations'],split))
        measured['binding']=dict(initial_joint_accuracy=first['development']['joint_accuracy'],
            final_joint_accuracy=last['development']['joint_accuracy'],
            change_pp=100*(last['development']['joint_accuracy']-first['development']['joint_accuracy']))
        measured['earlier_books']={name:dict(initial_loss=value['loss_nats_per_byte'],
            final_loss=last['earlier_books'][name]['loss_nats_per_byte'],
            change=last['earlier_books'][name]['loss_nats_per_byte']-value['loss_nats_per_byte'])
            for name,value in first['earlier_books'].items()}
        runs.append(dict(label=row['label'],seed=row['seed'],start=row['start'],
                         learning_rate=row['learning_rate'],**measured,
                         update_seconds=row['native_result']['update_seconds'],
                         peak_explicit_model_bytes=row['native_result']['peak_explicit_model_bytes']))
    aggregates=[]
    for rate in p['learning_rates']:
        for start in p['starts']:
            rows=[r for r in runs if r['start']==start and r['learning_rate']==rate]
            assert sorted(r['seed'] for r in rows)==p['seeds']
            aggregate=dict(learning_rate=rate,start=start,seeds=len(rows))
            for split in ('train','validation','binding'):
                aggregate[split]={key:statistics.mean(r[split][key] for r in rows) for key in rows[0][split]}
            aggregate['earlier_book_changes']={name:statistics.mean(r['earlier_books'][name]['change'] for r in rows)
                                               for name in rows[0]['earlier_books']}
            aggregate['mean_update_seconds']=statistics.mean(r['update_seconds'] for r in rows)
            aggregates.append(aggregate)
    fresh_pairs,reset_pairs=[],[]
    for row in runs:
        if row['start']=='fresh':
            continue
        fresh=next(r for r in runs if r['start']=='fresh' and r['seed']==row['seed']
                   and r['learning_rate']==row['learning_rate'])
        fresh_pairs.append(dict(seed=row['seed'],start=row['start'],learning_rate=row['learning_rate'],
            **{split:dict(final_loss_difference=row[split]['final_loss']-fresh[split]['final_loss'],
                           curve_area_difference=row[split]['normalized_curve_area']-fresh[split]['normalized_curve_area'],
                           initial_loss_difference=row[split]['initial_loss']-fresh[split]['initial_loss'])
               for split in ('train','validation')}))
        if row['start'].endswith('-reset'):
            carry=next(r for r in runs if r['start']==row['start'].replace('-reset','-carry')
                       and r['seed']==row['seed'] and r['learning_rate']==row['learning_rate'])
            reset_pairs.append(dict(seed=row['seed'],origin=row['start'].split('-')[0],learning_rate=row['learning_rate'],
                train_final_reset_minus_carry=row['train']['final_loss']-carry['train']['final_loss'],
                validation_final_reset_minus_carry=row['validation']['final_loss']-carry['validation']['final_loss'],
                binding_final_reset_minus_carry_pp=100*(row['binding']['final_joint_accuracy']-carry['binding']['final_joint_accuracy'])))
    return dict(kind='Completed native adaptation measurements; no automatic policy promotion',
        status=p['status'],comparison_sha256=sha(root/'comparison.json'),
        execution_check_sha256=sha(root/'execution-check.json'),summary_source_sha256=sha(__file__),
        seeds=p['seeds'],endpoints=p['endpoints'],learning_rates=p['learning_rates'],trajectories=len(runs),
        runs=runs,aggregates=aggregates,experienced_minus_fresh=fresh_pairs,reset_minus_carry=reset_pairs,
        all_model_sources_selected=True,reserved_tests_scored=False,
        limitations=['Small early-reader edition, repeated training windows and only three declared full-study seeds.',
                    'Related starts are not independent replicates; the two rates remain separate conditions.',
                    'Fitting, generalization and earlier-skill retention are different outcomes.',
                    'Starting-loss advantages and curve area alone do not establish preserved plasticity.',
                    'Reset-window diagnostic without replay or teachers; not the complete live policy.',
                    'Known independent native/CPU numerical discrepancies remain unresolved.',
                    'Update timings exclude evaluation/persistence/startup and lack whole-device isolation.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--study',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True); args=parser.parse_args()
    result=summarize(args.study); write(args.out,result)
    print(result['status'],result['trajectories'],'trajectories summarized without selecting a best rate or seed.')
