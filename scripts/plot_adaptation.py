"""Plot audited adaptation curves and the separate retention tradeoff."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from native_experiment import read, sha, write


def plot(root, out):
    if out.suffix.lower() not in ('.png', '.svg', '.pdf'):
        raise ValueError('Choose a PNG, SVG or PDF image output')
    metadata = out.with_name(out.name + '.metadata.json')
    data, audit = read(root/'comparison.json'), read(root/'execution-check.json')
    assert audit['passed'] and audit['comparison_sha256']==sha(root/'comparison.json')
    p=data['protocol']; rates=p['learning_rates']; starts=p['starts']
    labels=['Fresh','Parent: carry Adam','Parent: reset Adam','Later replay: carry Adam',
            'Later replay: reset Adam','Later taught: carry Adam','Later taught: reset Adam']
    colors=['#1d4ed8','#00856a','#73a600','#c75a00','#e6a100','#7d3bb3','#ce53a1']
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'axes.titleweight':'bold','figure.facecolor':'white'})
    fig, axes=plt.subplots(3,2,figsize=(14.8,12.4))
    for column,rate in enumerate(rates):
        for index,start in enumerate(starts):
            rows=[r for r in data['runs'] if r['learning_rate']==rate and r['start']==start]
            assert sorted(r['seed'] for r in rows)==p['seeds']
            for line,split in enumerate(('train','validation')):
                values=np.array([[e[split]['loss'] for e in r['evaluations']] for r in rows])
                ax=axes[line,column]
                ax.plot(p['endpoints'],values.mean(axis=0),color=colors[index],marker='o',markersize=3.5,
                        linewidth=1.7,linestyle='--' if start.endswith('reset') else '-',label=labels[index])
                ax.fill_between(p['endpoints'],values.min(axis=0),values.max(axis=0),
                                color=colors[index],alpha=.07)
            x=np.array([100*r['evaluations'][-1]['development']['joint_accuracy'] for r in rows])
            y=np.array([r['evaluations'][-1]['validation']['loss'] for r in rows])
            axes[2,column].scatter(x,y,color=colors[index],s=27,alpha=.35)
            axes[2,column].scatter([x.mean()],[y.mean()],color=colors[index],marker='s',s=65,
                                   edgecolor='white',linewidth=.7)
        for line,description in enumerate(('Fit to selected training stories','Generalization to development stories')):
            ax=axes[line,column]
            ax.set_title(f'{description}\nLearning rate {rate:g}')
            ax.set_xscale('symlog',linthresh=64,base=4)
            ax.set_xticks(p['endpoints'],[f'{x:,}' for x in p['endpoints']])
            ax.set_xlabel('New-source updates (log scale above 64)')
            ax.set_ylabel('Loss (nats / byte; lower is better)')
            ax.grid(alpha=.17)
        ax=axes[2,column]
        ax.set_title(f'Final new-reading loss and earlier binding\nLearning rate {rate:g}')
        ax.set_xlabel('Earlier binding: complete groups correct (%)')
        ax.set_ylabel('Development story loss (nats / byte)')
        ax.set_xlim(-3,103)
        ax.grid(alpha=.17)
    status='SMOKE ONLY — ' if p['status']=='smoke_only' else ''
    fig.suptitle(f'{status}Reading adaptation: fitting, generalization and retention',fontsize=18,y=.987)
    fig.text(.5,.952,f"Associative C256/H512/L4 | {len(p['seeds'])} seed(s) | {p['trajectories']} trajectories | "
             f"{p['endpoints'][-1]*128:,} source-target presentations per trajectory",ha='center',fontsize=10.5)
    handles,names=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,names,loc='lower center',bbox_to_anchor=(.5,.035),ncol=4,frameon=False,fontsize=9)
    fig.text(.5,.026,'Curves: means and min–max over seeds. Scatter: individual seeds; squares: means. '
             'Related starts are not independent replicates.',ha='center',fontsize=8.5)
    fig.text(.5,.009,'Reset-window diagnostic with no replay or teacher updates. Tiny selected corpus; '
             'reserved tests unused. Activity or age alone is not a plasticity measure.',ha='center',fontsize=8.5)
    fig.subplots_adjust(top=.90,bottom=.15,hspace=.46,wspace=.24)
    fig.savefig(out,dpi=140)
    plt.close(fig)
    write(metadata,dict(comparison_sha256=sha(root/'comparison.json'),
        execution_check_sha256=sha(root/'execution-check.json'),plot_source_sha256=sha(__file__),
        image_sha256=sha(out),status=p['status'],seeds=p['seeds'],
        limits='Seed min-max ranges are descriptive, not confidence intervals. No policy selected from this figure.'))
    print(out)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--study',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True); args=parser.parse_args()
    plot(args.study,args.out)
