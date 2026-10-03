"""Show all three seeds; bars are means, not confidence intervals."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def plot(source, output):
    data = json.loads(source.read_text())
    arms, seeds = data['protocol']['arms'], data['protocol']['seeds']
    colors = ['#687781', '#33839a', '#b85c3c']
    labels = ['Selective\n1.717M parameters', 'Wider selective\n1.952M parameters', 'Associative\n1.952M parameters']
    panels = [('development_joint',100,'Development binding: all four answers','Correct groups (%)'),
              ('reader_change',1,'Earlier-reader loss change after binding','Nats / byte (lower is better)'),
              ('live_segment_seconds',1,'24,000 live binding observations','Seconds, including replay and speech'),
              ('graph_us_per_byte',1,'Sustained graph generation, strict FP32','Microseconds / byte (lower is better)')]
    fig, axes = plt.subplots(2,2,figsize=(12.6,9.4))
    fig.subplots_adjust(left=.09,right=.98,top=.84,bottom=.20,hspace=.46,wspace=.27)
    markers = ['o','s','^']
    for ax,(key,scale,title,ylabel) in zip(axes.flat,panels):
        for i,arm in enumerate(arms):
            mean = next(r for r in data['means'] if r['arm']==arm)[key]*scale
            ax.bar(i,mean,color=colors[i],alpha=.32,width=.66)
            for j,seed in enumerate(seeds):
                value = next(r for r in data['rows'] if r['arm']==arm and r['seed']==seed)[key]*scale
                ax.scatter(i+(j-1)*.09,value,color=colors[i],marker=markers[j],s=52,zorder=3,
                           edgecolor='white',linewidth=.5)
        ax.set(title=title,ylabel=ylabel,xticks=range(3),xticklabels=labels)
        ax.grid(axis='y',alpha=.2)
        ax.set_axisbelow(True)
        ax.spines[['top','right']].set_visible(False)
        ax.axhline(0,color='#999999',linewidth=.7)
        if key=='development_joint':
            ax.set_ylim(-3,105)
    fig.suptitle('Associative memory: controlled early learning screen',fontsize=17,fontweight='bold',y=.97)
    fig.text(.5,.91,'Same selected sources and exposure • 34,000 observations per model • three paired seeds',
             ha='center',fontsize=11,color='#444444')
    fig.legend([Line2D([0],[0],marker=m,color='#555555',linestyle='None',markersize=7) for m in markers],
               [str(seed) for seed in seeds],loc='lower center',bbox_to_anchor=(.5,.047),ncol=3,
               title='Individual seeds; bars show means',frameon=False)
    fig.text(.5,.015,'Binding assay uses box/bag answers. Reading change is relative to each model at 10,000 observations. '
             'Reserved tests unused.',ha='center',fontsize=9,color='#555555')
    fig.savefig(output,dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,default=Path('reports/associative-summary.json'))
    p.add_argument('--output',type=Path,default=Path('reports/associative-comparison.png'))
    a=p.parse_args()
    plot(a.source,a.output)
