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
    if data['protocol'].get('longitudinal'):
        return plot_long(data, output)
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


def plot_long(data, output):
    arms, seeds = data['protocol']['arms'], data['protocol']['seeds']
    endpoints = data['protocol']['online_endpoints']
    colors = ['#687781', '#33839a', '#b85c3c']
    short = ['Selective H512', 'Selective H588', 'Associative H512']
    labels = ['Selective\n1.717M parameters', 'Wider selective\n1.952M parameters',
              'Associative\n1.952M parameters']
    markers = ['o', 's', '^']
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.8))
    fig.subplots_adjust(left=.09, right=.98, top=.82, bottom=.21, hspace=.58, wspace=.27)
    for ax, (key, scale, title, ylabel) in zip(axes[0], [
        ('development_joint', 100, 'Development binding: all four answers', 'Correct groups (%)'),
        ('reader_change', 1, 'Earlier-reader loss change since 10,000 observations', 'Nats / byte (lower is better)')]):
        for i, arm in enumerate(arms):
            for j, seed in enumerate(seeds):
                values = [next(r for r in data['rows'] if r['arm'] == arm and r['seed'] == seed
                               and r['online_updates'] == end)[key] * scale for end in endpoints]
                ax.plot([end / 1000 for end in endpoints], values, color=colors[i],
                        marker=markers[j], markersize=4.5, alpha=.65, linewidth=.8)
            means = [next(r for r in data['means'] if r['arm'] == arm and r['online_updates'] == end)[key] * scale
                     for end in endpoints]
            ax.plot([end / 1000 for end in endpoints], means, color=colors[i], linewidth=2.5, label=short[i])
        ax.set(title=title, ylabel=ylabel, xlabel='Online observations (thousands)',
               xticks=[end / 1000 for end in endpoints])
        if key == 'development_joint':
            ax.set_ylim(-3, 103)
    for ax, (key, title, ylabel) in zip(axes[1], [
        ('cumulative_binding_seconds', '120,000 live binding observations', 'Seconds, including replay and speech'),
        ('graph_us_per_byte', 'Final graph generation, strict FP32', 'Microseconds / byte (lower is better)')]):
        for i, arm in enumerate(arms):
            mean = next(r for r in data['means'] if r['arm'] == arm and r['online_updates'] == endpoints[-1])[key]
            ax.bar(i, mean, color=colors[i], alpha=.32, width=.66)
            for j, seed in enumerate(seeds):
                value = next(r for r in data['rows'] if r['arm'] == arm and r['seed'] == seed
                             and r['online_updates'] == endpoints[-1])[key]
                ax.scatter(i + (j - 1) * .09, value, color=colors[i], marker=markers[j], s=52,
                           zorder=3, edgecolor='white', linewidth=.5)
        ax.set(title=title, ylabel=ylabel, xticks=range(3), xticklabels=labels)
    for ax in axes.flat:
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
        ax.axhline(0, color='#999999', linewidth=.7)
    fig.suptitle('Associative memory: longer exposure, every seed', fontsize=17, fontweight='bold', y=.98)
    fig.text(.5, .934, 'Same selected sources and exposure • 130,000 observations per model • three paired seeds',
             ha='center', fontsize=11, color='#444444')
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='upper center',
               bbox_to_anchor=(.5, .919), ncol=3, frameon=False)
    fig.legend([Line2D([0], [0], marker=m, color='#555555', linestyle='None', markersize=7) for m in markers],
               [str(seed) for seed in seeds], loc='lower center', bbox_to_anchor=(.5, .058), ncol=3,
               title='Markers and thin lines: individual seeds. Thick lines and bars: means.', frameon=False)
    fig.text(.5, .021, 'Binding uses box/bag answers; it does not assess general conversation. '
             'Reserved tests unused. No checkpoint selected by intermediate results.',
             ha='center', fontsize=9, color='#555555')
    fig.savefig(output, dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,default=Path('reports/associative-summary.json'))
    p.add_argument('--output',type=Path,default=Path('reports/associative-comparison.png'))
    a=p.parse_args()
    plot(a.source,a.output)
