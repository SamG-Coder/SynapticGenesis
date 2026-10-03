"""Plot language/skill retention and cost for every declared continuation seed."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from native_experiment import read


def plot(source, output):
    data = read(source)
    p = data['protocol']
    endpoints = [p['baseline_online_updates']] + p['online_endpoints']
    colors, markers = ['#687781', '#33839a', '#b85c3c'], ['o', 's', '^']
    names = ['Selective H512', 'Selective H588', 'Associative H512']
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 10))
    fig.subplots_adjust(left=.065, right=.985, top=.82, bottom=.19, hspace=.50, wspace=.30)
    panels = [(axes[0, 0], 'narrative_change', 1, 'New narrative loss change'),
              (axes[0, 1], 'reader_change', 1, 'Earlier-reader loss change'),
              (axes[0, 2], 'geography_change', 1, 'Geography loss change'),
              (axes[1, 0], 'development_joint', 100, 'Retained binding: all four answers')]
    for ax, key, scale, title in panels:
        for index, arm in enumerate(p['arms']):
            for offset, seed in enumerate(p['seeds']):
                values = [next(r for r in data['rows'] if r['arm'] == arm and r['seed'] == seed
                               and r['online_updates'] == end)[key] * scale for end in endpoints]
                ax.plot([end / 1000 for end in endpoints], values, color=colors[index],
                        marker=markers[offset], markersize=4.5, alpha=.65, linewidth=.8)
            means = [next(r for r in data['means'] if r['arm'] == arm and r['online_updates'] == end)[key] * scale
                     for end in endpoints]
            ax.plot([end / 1000 for end in endpoints], means, color=colors[index], linewidth=2.5, label=names[index])
        ax.set(title=title, xlabel='Online observations (thousands)', xticks=[end / 1000 for end in endpoints],
               ylabel='Correct groups (%)' if scale == 100 else 'Nats / byte change (lower is better)')
        if scale == 100:
            ax.set_ylim(-3, 103)
    for ax, key, title, ylabel in [(axes[1, 1], 'cumulative_narrative_seconds', '60,000 live narrative observations',
                                    'Seconds, including replay and speech'),
                                   (axes[1, 2], 'graph_us_per_byte', 'Final graph generation, strict FP32',
                                    'Microseconds / byte (lower is better)')]:
        for index, arm in enumerate(p['arms']):
            mean = next(r for r in data['means'] if r['arm'] == arm and r['online_updates'] == endpoints[-1])[key]
            ax.bar(index, mean, color=colors[index], alpha=.32, width=.66)
            for offset, seed in enumerate(p['seeds']):
                value = next(r for r in data['rows'] if r['arm'] == arm and r['seed'] == seed
                             and r['online_updates'] == endpoints[-1])[key]
                ax.scatter(index + (offset - 1) * .09, value, color=colors[index], marker=markers[offset],
                           s=52, zorder=3, edgecolor='white', linewidth=.5)
        ax.set(title=title, ylabel=ylabel, xticks=range(3),
               xticklabels=['Selective\nH512', 'Selective\nH588', 'Associative\nH512'])
    for ax in axes.flat:
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
        ax.axhline(0, color='#999999', linewidth=.7)
    fig.suptitle('Selected narrative learning: new language, retained skills and cost',
                 fontsize=18, fontweight='bold', y=.98)
    fig.text(.5, .935, 'Continue every parent • seven selected books • matched new exposure • three paired seeds',
             ha='center', fontsize=11, color='#444444')
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='upper center', bbox_to_anchor=(.5, .919),
               ncol=3, frameon=False)
    fig.legend([Line2D([0], [0], marker=marker, color='#555555', linestyle='None', markersize=7)
                for marker in markers], [str(seed) for seed in p['seeds']],
               loc='lower center', bbox_to_anchor=(.5, .055), ncol=3, frameon=False,
               title='Markers and thin lines: individual seeds. Thick lines and bars: means.')
    fig.text(.5, .015, 'Loss change is relative to each model at 130,000 observations. '
             'Sampled 128-byte windows; no general-conversation claim. Reserved tests unused.',
             ha='center', fontsize=10, color='#555555')
    fig.savefig(output, dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=Path('reports/narrative-summary.json'))
    parser.add_argument('--output', type=Path, default=Path('reports/narrative-comparison.png'))
    args = parser.parse_args()
    plot(args.source, args.output)
