"""Plot all three seeds in the read-only learned-memory intervention."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from native_experiment import read


def plot(source, output):
    data = read(source)
    modes, seeds = data['protocol']['modes'], data['protocol']['seeds']
    colors = ['#b85c3c', '#33839a', '#687781']
    labels = ['Normal\nCPU reference', 'Discard matrix history\nKeep current write/read',
              'Zero matrix read\nKeep output bias']
    markers = ['o', 's', '^']
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 6.4))
    fig.subplots_adjust(left=.075, right=.985, top=.77, bottom=.28, wspace=.23)
    for ax, (key, title) in zip(axes, [('joint_accuracy', 'Candidate ranking'),
                                     ('greedy_joint_accuracy', 'Unconstrained four-byte answers')]):
        for index, mode in enumerate(modes):
            mean = next(r for r in data['means'] if r['mode'] == mode)[key] * 100
            ax.bar(index, mean, color=colors[index], alpha=.30, width=.68)
            for offset, seed in enumerate(seeds):
                row = next(r for r in data['rows'] if r['seed'] == seed and r['mode'] == mode)
                ax.scatter(index + (offset - 1) * .09, row[key] * 100, marker=markers[offset],
                           color=colors[index], s=52, zorder=3, edgecolor='white', linewidth=.5)
        ax.set(title=title, ylabel='All four answers correct (%)', ylim=(-3, 105),
               xticks=range(3), xticklabels=labels)
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Does learned binding depend on fast matrix history?', y=.975,
                 fontsize=16, fontweight='bold')
    fig.text(.5, .91, 'Same trained weights • all 576 development questions • three final models',
             ha='center', fontsize=11, color='#444444')
    fig.text(.5, .854,
             f'Normal CPU/native: {data["normal_score_failure_questions"]:,}/{data["normal_questions"]:,} questions fail score tolerance; '
             f'{data["normal_candidate_disagreements"]} choice and {data["normal_greedy_disagreements"]} greedy disagreements',
             ha='center', fontsize=10, color='#444444')
    fig.legend([Line2D([0], [0], marker=marker, color='#555555', linestyle='None', markersize=7)
                for marker in markers], [str(seed) for seed in seeds],
               loc='lower center', bbox_to_anchor=(.5, .10), ncol=3,
               title='Individual seeds; bars show means', frameon=False)
    fig.text(.5, .035, 'Interventions measure trained-model reliance, not retrained architecture quality. '
             'No reserved test or general-conversation assessment.',
             ha='center', fontsize=9, color='#555555')
    fig.savefig(output, dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=Path('reports/associative-history.json'))
    parser.add_argument('--output', type=Path, default=Path('reports/associative-history.png'))
    args = parser.parse_args()
    plot(args.source, args.output)
