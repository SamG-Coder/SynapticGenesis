"""Plot the recorded three-founder teaching comparison (optional Matplotlib)."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def plot(report, output):
    data = json.loads(report.read_text())
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.spines.top': False,
                         'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.7))
    fig.patch.set_facecolor('#f7f8fb')
    colors = ['#2d6b98', '#9560ab', '#c27c28']
    for ax in axes:
        ax.set_facecolor('#f7f8fb')
        ax.grid(axis='y', color='#dde2ea', linewidth=.8)
        ax.set_xticks([0, 1], ['Uniform targets\n+ replay', 'Answer emphasis\n+ replay'])
        ax.set_xlim(-.25, 1.25)
    for index, founder in enumerate(data['founders']):
        color = colors[index]
        values = [founder['arms'][arm] for arm in ('uniform', 'emphasis')]
        axes[0].plot([0, 1], [100*x['probes']['paired_accuracy'] for x in values],
                     '-o', color=color, linewidth=1.7, markersize=7, label=str(founder['founder_seed']))
        axes[1].plot([0, 1], [x['session']['final_validation_loss'] for x in values],
                     '-o', color=color, linewidth=1.7, markersize=7)
    axes[0].set_ylim(-3, 100)
    axes[0].set_yticks([0, 20, 40, 60, 80, 100])
    axes[0].set_ylabel('Both reversed facts answered correctly (%)')
    axes[0].set_title('Context-dependent answers\nHigher is better', loc='left', fontsize=13, pad=12)
    axes[1].set_ylabel('Held-out reader cross-entropy (nats/byte)')
    axes[1].set_title('Earlier reading after teaching\nLower is better', loc='left', fontsize=13, pad=12)
    axes[0].legend(title='Founder seed', frameon=False, loc='upper left')
    fig.suptitle('Does stronger answer supervision help live learning?', x=.07, ha='left',
                 fontsize=17, fontweight='bold', y=.98)
    fig.text(.07, .035, 'Same observations and replay budget within each founder. 72 development items / 36 pairs.\n'
             'Separate reading phases; three exploratory runs per setting. Narrow first-object task; reserved test unused.',
             fontsize=10, color='#465063')
    fig.subplots_adjust(left=.08, right=.97, top=.78, bottom=.23, wspace=.34)
    fig.savefig(output, dpi=160, facecolor=fig.get_facecolor())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', type=Path, default=Path('reports/teaching-v1.json'))
    parser.add_argument('--output', type=Path, default=Path('reports/teaching-v1.png'))
    args = parser.parse_args()
    plot(args.report, args.output)
