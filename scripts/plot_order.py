"""Plot every declared seed and checkpoint of the matched ordering comparison."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def plot(source, output):
    data = json.loads(source.read_text())
    protocol = data['protocol']
    seeds, arms = protocol['seeds'], protocol['arms']
    assert arms == ['ramped', 'shuffled']
    assert len(data['runs']) == len(seeds)*len(arms)*len(protocol['online_endpoints'])
    ancestors = {row['seed']: row['session']['final_validation_loss']
                 for row in data['shared_ancestors']}
    colors = {'ramped': '#256f8c', 'shuffled': '#ad543b'}
    names = {'ramped': 'Gradual: 6 / 12 / 24 objects', 'shuffled': 'Same instances, shuffled'}
    styles = {1337: '-', 2026: '--', 31415: ':'}
    common_end = protocol['prerequisite_end']
    ends = protocol['online_endpoints']
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 9.4))
    fig.subplots_adjust(left=.09, right=.98, bottom=.17, top=.865, hspace=.40, wspace=.24)
    for seed in seeds:
        for arm in arms:
            rows = sorted((row for row in data['runs'] if row['seed'] == seed and row['arm'] == arm),
                          key=lambda row: row['online_updates'])
            assert [row['online_updates'] for row in rows] == ends
            x = [row['online_updates']/1000 for row in rows]
            style = dict(color=colors[arm], linestyle=styles[seed], marker='o', markersize=4, linewidth=1.8)
            axes[0, 0].plot(x, [100*row['development']['joint_accuracy'] for row in rows], **style)
            axes[0, 1].plot(x, [100*row['development']['greedy_exact_joint_accuracy'] for row in rows], **style)
            axes[1, 0].plot(x, [100*row['expanded-train-monitor']['joint_accuracy'] for row in rows], **style)
            change = [row['session']['final_validation_loss']-ancestors[seed] for row in rows]
            axes[1, 1].plot([common_end/1000]+x, [0]+change, **style)
    axes[0, 0].set(title='Fixed development questions: candidate ranking', ylabel='All four answers correct (%)')
    axes[0, 1].set(title='Fixed development questions: free answer generation', ylabel='All four exact answers (%)')
    axes[1, 0].set(title='Expanded-training monitor: fixed 432 groups', ylabel='All four answers correct (%)')
    axes[1, 1].set(title='Earlier reading: change from shared 10k foundation',
                   ylabel='Loss increase (nats/byte; lower is better)')
    if protocol['status'] == 'smoke_only':
        axes[1, 1].set_title('Earlier reading: change from shared smoke foundation')
    axes[1, 1].axhline(0, color='#888888', linewidth=.8)
    for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
        ax.set_ylim(-3, 103)
    for ax in axes.flat:
        ax.set_xlabel('Total online observations (thousands)')
        ax.set_xticks([v/1000 for v in [common_end]+ends])
        ax.grid(axis='y', alpha=.2)
        ax.spines[['top', 'right']].set_visible(False)
        for end in ends[1:-1]:
            ax.axvline(end/1000, color='#aaaaaa', linewidth=.7, linestyle=':', zorder=0)
    color_handles = [Line2D([], [], color=colors[arm], label=names[arm], linewidth=2) for arm in arms]
    seed_handles = [Line2D([], [], color='#444444', linestyle=styles[seed], label=f'Seed {seed}') for seed in seeds]
    fig.legend(handles=color_handles+seed_handles, loc='lower center', bbox_to_anchor=(.5, .085),
               ncol=5, frameon=False)
    subtitle = ('SMOKE CHECK ONLY: no learning-quality claim' if protocol['status'] == 'smoke_only' else
                'Same online lessons and repetitions; common ancestor per seed; reserved tests unused')
    fig.suptitle('SynapticGenesis: gradual associations versus shuffled practice\n'+subtitle,
                 fontsize=13, y=.97, linespacing=1.6)
    fig.text(.5, .046, 'Dotted vertical lines: curriculum boundaries. Lines connect scheduled measurements.',
             ha='center', fontsize=10, color='#555555')
    fig.text(.5, .020, 'The expanded monitor includes not-yet-observed examples at earlier checkpoints.',
             ha='center', fontsize=10, color='#555555')
    fig.savefig(output, dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, default=Path('reports/curriculum-order-language.json'))
    p.add_argument('--output', type=Path, default=Path('reports/curriculum-order-comparison.png'))
    a = p.parse_args()
    plot(a.input, a.output)
