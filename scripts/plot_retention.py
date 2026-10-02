"""Plot saved experiment evidence; optional Matplotlib, no model computation."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', type=Path, default=Path('reports/retention-v1.json'))
    parser.add_argument('--out', type=Path, default=Path('reports/retention-v1.png'))
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    rows = report['aggregate']
    names = ['No replay', 'Repeat current chunk', 'Replay earlier windows', 'Replay + SI',
             'Replay + slower core', 'Replay + lower learning rate', 'Lower rate, no replay']
    assert [r['arm'] for r in rows] == ['none','current','reservoir','reservoir_si','reservoir_slow_core',
                                       'reservoir_low_lr','none_low_lr']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11})
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.4), sharey=True)
    fig.subplots_adjust(left=.27, right=.96, bottom=.2, top=.78, wspace=.19)
    fig.suptitle('Retaining earlier reading while learning science', fontsize=17, x=.54, y=.96)
    fig.text(.54, .88, f"{len(report['seeds'])} seeds · Same new-source exposure · 1.19M parameters",
             ha='center', fontsize=11, color='#475569')
    for ax, metric, title in zip(axes, ['old_loss', 'new_loss'], ['Earlier held-out reader', 'New held-out geography book']):
        for i, row in enumerate(rows):
            chosen = row['arm'] == 'reservoir_low_lr'
            color = '#087f70' if chosen else '#486a96'
            data = row[metric]
            ax.errorbar(data['mean'], i, xerr=data['sample_sd'], fmt='o', color=color,
                        markersize=7 if chosen else 5.5, capsize=3, elinewidth=1.5, zorder=3)
        ax.set_title(title, fontsize=12, pad=12)
        ax.set_yticks(range(len(rows)), names)
        ax.set_xlabel('Cross-entropy (nats / byte)', fontsize=10)
        ax.grid(axis='x', color='#e2e8f0', zorder=0)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.spines['bottom'].set_color('#94a3b8')
        ax.tick_params(axis='y', length=0, pad=12)
        ax.tick_params(axis='x', labelsize=9)
    axes[0].set_ylim(len(rows)-.45, -.65)
    axes[0].set_xlim(2.16, 2.73)
    axes[1].set_xlim(1.84, 2.1)
    fig.text(.54, .08, 'Lower is better. Points show means; bars show sample standard deviations.',
             ha='center', fontsize=10, color='#475569')
    fig.text(.54, .035, 'Exploratory byte-language evaluation on selected historical books.',
             ha='center', fontsize=9, color='#64748b')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160, facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    main()
