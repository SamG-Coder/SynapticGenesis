"""Paired retention, book-loss tradeoff and complete live cost, with every seed."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from native_experiment import read


def plot(source, output):
    data = read(source); p = data['protocol']
    endpoints = [p['baseline_online_updates']]+p['online_endpoints']
    colors, markers = ['#267aa2', '#c56430'], ['o', 's', '^']
    names = ['Selective H512', 'Selective H588', 'Associative H512']
    divisor = 1000 if endpoints[-1] > 1000 else 1
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    fig.subplots_adjust(left=.065, right=.985, top=.87, bottom=.14, hspace=.58, wspace=.3)
    for col, architecture in enumerate(p['architectures']):
        ax = axes[0, col]
        for variant_index, variant in enumerate(p['variants']):
            for seed_index, seed in enumerate(p['seeds']):
                values = [next(r for r in data['rows'] if r['architecture'] == architecture and r['variant'] == variant and
                               r['seed'] == seed and r['online_updates'] == end)['development_joint']*100 for end in endpoints]
                ax.plot([e/divisor for e in endpoints], values, color=colors[variant_index], marker=markers[seed_index],
                        markersize=4, linewidth=.8, alpha=.55)
            means = [next(r for r in data['means'] if r['architecture'] == architecture and r['variant'] == variant and
                          r['online_updates'] == end)['development_joint']*100 for end in endpoints]
            ax.plot([e/divisor for e in endpoints], means, color=colors[variant_index], linewidth=2.2)
        ax.set(title=names[col]+' — retained binding', ylim=(-3, 103), ylabel='Complete groups (%)',
               xticks=[e/divisor for e in endpoints], xlabel='Online observations'+(' (thousands)' if divisor > 1 else ''))
        ax = axes[1, col]
        for book_index, book in enumerate(('narrative', 'reader', 'geography')):
            paired = [r for r in data['paired'] if r['architecture'] == architecture and r['online_updates'] == endpoints[-1]]
            values = [r['teacher_minus_control'][book+'_loss'] for r in paired]
            ax.bar(book_index, sum(values)/len(values), color=colors[1], alpha=.25, width=.65)
            for seed_index, value in enumerate(values):
                ax.scatter(book_index+(seed_index-(len(values)-1)/2)*.09, value, color=colors[1], marker=markers[seed_index], s=43)
        ax.axhline(.02, color='#555555', linestyle=':', linewidth=1)
        ax.set(title='Final book-loss tradeoff', ylabel='Teacher minus control (nats / byte)',
               xticks=range(3), xticklabels=['Narrative', 'Reader', 'Geography'])
        ax = axes[2, col]
        for index, variant in enumerate(p['variants']):
            values = [next(r for r in data['rows'] if r['architecture'] == architecture and r['variant'] == variant and
                           r['seed'] == seed and r['online_updates'] == endpoints[-1])['cumulative_narrative_seconds'] for seed in p['seeds']]
            ax.bar(index, sum(values)/len(values), color=colors[index], alpha=.3, width=.6)
            for seed_index, value in enumerate(values):
                ax.scatter(index+(seed_index-(len(values)-1)/2)*.09, value, marker=markers[seed_index], color=colors[index], s=43)
        ax.set(title='Complete added live-learning phase', ylabel='Seconds including replay and speech',
               xticks=[0, 1], xticklabels=['Ordinary replay', 'Frozen teacher'])
    for ax in axes.flat:
        ax.grid(axis='y', alpha=.18); ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
        ax.axhline(0, color='#888888', linewidth=.7)
    title = 'Frozen-self teaching: retention, new learning and cost'
    if p['status'] == 'smoke_only': title = 'Rehearsal only — '+title
    fig.suptitle(title, fontsize=18, fontweight='bold', y=.98)
    fig.text(.5, .942, 'Identical parents and selected-source exposure • temperature 2 • teacher strength 0.5 • fixed endpoints', ha='center', fontsize=11)
    fig.legend([Line2D([0], [0], color=c, lw=3) for c in colors], ['Ordinary replay', 'Frozen-self teacher'],
               loc='upper center', bbox_to_anchor=(.5, .928), ncol=2, frameon=False)
    fig.legend([Line2D([0], [0], marker=markers[i], color='#555555', ls='None') for i in range(len(p['seeds']))],
               [str(s) for s in p['seeds']], loc='lower center', bbox_to_anchor=(.5, .055), ncol=3,
               title='Points and thin lines: each seed. Thick lines and bars: means.', frameon=False)
    fig.text(.5, .025, 'Book-loss deltas below zero favor teaching; dotted line is the declared +0.02 margin. '
             'Sampled byte windows and repeated development probes; reserved tests unused.', ha='center', fontsize=9.5, color='#555555')
    fig.savefig(output, dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    plot(a.source, a.output)
