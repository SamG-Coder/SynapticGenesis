"""Standalone figure of all declared seeds and observation endpoints."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


p = argparse.ArgumentParser()
p.add_argument('--input', type=Path, default=Path('reports/lesson-diversity-language.json'))
p.add_argument('--output', type=Path, default=Path('reports/lesson-diversity-comparison.png'))
a = p.parse_args()
data = json.loads(a.input.read_text())
ancestors = {row['seed']: row['session']['final_validation_loss'] for row in data['shared_ancestors']}
colors = {'control': '#b4573c', 'diversity': '#257e89'}
names = {'control': 'Original: 6 objects', 'diversity': 'Expanded: 24 objects'}
styles = {1337: '-', 2026: '--', 31415: ':'}
fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), layout='constrained')
for seed in data['protocol']['seeds']:
    for arm in data['protocol']['arms']:
        rows = sorted((row for row in data['runs'] if row['seed'] == seed and row['arm'] == arm),
                      key=lambda row: row['online_updates'])
        x = [row['online_updates']/1000 for row in rows]
        opts = dict(color=colors[arm], linestyle=styles[seed], marker='o', markersize=4, linewidth=1.8)
        axes[0, 0].plot(x, [100*row['development']['joint_accuracy'] for row in rows], **opts)
        axes[0, 1].plot(x, [100*row['development']['greedy_exact_joint_accuracy'] for row in rows], **opts)
        axes[1, 0].plot(x, [100*row['expanded-train-monitor']['joint_accuracy'] for row in rows], **opts)
        delta = [row['session']['final_validation_loss']-ancestors[seed] for row in rows]
        axes[1, 1].plot([data['protocol']['reading_end']/1000]+x, [0]+delta, **opts)
axes[0, 0].set(title='Frozen development questions: candidate ranking', ylabel='All four answers correct (%)')
axes[0, 1].set(title='Frozen development questions: free answer generation', ylabel='All four exact answers (%)')
axes[1, 0].set(title='Expanded-training monitor: preselected 432 groups', ylabel='All four answers correct (%)')
axes[1, 1].set(title='Earlier reading: change from the shared ancestor',
               ylabel='Loss increase (nats/byte; lower is better)')
axes[1, 1].axhline(0, color='#888888', linewidth=.8)
for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
    ax.set_ylim(-3, 103)
for ax in axes.flat:
    ax.set_xlabel('Total online observations (thousands)')
    ax.set_xticks([v/1000 for v in [data['protocol']['reading_end']]+data['protocol']['online_endpoints']])
    ax.grid(axis='y', alpha=.2)
    ax.spines[['top', 'right']].set_visible(False)
color_handles = [Line2D([], [], color=colors[arm], label=names[arm], linewidth=2) for arm in colors]
seed_handles = [Line2D([], [], color='#444444', linestyle=styles[seed], label=f'Seed {seed}') for seed in data['protocol']['seeds']]
fig.legend(handles=color_handles+seed_handles, loc='outside lower center', ncol=5, frameon=False)
subtitle = ('SMOKE CHECK ONLY: no learning-quality claim' if data['protocol']['status'] == 'smoke_only' else
            'Equal update budgets; vocabulary, corpus size and repetition change together; reserved test unused')
fig.suptitle('SynapticGenesis: selected lesson diversity\n'+subtitle, fontsize=13)
fig.savefig(a.output, dpi=160)
