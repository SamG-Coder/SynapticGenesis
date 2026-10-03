"""Static research figure for the matched architecture/exposure experiment."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


report = json.loads(Path('reports/selective-binding.json').read_text())
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout='constrained')
colors = {'gated': '#ba593e', 'selective': '#247585'}
labels = {'gated': 'Read gate', 'selective': 'Retention gate'}
for cell in ('gated', 'selective'):
    rows = sorted((r for r in report['runs'] if r['cell'] == cell), key=lambda r: r['online_updates'])
    updates = [r['online_updates'] / 1000 for r in rows]
    accuracy = [100 * r['development']['joint_accuracy'] for r in rows]
    losses = [r['session']['final_validation_loss'] for r in rows]
    axes[0].plot(updates, accuracy, 'o-', label=labels[cell], color=colors[cell], linewidth=2)
    axes[1].plot(updates, losses, 'o-', label=labels[cell], color=colors[cell], linewidth=2)
axes[0].set(title='New associations: development groups', ylabel='All four answers correct (%)', ylim=(-3, 103))
axes[1].set(title='Earlier reading: retention cost', ylabel='Cross-entropy (nats/byte; lower is better)')
for ax in axes:
    ax.set_xlabel('Total online observations (thousands)')
    ax.set_xticks([34, 67, 130])
    ax.grid(axis='y', alpha=.2)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(frameon=False)
fig.suptitle('SynapticGenesis — same starting weights, data and parameter count', fontsize=14)
fig.supxlabel('One seed (1337); 1,716,736 parameters; replay every 4 observations; reserved test not evaluated',
               fontsize=9)
fig.savefig('reports/selective-comparison.png', dpi=160)
