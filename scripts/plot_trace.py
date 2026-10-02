"""Plot the recorded cue-recall and binding controls (optional Matplotlib)."""
import json
from pathlib import Path
import statistics

import matplotlib.pyplot as plt


def plot():
    memory = json.loads(Path('reports/trace-memory.json').read_text())
    binding = json.loads(Path('reports/trace-binding.json').read_text())
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':11,
                         'axes.spines.top':False, 'axes.spines.right':False})
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.9))
    fig.patch.set_facecolor('#f7f8fb')
    colors = {'lif':'#3a6c98', 'alif':'#9b6ba8', 'trace':'#168071'}
    labels = {'lif':'LIF', 'alif':'Adaptive LIF', 'trace':'LIF + spike trace'}
    for ax in axes:
        ax.set_facecolor('#f7f8fb')
        ax.grid(axis='y', color='#dde2ea', linewidth=.8)
        ax.set_ylim(-3, 105)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_ylabel('Accuracy (%)')
    for offset, cell in enumerate(colors):
        xs, means, lower, upper = [], [], [], []
        for i, delay in enumerate(memory['delays']):
            ys = [100*r['accuracy'] for r in memory['runs'] if r['cell_option']==cell and r['delay']==delay]
            mean = statistics.mean(ys)
            xs.append(i + (offset-1)*.065)
            means.append(mean)
            lower.append(mean-min(ys))
            upper.append(max(ys)-mean)
        axes[0].errorbar(xs, means, yerr=[lower, upper], fmt='-o', color=colors[cell],
                         label=labels[cell], markersize=6, linewidth=1.8, capsize=4)
    axes[0].axhline(50, color='#777777', linewidth=1, linestyle=':')
    axes[0].set_xticks([0,1,2], memory['delays'])
    axes[0].set_xlabel('Distractor bytes between cue and query')
    axes[0].set_title('Remembering one cue\nThree seeds; bars show min–max', loc='left', pad=12)
    axes[0].legend(frameon=False, fontsize=9, loc='lower left')
    for i, arm in enumerate(binding['arms']):
        color = colors[arm['cell']]
        value = 100*arm['development']['accuracy']
        axes[1].bar(i-.16, value, .30, color=color, alpha=.75)
        joint = 100*arm['development']['joint_accuracy']
        axes[1].plot(i+.16, joint, 'D', color=color, markersize=7)
        axes[1].text(i-.16, value+3, f'{value:.1f}%', ha='center', fontsize=10)
    axes[1].axhline(50, color='#777777', linewidth=1, linestyle=':')
    axes[1].set_xticks([0,1,2], ['LIF','Adaptive\nLIF','LIF +\nspike trace'])
    axes[1].set_title('Binding facts to the queried object\nOne seed; 576 development questions', loc='left', pad=12)
    axes[1].text(.04, .82, 'Bars: individual answers\nDiamonds: all four answers correct (0%)',
                 transform=axes[1].transAxes, fontsize=9, color='#465063')
    fig.suptitle('A fading spike trace helps cue recall; binding remains unresolved',
                 x=.065, ha='left', fontsize=16, fontweight='bold', y=.98)
    fig.text(.065, .035, 'Cue models: 64 channels / 128 neurons / 2 blocks; 2,000 updates. Language models: 256 / 512 / 4; 34,000 live observations.\n'
             'Source and replay counts matched. Reserved language test unused. Different tasks and model sizes; no general-intelligence claim.',
             fontsize=9.5, color='#465063')
    fig.subplots_adjust(left=.065, right=.97, top=.78, bottom=.23, wspace=.28)
    fig.savefig('reports/trace-comparison.png', dpi=160, facecolor=fig.get_facecolor())


if __name__ == '__main__':
    plot()
