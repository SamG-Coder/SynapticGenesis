"""Standalone research figure; all declared seeds and measured checkpoints."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


data=json.loads(Path('reports/stage-replay-language.json').read_text())
ancestors={r['seed']:r['session']['final_validation_loss'] for r in data['shared_ancestors']}
fig,axes=plt.subplots(1,2,figsize=(11,5.1),layout='constrained')
colors={'reservoir':'#b4573c','stage':'#257e89'}
names={'reservoir':'Stream reservoir','stage':'Balanced source stages'}
styles={1337:'-',2026:'--',31415:':'}
for seed in data['protocol']['seeds']:
    for policy in ('reservoir','stage'):
        rows=sorted((r for r in data['runs'] if r['seed']==seed and r['policy']==policy),
                    key=lambda r:r['online_updates'])
        x=[r['online_updates']/1000 for r in rows]
        delta=[r['session']['final_validation_loss']-ancestors[seed] for r in rows]
        accuracy=[100*r['development']['joint_accuracy'] for r in rows]
        opts=dict(color=colors[policy],linestyle=styles[seed],marker='o',markersize=4,linewidth=1.8)
        axes[0].plot([6]+x,[0]+delta,**opts)
        axes[1].plot(x,accuracy,**opts)
axes[0].set(title='Earlier reading: change from the shared ancestor',
            ylabel='Cross-entropy increase (nats/byte; lower is better)')
axes[0].axhline(0,color='#888888',linewidth=.8)
axes[1].set(title='New associations: development groups',ylabel='All four answers correct (%)',ylim=(-3,103))
for ax in axes:
    ax.set_xlabel('Total online observations (thousands)')
    ax.set_xticks([6,34,67,130])
    ax.grid(axis='y',alpha=.2)
    ax.spines[['top','right']].set_visible(False)
color_handles=[Line2D([],[],color=colors[p],label=names[p],linewidth=2) for p in colors]
seed_handles=[Line2D([],[],color='#444444',linestyle=styles[s],label=f'Seed {s}') for s in styles]
axes[0].legend(handles=color_handles,frameon=False,loc='upper left')
axes[1].legend(handles=seed_handles,frameon=False,loc='upper left')
fig.suptitle('SynapticGenesis — balanced replay across three shared ancestors',fontsize=14)
fig.supxlabel('1,024 slots and equal replay-update budgets; replayed byte counts differ; reserved test not evaluated',fontsize=9)
fig.savefig('reports/stage-replay-comparison.png',dpi=160)
