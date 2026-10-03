"""Publish all declared models, retention, cost and numerical-check limitations."""
import argparse
from collections import Counter
from pathlib import Path
import statistics

from inspect_dynamics import inspect
from native_experiment import read, sha, write


def summarize(root, exe, prefix=None):
    data = read(root/'comparison.json')
    p = data['protocol']
    assert p['status'] == 'declared_before_training'
    endpoints = p['online_endpoints']
    assert endpoints == ([34000, 67000, 130000] if p.get('longitudinal') else [34000])
    assert p['seeds'] == [1337, 2026, 31415] and len(data['runs']) == 9 * len(endpoints)
    assert sha(exe) == p['executable_sha256']
    prefix = prefix or ('associative-long' if p.get('longitudinal') else 'associative')
    execution, oracle = read(root/'execution-check.json'), read(root/'learned-oracle.json')
    assert execution['passed'] and len(oracle['records']) == 9
    commands = read(root/'commands.json')
    assert len(commands) == data['native_commands'] == 9 * (1 + 5 * len(endpoints))
    probes = Counter(Path(c[c.index('--probes')+1]).name for c in commands if c[1] == 'language-probes')
    assert probes == {name:9 * len(endpoints) for name in
                      ('train.sgprobe', 'development.sgprobe', 'expanded-train-monitor.sgprobe')}
    assert all('15659' not in ' '.join(c) for c in commands)
    rows, samples, dynamics = [], [], []
    for r in data['runs']:
        s, common = r['session'], r['common_session']
        endpoint = r['online_updates']
        directory = root/f'{r["seed"]}-{r["arm"]}'
        checkpoint = directory/f'checkpoint-{endpoint}.ckpt'
        assert sha(checkpoint) == r['checkpoint_sha256']
        assert sha(directory/'checkpoint-10000.ckpt') == r['prerequisite_checkpoint_sha256']
        cumulative = r.get('cumulative_binding_seconds', s['elapsed_seconds'])
        rows.append(dict(seed=r['seed'],arm=r['arm'],online_updates=endpoint,parameters=r['parameters'],
            train_joint=r['train']['joint_accuracy'],development_joint=r['development']['joint_accuracy'],
            development_greedy_joint=r['development']['greedy_exact_joint_accuracy'],
            development_item_accuracy=r['development']['accuracy'],
            development_context_erased_accuracy=r['development']['context_erased_accuracy'],
            development_answer_loss=r['development']['answer_loss_nats_per_byte'],
            expanded_monitor_joint=r['expanded-train-monitor']['joint_accuracy'],
            common_reader_loss=common['final_validation_loss'],final_reader_loss=s['final_validation_loss'],
            reader_change=s['final_validation_loss']-common['final_validation_loss'],
            foundation_seconds=common['elapsed_seconds'],live_segment_seconds=s['elapsed_seconds'],
            cumulative_binding_seconds=cumulative,total_live_seconds=common['elapsed_seconds']+cumulative,
            graph_us_per_byte=r['decode']['graph_us_per_byte'],regular_us_per_byte=r['decode']['regular_us_per_byte'],
            normal_tick_p95_ms=s['update_tick_p95_ms'],speech_tick_p95_ms=s['update_and_speech_tick_p95_ms'],
            observed_pairs=s['observed_pairs'],replay_pairs=s['replay_pairs'],global_updates=s['global_updates'],
            replay_updates=s['replay_updates'],generated_bytes=s['generated_bytes'],
            checkpoint_sha256=r['checkpoint_sha256']))
        if endpoint == endpoints[-1]:
            decode = directory/(f'decode-{endpoint}' if p.get('longitudinal') else 'decode')
            samples.append(dict(seed=r['seed'],arm=r['arm'],prompt='The bird ',
                                first_192_generated_bytes=(decode/'sample.txt').read_bytes().decode('latin1')[:192]))
            if r['arm'] == 'associative':
                dynamics.append(inspect(checkpoint))
    fields = ('train_joint','development_joint','development_greedy_joint','expanded_monitor_joint',
              'development_item_accuracy','development_context_erased_accuracy','development_answer_loss',
              'common_reader_loss','final_reader_loss','reader_change','foundation_seconds','live_segment_seconds',
              'cumulative_binding_seconds','total_live_seconds','graph_us_per_byte','regular_us_per_byte')
    means = [dict(arm=arm,online_updates=endpoint,
                  **{key:statistics.mean(r[key] for r in rows if r['arm']==arm and r['online_updates']==endpoint)
                     for key in fields}) for endpoint in endpoints for arm in p['arms']]
    paired = []
    for seed in p['seeds']:
        for endpoint in endpoints:
            memory = next(r for r in rows if r['seed']==seed and r['arm']=='associative' and r['online_updates']==endpoint)
            for control in ('selective','wide-selective'):
                baseline = next(r for r in rows if r['seed']==seed and r['arm']==control and r['online_updates']==endpoint)
                paired.append(dict(seed=seed,control=control,online_updates=endpoint,
                    associative_minus_control={key:memory[key]-baseline[key] for key in fields}))
    result = dict(protocol=p,rows=rows,means=means,paired=paired,
                  execution_check=execution,cpu_oracle=oracle,executed_probe_commands=dict(probes),
                  full_development_cpu_audit_completed=False,
                  samples=samples,associative_dynamics=dynamics,
                  interpretation='All models and declared endpoints are reported. The CPU check covers one fixed '
                    'four-question group per model, not the entire development set. These results do not establish '
                    'general language competence, lifelong learning or a biological brain.')
    write(f'reports/{prefix}-language.json',data)
    write(f'reports/{prefix}-summary.json',result)
    write(f'reports/{prefix}-learned-oracle.json',oracle)
    if p.get('longitudinal'):
        write(f'reports/{prefix}-execution.json',execution)
    print('endpoint, arm, dev, greedy, reader loss, reader change, binding seconds, graph us/byte')
    for row in means:
        print(row['online_updates'],row['arm'],*[round(row[key],6) for key in
              ('development_joint','development_greedy_joint','final_reader_loss','reader_change',
               'cumulative_binding_seconds','graph_us_per_byte')])


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path('runs/associative-screen-panel'))
    p.add_argument('--exe',type=Path,default=Path('build/synapticgenesis.exe'))
    p.add_argument('--prefix')
    a = p.parse_args()
    summarize(a.root, a.exe, a.prefix)
