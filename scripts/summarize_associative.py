"""Publish all declared models, retention, cost and numerical-check limitations."""
import argparse
from collections import Counter
from pathlib import Path
import statistics

from inspect_dynamics import inspect
from native_experiment import read, sha, write


def summarize(root):
    data = read(root/'comparison.json')
    p = data['protocol']
    assert p['status'] == 'declared_before_training' and p['online_endpoints'] == [34000]
    assert p['seeds'] == [1337, 2026, 31415] and len(data['runs']) == 9
    assert sha('build/synapticgenesis.exe') == p['executable_sha256']
    execution, oracle = read(root/'execution-check.json'), read(root/'learned-oracle.json')
    assert execution['passed'] and len(oracle['records']) == 9
    commands = read(root/'commands.json')
    assert len(commands) == data['native_commands'] == 54
    probes = Counter(Path(c[c.index('--probes')+1]).name for c in commands if c[1] == 'language-probes')
    assert probes == {'train.sgprobe':9, 'development.sgprobe':9, 'expanded-train-monitor.sgprobe':9}
    assert all('15659' not in ' '.join(c) for c in commands)
    rows, samples, dynamics = [], [], []
    for r in data['runs']:
        s, common = r['session'], r['common_session']
        directory = root/f'{r["seed"]}-{r["arm"]}'
        checkpoint = directory/'checkpoint-34000.ckpt'
        assert sha(checkpoint) == r['checkpoint_sha256']
        assert sha(directory/'checkpoint-10000.ckpt') == r['prerequisite_checkpoint_sha256']
        rows.append(dict(seed=r['seed'],arm=r['arm'],parameters=r['parameters'],
            train_joint=r['train']['joint_accuracy'],development_joint=r['development']['joint_accuracy'],
            development_greedy_joint=r['development']['greedy_exact_joint_accuracy'],
            development_item_accuracy=r['development']['accuracy'],
            development_context_erased_accuracy=r['development']['context_erased_accuracy'],
            development_answer_loss=r['development']['answer_loss_nats_per_byte'],
            expanded_monitor_joint=r['expanded-train-monitor']['joint_accuracy'],
            common_reader_loss=common['final_validation_loss'],final_reader_loss=s['final_validation_loss'],
            reader_change=s['final_validation_loss']-common['final_validation_loss'],
            foundation_seconds=common['elapsed_seconds'],live_segment_seconds=s['elapsed_seconds'],
            total_live_seconds=common['elapsed_seconds']+s['elapsed_seconds'],
            graph_us_per_byte=r['decode']['graph_us_per_byte'],regular_us_per_byte=r['decode']['regular_us_per_byte'],
            normal_tick_p95_ms=s['update_tick_p95_ms'],speech_tick_p95_ms=s['update_and_speech_tick_p95_ms'],
            observed_pairs=s['observed_pairs'],replay_pairs=s['replay_pairs'],global_updates=s['global_updates'],
            replay_updates=s['replay_updates'],generated_bytes=s['generated_bytes'],
            checkpoint_sha256=r['checkpoint_sha256']))
        samples.append(dict(seed=r['seed'],arm=r['arm'],prompt='The bird ',
                            first_192_generated_bytes=(directory/'decode/sample.txt').read_bytes().decode('latin1')[:192]))
        if r['arm'] == 'associative':
            dynamics.append(inspect(checkpoint))
    fields = ('train_joint','development_joint','development_greedy_joint','expanded_monitor_joint',
              'development_item_accuracy','development_context_erased_accuracy','development_answer_loss',
              'common_reader_loss','final_reader_loss','reader_change','foundation_seconds','live_segment_seconds',
              'total_live_seconds','graph_us_per_byte','regular_us_per_byte')
    means = [dict(arm=arm,**{key:statistics.mean(r[key] for r in rows if r['arm']==arm) for key in fields})
             for arm in p['arms']]
    paired = []
    for seed in p['seeds']:
        memory = next(r for r in rows if r['seed']==seed and r['arm']=='associative')
        for control in ('selective','wide-selective'):
            baseline = next(r for r in rows if r['seed']==seed and r['arm']==control)
            paired.append(dict(seed=seed,control=control,
                associative_minus_control={key:memory[key]-baseline[key] for key in fields}))
    result = dict(protocol=p,rows=rows,means=means,paired=paired,
                  execution_check=execution,cpu_oracle=oracle,executed_probe_commands=dict(probes),
                  full_development_cpu_audit_completed=False,
                  samples=samples,associative_dynamics=dynamics,
                  interpretation='All models and declared endpoints are reported. The CPU check covers one fixed '
                    'four-question group per model, not the entire development set. These results do not establish '
                    'general language competence, lifelong learning or a biological brain.')
    write('reports/associative-language.json',data)
    write('reports/associative-summary.json',result)
    write('reports/associative-learned-oracle.json',oracle)
    print('arm, dev, greedy, reader loss, reader change, live seconds, graph us/byte')
    for row in means:
        print(row['arm'],*[round(row[key],6) for key in
              ('development_joint','development_greedy_joint','final_reader_loss','reader_change',
               'live_segment_seconds','graph_us_per_byte')])


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path('runs/associative-screen-panel'))
    summarize(p.parse_args().root)
