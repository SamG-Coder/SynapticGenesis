"""Collect replay mechanism checks, the stopped pilot and complete language evidence."""
import argparse
from array import array
import hashlib
import json
from pathlib import Path
import statistics

from experiment_checkpoint import checkpoint, distribution


def read(path):
    return json.loads(Path(path).read_text())


def write(name, value):
    Path('reports',name).write_text(json.dumps(value,indent=2)+'\n')


def validation():
    executable_sha=hashlib.sha256(Path('build/synapticgenesis.exe').read_bytes()).hexdigest()
    protocol=read('runs/stage-replay-shared-panel/protocol.json')
    if executable_sha != protocol['executable_sha256']:
        raise ValueError('Historical replay collection requires its recorded executable and validation artifacts; '
                         'do not relabel earlier results with a newer build.')
    sources = dict(native_policy='build/stage-replay-test-results/native.json',
                   cli_policy='runs/stage-replay-cli-v3/result.json',
                   curriculum_extension='runs/stage-replay-extension-v2/result.json',
                   population='runs/stage-replay-population-v3/result.json')
    report = {name:read(path) for name,path in sources.items()}
    assert all(value['passed'] for value in report.values())
    assert '100% tests passed out of 13' in Path('build-stage-replay-final.log').read_text(encoding='utf-8-sig')
    report.update(native_suites_passed=13,
                  executable_sha256=executable_sha)
    write('stage-replay-validation.json',report)
    old = []
    for cell in ('gated','selective'):
        for step in (34000,67000,130000):
            path=Path(f'runs/selective-binding-panel/{cell}/checkpoint-{step}.ckpt')
            old.append(dict(cell=cell, online_updates=step, stored_windows_by_source_stage=distribution(path,[5,101,1829]),
                            checkpoint_sha256=checkpoint(path)[3]))
    write('replay-dilution.json',dict(source='Earlier selective-trace experiment, not the new policy comparison.',
                                  stage_order=['reading','prerequisites','binding'],runs=old))
    root=Path('runs/stage-replay-panel')
    initial=[checkpoint(root/f'1337-{policy}/initial.ckpt') for policy in ('reservoir','stage')]
    learned=[checkpoint(root/f'1337-{policy}/stage-1.ckpt') for policy in ('reservoir','stage')]
    assert initial[0][2]==initial[1][2]
    p=learned[0][0][14]
    values=[array('f',x[2]) for x in learned]
    errors={}
    for name,a,b in [('weights',0,p),('first_moment',p,2*p),('second_moment',2*p,3*p),('recurrence',3*p,len(values[0]))]:
        errors[name]=max(abs(x-y) for x,y in zip(values[0][a:b],values[1][a:b]))
    pilot=dict(status='Stopped; not used as a complete policy comparison.',
               reason='Identical initialization and first-stage data/replay history produced divergent learned arrays. '
                      'Cause not fully isolated; replaced by a common learned ancestor per seed.',
               protocol=read(root/'protocol.json'),
               initial_arrays_identical=True, first_stage_checkpoint_hashes=[x[3] for x in learned],
               first_stage_policy_rng_identical=learned[0][1][4]==learned[1][1][4],
               first_stage_descriptors_identical=learned[0][1][16:]==learned[1][1][22:],
               first_stage_max_errors=errors,
               limits='Interrupted intermediate rows are not accepted as completed observation endpoints.')
    write('replay-control-pilot.json',pilot)
    return report


def main(validation_only):
    checked=validation()
    if validation_only:
        print('Collected 13-suite validation and stopped-pilot evidence.')
        return
    data=read('runs/stage-replay-shared-panel/comparison.json')
    assert data['protocol']['executable_sha256']==checked['executable_sha256']
    assert data['protocol']['seeds']==[1337,2026,31415] and len(data['runs'])==18
    assert not data['protocol']['reserved_test_evaluated']
    ancestors={r['seed']:r for r in data['shared_ancestors']}
    rows=[]
    for seed in data['protocol']['seeds']:
        for policy in ('reservoir','stage'):
            points=[r for r in data['runs'] if r['seed']==seed and r['policy']==policy]
            final=next(r for r in points if r['online_updates']==130000)
            assert final['ancestor_checkpoint_sha256']==ancestors[seed]['checkpoint_sha256']
            session=final['session']
            rows.append(dict(seed=seed, policy=policy,
                             ancestor_reader_loss=ancestors[seed]['session']['final_validation_loss'],
                             final_reader_loss=session['final_validation_loss'],
                             reader_loss_change=session['final_validation_loss']-ancestors[seed]['session']['final_validation_loss'],
                             train_joint=final['train']['joint_accuracy'],
                             development_joint=final['development']['joint_accuracy'],
                             development_greedy_joint=final['development']['greedy_exact_joint_accuracy'],
                             live_segment_seconds=sum(r['session']['elapsed_seconds'] for r in points),
                             replay_pairs=session['replay_pairs'], replay_updates=session['replay_updates'],
                             observed_pairs=session['observed_pairs'], generated_bytes=session['generated_bytes'],
                             replay_state_bytes=session['replay_state_bytes'],
                             final_tick_p95_ms=session['update_tick_p95_ms'],
                             final_speech_tick_p95_ms=session['update_and_speech_tick_p95_ms']))
    means=[]
    for policy in ('reservoir','stage'):
        selected=[r for r in rows if r['policy']==policy]
        means.append(dict(policy=policy, **{key:statistics.mean(r[key] for r in selected)
                     for key in ('final_reader_loss','reader_loss_change','train_joint','development_joint',
                                 'development_greedy_joint','live_segment_seconds','replay_pairs')}))
    write('stage-replay-language.json',data)
    summary=dict(protocol=data['protocol'], final_rows=rows, means=means,
                 cost_boundary='Live segments after the shared 6000-observation ancestor; setup and scoring excluded. '
                               'Replay update and slot counts match; replay target-byte counts differ.',
                 no_general_language_or_biological_equivalence_claim=True)
    write('stage-replay-summary.json',summary)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--validation-only',action='store_true')
    main(p.parse_args().validation_only)
