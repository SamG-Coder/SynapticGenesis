"""Audit recorded adaptation trajectories without rerunning model computation."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import struct
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from adaptation_sources import fnv
from native_experiment import read, sha, write
from audit_binding import audit


def model_state(path):
    raw=path.read_bytes()
    meta=struct.unpack_from('<32Q',raw)
    n=4*meta[14]
    assert len(raw)==288+3*n and meta[17]==0
    return meta,[raw[288+i*n:288+(i+1)*n] for i in range(3)]


def check(root):
    data=read(root/'comparison.json')
    p=data['protocol']
    assert p==read(root/'protocol.json') and data['all_declared_trajectories_completed']
    assert p['status'] in ('smoke_only','declared_before_training')
    expected=14*len(p['seeds'])
    assert len(data['runs'])==expected==p['trajectories']
    assert len({(r['seed'],r['start'],r['learning_rate']) for r in data['runs']})==expected
    assert not p['reserved_tests_scored'] and not p['generated_text_targets']
    assert all(sha(path)==digest for path,digest in p['source_sha256'].items())
    assert sha(root/'edition/manifest.json')==p['source_manifest_sha256']
    docs={split:(root/'edition'/f'{split}.dat').read_bytes().split(b'\x1e') for split in ('train','validation')}
    for split in docs:
        assert sha(root/'edition'/f'{split}.dat')==p['source_sha'][split]
    schedule_records={}
    for seed in p['seeds']:
        schedule=root/'edition'/f'{seed}.sg'
        record=read(schedule.with_suffix('.json'))
        policy=p['schedules'][str(seed)]
        assert sha(schedule)==policy['schedule_sha256']==record['schedule_sha256']
        assert sha(schedule.with_suffix('.json'))==policy['window_records_sha256']
        lines=schedule.read_text().splitlines()
        assert lines[0]==f"SGADAPT1 128 {p['endpoints'][-1]} {fnv((root/'edition/train.dat').read_bytes())}"
        assert len(lines)-1==len(record['windows'])==p['endpoints'][-1]
        for index,(line,window) in enumerate(zip(lines[1:],record['windows'])):
            document,offset,digest=map(int,line.split())
            assert (index+1,document,offset)==(window['update'],window['document'],window['offset'])
            raw=docs['train'][document][offset:offset+129]
            assert len(raw)==129 and b'\x1e' not in raw and fnv(raw)==digest
            import hashlib
            assert hashlib.sha256(raw).hexdigest()==window['sha256']
        schedule_records[seed]=record['windows']
    training=read(root/'probe-commands/commands.json')
    assessment=read(root/'assessment-commands/commands.json')
    assert len(training)==data['native_training_commands']==expected
    assert len(assessment)==data['native_assessment_commands']==expected*8
    assert Counter(c[1] for c in assessment)=={'evaluate':expected*6,'language-probes':expected*2}
    assert all(sha(c[0])==p['executable_sha256'] for c in training)
    assert all(sha(c[0])==p['main_executable_sha256'] for c in assessment)
    for command in training:
        assert command[1]=='run'
        assert Path(command[command.index('--train')+1]).resolve()==(root/'edition/train.dat').resolve()
        assert Path(command[command.index('--validation')+1]).resolve()==(root/'edition/validation.dat').resolve()
    for command in assessment:
        if '--data' in command:
            path=Path(command[command.index('--data')+1])
            assert path.name in ('13853.txt','12228.txt','11757.txt')
            identity=next(v['sha256'] for v in p['book_identities'].values() if str(v['id'])+'.txt'==path.name)
            assert sha(path)==identity
        else:
            assert sha(command[command.index('--probes')+1])==p['development_probes_sha256']
    for row in data['runs']:
        directory=root/row['label']
        result=read(directory/'result.json')
        assert result==row['native_result'] and result['completed'] and result['inputs_unchanged']
        assert result['cell']==6 and (result['channels'],result['hidden'],result['layers'])==(256,512,4)
        assert result['start_updates']==0 and result['end_updates']==p['endpoints'][-1]
        assert result['session_presented_pairs']==128*p['endpoints'][-1]
        assert [e['updates'] for e in row['evaluations']]==p['endpoints']
        initial_meta,initial=model_state(directory/'checkpoint-0.ckpt')
        mode=result['mode']
        if mode!='fresh':
            origin=Path(row['origin']['checkpoint'])
            assert sha(origin)==row['origin']['checkpoint_sha256']
            raw=origin.read_bytes(); meta=struct.unpack_from('<32Q',raw); n=4*meta[14]
            old=[raw[288+i*n:288+(i+1)*n] for i in range(3)]
            assert initial[0]==old[0]
            if mode=='carry':
                assert initial==old and initial_meta[7]==meta[7]
        if mode in ('fresh','reset'):
            assert initial_meta[7]==0 and all(not any(v) for v in initial[1:])
        assert result['initial_global_updates']==initial_meta[7]
        assert result['initial_weights_hash']==str(fnv(initial[0]))
        assert result['initial_moments_hash']==str(fnv(initial[2],fnv(initial[1])))
        for evaluation in row['evaluations']:
            step=evaluation['updates']; file=directory/f'checkpoint-{step}.ckpt'
            assert sha(file)==evaluation['checkpoint_sha256']
            meta,arrays=model_state(file)
            assert meta[7]==initial_meta[7]+step and meta[16]==0
            sidecar=Path(str(file)+'.sgadapt')
            assert sha(sidecar)==evaluation['checkpoint_sidecar_sha256']
            raw=sidecar.read_bytes(); identity=struct.unpack('<16Q',raw)
            assert identity[2]==step and identity[7]==initial_meta[7]
            assert identity[11]==fnv(file.read_bytes()) and identity[15]==fnv(raw[:120])
            actual=read(directory/f'evaluation-{step}.json')
            assert all(evaluation[k]==v for k,v in actual.items())
            assert actual['weights_optimizer_unchanged'] and actual['presented_pairs']==step*128
            for split in docs:
                values=actual[split]
                assert values['pairs']==sum(len(d)-1 for d in docs[split])
                assert len(values['documents'])==len(docs[split])
                weighted=0
                for index,(score,document) in enumerate(zip(values['documents'],docs[split])):
                    assert score['index']==index and score['pairs']==len(document)-1
                    assert math.isfinite(score['loss']) and score['loss']>=0
                    weighted+=score['loss']*score['pairs']
                assert abs(weighted/values['pairs']-values['loss'])<1e-12
            if step in (0,p['endpoints'][-1]):
                for name,book in evaluation['earlier_books'].items():
                    assert book==read(directory/f'{name}-{step}.json')
                    assert book['evaluated_bytes']==16*128*p['book_evaluation_batches']
                binding=directory/f'development-{step}.json'
                assert evaluation['development']=={k:v for k,v in read(binding).items() if k!='results'}
                assert evaluation['development_audit']==audit(binding,'development')
        assert sha(directory/'updates.jsonl')==row['update_journal_sha256']
        updates=[json.loads(line) for line in (directory/'updates.jsonl').read_text().splitlines()]
        assert len(updates)==p['endpoints'][-1]
        for update,window in zip(updates,schedule_records[row['seed']]):
            assert all(update[k]==window[k] for k in ('update','document','offset'))
            assert all(math.isfinite(update[k]) and update[k]>=0 for k in ('loss_before_update','gradient_norm','seconds'))
        assert abs(sum(u['seconds'] for u in updates)-result['update_seconds'])<1e-8
    for original in p['proposal']['experienced_starts']:
        assert sha(original['checkpoint'])==original['checkpoint_sha256']
    summary=dict(passed=True,status=p['status'],trajectories=expected,
        training_commands=len(training),assessment_commands=len(assessment),
        presented_pairs_per_trajectory=p['endpoints'][-1]*128,
        checkpoints=expected*len(p['endpoints']), original_checkpoints_unchanged=True,
        exact_window_exposure_verified=True, exact_initial_parameter_state_verified=True,
        all_evaluation_aggregates_verified=True, reserved_tests_scored=False,
        comparison_sha256=sha(root/'comparison.json'),audit_source_sha256=sha(__file__),
        scope='Execution/state audit; no independent CPU model-parity or general-language claim.')
    write(root/'execution-check.json',summary)
    print('Adaptation execution audit passed:',expected,'trajectories,',summary['checkpoints'],'checkpoints.')
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--study',type=Path,required=True)
    check(parser.parse_args().study.resolve())
