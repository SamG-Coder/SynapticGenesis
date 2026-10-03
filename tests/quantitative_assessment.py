"""Host report/ordering checks with synthetic scores; never model-quality evidence."""
import argparse
from collections import defaultdict
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import quantitative_assessment as runner
from native_experiment import read
from prose_founder import file_hash, write
from quantitative_assessment_inputs import admit, arguments, completed_predecessor
from quantitative_report import audit, prediction, suite_hash


def synthetic(rows, raw_suite, payload):
    records = []
    for index, r in enumerate(rows):
        choice = r[f"choice{r['correct']}"]
        generated = choice if index else '\xff' + choice[1:]
        scores = [4.25, 4.25]
        scores[r['correct']] = 1.25
        records.append(dict(id=r['id'], pair=r['pair'], skill=r['skill'], gold=r['correct'],
            candidate_nll=scores, context_erased_nll=[2., 3.], prediction=r['correct'],
            context_erased_prediction=0, greedy=generated))
    per_skill, groups = defaultdict(list), defaultdict(list)
    totals = [0, 0, 0]
    nll, count = 0., 0
    for r, expected in zip(records, rows):
        flags = (r['prediction']==r['gold'], r['context_erased_prediction']==r['gold'],
                 r['greedy']==expected[f"choice{r['gold']}"])
        per_skill[r['skill']].append(flags)
        groups[r['pair']].append(flags)
        totals = [a + b for a,b in zip(totals, flags)]
        nll += r['candidate_nll'][r['gold']]
        count += len(expected[f"choice{r['gold']}"])
    joint = [sum(all(flags[i] for flags in values) for values in groups.values()) for i in range(3)]
    result = dict(format='SGPROBE2', suite_hash=suite_hash(raw_suite), checkpoint_payload_hash=str(payload),
        items=128, groups=32, group_size=4, results=records, elapsed_seconds=1.25,
        strict_fp32=True, parameters_and_optimizer_unchanged=True, answer_bytes=count,
        answer_loss_nats_per_byte=nll/count, skills={})
    for i, key in enumerate(('accuracy','context_erased_accuracy','greedy_exact_accuracy')):
        result[key]=totals[i]/128
    for i, key in enumerate(('joint_accuracy','context_erased_joint_accuracy','greedy_exact_joint_accuracy')):
        result[key]=joint[i]/32
    for skill, flags in per_skill.items():
        result['skills'][skill]=dict(items=len(flags), **{key:sum(r[i] for r in flags)/len(flags)
            for i,key in enumerate(('accuracy','context_erased_accuracy','greedy_exact_accuracy'))})
    def printed(value):
        if isinstance(value, dict): return {k:printed(v) for k,v in value.items()}
        if isinstance(value, list): return [printed(v) for v in value]
        return float(format(value,'.12g')) if type(value) is float else value
    return printed(result)


def rejected(label, action, records):
    try:
        action()
    except (ValueError, RuntimeError, FileNotFoundError) as error:
        records.append(dict(case=label, reason=str(error)))
    else:
        raise AssertionError('Invalid assessment accepted: ' + label)


def report_checks(rows, raw_suite, rejections):
    assert suite_hash(b'') == '14695981039346656037'
    assert suite_hash(b'hello') == str(0xa430d84680aabd0b)
    report = synthetic(rows, raw_suite, 42)
    checked = audit(report, rows, raw_suite, 42)
    assert (checked['correct_groups'], checked['greedy_exact_groups'], checked['greedy_exact_items']) == (32,31,127)
    assert checked['outputs'][0]['greedy_hex'].startswith('ff')
    assert checked['outputs'][0]['greedy_utf8'].startswith('\ufffd')
    assert not prediction([2.,2.00000002],0) and not prediction([2.,2.000000005],-1)
    assert prediction([2.,2.00000001],0) and prediction([2.,2.00000001],-1)
    rejected('contradictory-rounded-decision', lambda: prediction([2.,2.00000002],1), rejections)
    mutations = {
        'suite-identity': lambda r:r.update(suite_hash='0'),
        'checkpoint-identity': lambda r:r.update(checkpoint_payload_hash='43'),
        'nonfinite-score': lambda r:r['results'][0].update(candidate_nll=[float('nan'),2.]),
        'negative-score': lambda r:r['results'][0].update(candidate_nll=[-1.,2.]),
        'wrong-prediction': lambda r:r['results'][0].update(prediction=1-r['results'][0]['gold']),
        'wrong-label': lambda r:r['results'][0].update(gold=1-r['results'][0]['gold']),
        'missing-item': lambda r:r['results'].pop(),
        'reordered-items': lambda r:r['results'].reverse(),
        'boolean-prediction': lambda r:r['results'][0].update(prediction=True),
        'unicode-not-raw-byte': lambda r:r['results'][0].update(greedy='\u0100'+r['results'][0]['greedy'][1:]),
        'wrong-generation-budget': lambda r:r['results'][0].update(greedy=r['results'][0]['greedy']+'x'),
        'wrong-item-aggregate': lambda r:r.update(accuracy=.5),
        'wrong-group-aggregate': lambda r:r.update(joint_accuracy=.5),
        'wrong-greedy-aggregate': lambda r:r.update(greedy_exact_accuracy=1.),
        'wrong-loss-aggregate': lambda r:r.update(answer_loss_nats_per_byte=2.),
        'wrong-answer-denominator': lambda r:r.update(answer_bytes=r['answer_bytes']+1),
        'wrong-skill-aggregate': lambda r:r['skills']['addition'].update(accuracy=.5),
        'state-mutation-flag': lambda r:r.update(parameters_and_optimizer_unchanged=False),
        'wrong-math': lambda r:r.update(strict_fp32=False),
        'invalid-time': lambda r:r.update(elapsed_seconds=-1.),
        'context-erased-drift': lambda r:r['results'][2].update(context_erased_nll=[2.1,3.]),
    }
    for name, edit in mutations.items():
        changed=deepcopy(report)
        edit(changed)
        rejected(name, lambda changed=changed:audit(changed,rows,raw_suite,42),rejections)
    return dict(synthetic_only=True, raw_non_utf8_bytes_preserved=True,
                rounded_threshold_ambiguity_preserved=True, known_fnv_vectors=True,
                item_group_skill_and_loss_aggregates_checked=True)


def predecessor_checks(temporary, rejections):
    root=temporary/'predecessor'; root.mkdir()
    plan=dict(cases=[])
    rows=[]
    for i in range(15):
        directory=root/f'case-{i}'; directory.mkdir()
        case=dict(seed=i//5,arm=dict(name=str(i%5)),directory=str(directory))
        row=dict(seed=case['seed'],arm=case['arm'],assessments=[dict(complete=True),dict(complete=True)])
        plan['cases'].append(case); rows.append(row)
        write(directory/'result.json',dict(complete=True,**row))
    write(root/'protocol.json',plan)
    identity=file_hash(root/'protocol.json')
    result=dict(complete=True,protocol_sha256=identity,protocol=plan,rows=rows,
                native_learning_commands=15,native_assessment_commands=360)
    write(root/'result.json',result)
    execution=dict(phase='complete',protocol_sha256=identity,completed_models=15)
    write(root/'execution.json',execution)
    assert completed_predecessor(root,identity)['cases']==15
    for name,edit in (
        ('unfinished-predecessor',lambda r:r.update(complete=False)),
        ('missing-predecessor-case',lambda r:r['rows'].pop()),
        ('predecessor-command-count',lambda r:r.update(native_assessment_commands=359)),
        ('predecessor-case-order',lambda r:r['rows'].reverse()),
    ):
        changed=deepcopy(result); edit(changed); write(root/'result.json',changed)
        rejected(name,lambda:completed_predecessor(root,identity),rejections)
    write(root/'result.json',result)
    write(root/'failure.json',dict(error='synthetic failure'))
    rejected('failed-predecessor',lambda:completed_predecessor(root,identity),rejections)


def handoff_checks(temporary, rows, raw_suite, rejections):
    runtime=temporary/'runtime.exe'; runtime.write_bytes(b'not executable: mock only')
    suite=temporary/'suite'; suite.mkdir()
    (suite/'development.sgprobe').write_bytes(raw_suite); write(suite/'questions.json',rows)
    identity=dict(pid=123,executable='mock-python.exe',creation_filetime=456)
    spec=dict(version='synthetic-test-only',suite=str(suite),runtime=str(runtime),cases=[],
        predecessor='mock-study',predecessor_protocol_sha256='mock',limits=['synthetic'],
        generation=dict(bytes=128,seed=42,temperature=.8,top_k=40,arithmetic='synthetic'))
    states={}
    for number,profile in enumerate(('105m','411m'),1):
        path=temporary/f'{profile}.ckpt'; path.write_bytes(b'mock checkpoint '+profile.encode())
        spec['cases'].append(dict(profile=profile,stage=4,checkpoint=str(path),checkpoint_sha256=file_hash(path)))
        states[str(path)]=dict(meta=[0]*15+[number])
    spec_path=temporary/'spec.json'; write(spec_path,spec)
    calls=[]
    class FakeNative:
        def __init__(self,exe,out): self.exe,self.out,self.commands=str(exe),out,[]
        def __call__(self,*args):
            command=[self.exe,*map(str,args)]
            self.commands.append(command); calls.append(command)
            write(self.out/'commands.json',self.commands)
            (self.out/f'command-{len(self.commands):03d}.log').write_bytes(b'mocked native output')
            output=Path(command[command.index('--output')+1])
            cp=command[command.index('--checkpoint')+1]
            if command[1]=='language-probes': write(output,synthetic(rows,raw_suite,states[cp]['meta'][15]))
            else: output.write_bytes(command[command.index('--prompt')+1].encode()+b'X'*128)
    def plan_at(label):
        out=temporary/label; out.mkdir()
        cases=[dict(c,state=states[c['checkpoint']],directory=str(out/c['profile'])) for c in spec['cases']]
        for case in cases: case['commands']=arguments(runtime,case,suite,rows,spec['generation'],Path(case['directory']))
        plan=dict(version='quantitative-assessment-plan-v1',status='declared_before_native_scoring',
            source_checkout=str(ROOT),workspace=str(Path.cwd().resolve()),specification=spec,
            native_commands_planned=18,learning_commands_planned=0,cases=cases,questions=rows,
            authenticated_inputs={},wait_for_study_driver=identity)
        write(out/'protocol.json',plan)
        return out,file_hash(out/'protocol.json')
    gate=Mock(identity=identity); gate.wait.return_value=0
    with patch.object(runner,'SPEC',spec_path), patch.object(runner,'authenticate'), \
         patch.object(runner,'read_state',side_effect=lambda p:states[str(p)]), \
         patch.object(runner,'ProcessGate',return_value=gate), \
         patch.object(runner,'completed_predecessor',return_value=dict(complete=True,synthetic=True)) as completion, \
         patch.object(runner,'NativeCommands',side_effect=FakeNative) as native:
        out,identity_hash=plan_at('success')
        result=runner.execute(out,identity_hash)
        assert result['complete'] and len(calls)==18 and native.call_count==2
        assert all(c[1] in ('language-probes','sample') for c in calls)
        assert len(result['rows'])==2 and all(len(r['samples'])==8 for r in result['rows'])
        success_count=len(calls)
        gate.wait.side_effect=RuntimeError('synthetic predecessor exited nonzero')
        out,identity_hash=plan_at('failed-exit')
        rejected('failed-process-gate',lambda:runner.execute(out,identity_hash),rejections)
        assert len(calls)==success_count and not read(out/'failure.json')['native_work_started']
        gate.wait.side_effect=None
        gate.identity=dict(identity,creation_filetime=999)
        out,identity_hash=plan_at('reused-pid')
        rejected('reused-process-identity',lambda:runner.execute(out,identity_hash),rejections)
        assert len(calls)==success_count
        gate.identity=identity
        completion.side_effect=ValueError('synthetic incomplete predecessor')
        out,identity_hash=plan_at('incomplete-study')
        rejected('incomplete-study-before-native',lambda:runner.execute(out,identity_hash),rejections)
        assert len(calls)==success_count and not read(out/'failure.json')['native_work_started']
    return dict(synthetic_only=True,mocked_native_commands=18,mocked_complete_models=2,
                failed_or_reused_predecessor_starts_zero_native_commands=True)


def run(report):
    assert not report.exists()
    prepared=Path('data/prepared/quantitative-development-v2')
    rows=read(prepared/'questions.json'); raw_suite=(prepared/'development.sgprobe').read_bytes()
    rejections=[]
    score_checks=report_checks(rows,raw_suite,rejections)
    with tempfile.TemporaryDirectory(prefix='quantitative-assessment-') as directory:
        root=Path(directory)
        predecessor_checks(root,rejections)
        control_checks=handoff_checks(root,rows,raw_suite,rejections)
    actual=admit()
    result=dict(passed=True,report_checks=score_checks,handoff_checks=control_checks,
        real_input_admission=dict(cases=[{k:c[k] for k in ('profile','checkpoint','checkpoint_sha256')}
                                        for c in actual['cases']],
            authenticated_inputs=len(actual['authenticated_inputs']),
            host_only=True),rejected_inputs=rejections,
        implementation_sha256={p.as_posix():file_hash(p) for p in [Path(__file__),
            *(ROOT/'scripts'/name for name in ('quantitative_report.py','quantitative_assessment.py',
                                             'quantitative_assessment_inputs.py'))]},
        native_commands_launched=0,model_quality_verified=False)
    write(report,result)
    print('Host assessment checks passed:',len(rejections),'rejections; 18 mocked commands; zero native commands.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--report',type=Path,required=True)
    run(parser.parse_args().report)
