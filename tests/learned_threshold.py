"""Localize a learned CPU/native discrepancy using read-only native trace dumps.

The single-spike intervention is a diagnostic, not an independent oracle pass.
The original score-tolerance failure remains a failure in the study report.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch

from binding_learned_oracle import development_rows, sha
from probes_cli import Reference
from native_experiment import NativeCommands
from prepare_lessons import write_probes


def prepare(root,study):
    data=json.loads((study/'comparison.json').read_text())
    assert sha(Path('build/synapticgenesis.exe')) == data['protocol']['executable_sha256']
    rows=development_rows(data['protocol'])[:4]
    root.mkdir(parents=True,exist_ok=False)
    write_probes(root/'group.sgprobe',rows,'SGPROBE2')
    checkpoint=study/'31415-ramped/checkpoint-130000.ckpt'
    native=NativeCommands(Path('build/synapticgenesis.exe'),root)
    native('language-probes','--checkpoint',checkpoint,'--probes',root/'group.sgprobe',
           '--output',root/'native-repeat.json')
    # Keep every native call in one journal while using the separately built
    # trace executable for the remaining four read-only diagnostic commands.
    native.exe=Path('build/trace-diagnostic/synaptic-trace-diagnostic.exe').resolve()
    for row,item in enumerate(rows[:2]):
        for choice in range(2):
            source=root/f'row-{row}-choice-{choice}.txt'
            source.write_bytes((item['context']+item['query']+item[f'choice{choice}']).encode('ascii'))
            native(checkpoint,source,root/f'row-{row}-choice-{choice}')


def check(root, study):
    checkpoint=study/'31415-ramped/checkpoint-130000.ckpt'
    identity=sha(checkpoint)
    original=json.loads((study/'31415-ramped/development-130000.json').read_text())['results'][:4]
    repeated=json.loads((root/'native-repeat.json').read_text())['results']
    assert original == repeated
    ref=Reference(checkpoint)
    records=[]
    for row in range(2):
        for choice in range(2):
            directory=root/f'row-{row}-choice-{choice}'
            raw=(root/f'row-{row}-choice-{choice}.txt').read_bytes()
            traces=[]
            logits=ref.logits(raw[:-1],traces=traces)
            native=[]
            first=None
            for layer,trace in enumerate(traces):
                arrays={name:np.fromfile(directory/f'layer-{layer}-{name}.f32',dtype='<f4').reshape(len(raw)-1,ref.h)
                        for name in ('u','spikes')}
                assert np.array_equal(arrays['spikes'],(arrays['u']>=1).astype('float32')-(arrays['u']<=-1).astype('float32'))
                native.append(arrays)
                differences=np.argwhere(trace['spikes'].numpy()!=arrays['spikes'])
                if len(differences) and first is None:
                    t,h=map(int,differences[0])
                    first=dict(layer_zero_based=layer,byte_zero_based=t,neuron_zero_based=h,
                        cpu_membrane=float(trace['u'][t,h]),native_membrane=float(arrays['u'][t,h]),
                        cpu_spike=float(trace['spikes'][t,h]),native_spike=float(arrays['spikes'][t,h]))
            assert first is not None
            layer,t,h=(first[k] for k in ('layer_zero_based','byte_zero_based','neuron_zero_based'))
            assert abs(first['cpu_membrane']-first['native_membrane']) < 1e-5
            assert max(abs(abs(first[k])-1) for k in ('cpu_membrane','native_membrane')) < 1e-5
            override=[torch.full((len(raw)-1,ref.h),float('nan')) for _ in ref.blocks]
            override[layer][t,h]=first['native_spike']
            controlled_traces=[]
            controlled=ref.logits(raw[:-1],traces=controlled_traces,forced_spikes=override)
            assert all(np.array_equal(trace['spikes'].numpy(),n['spikes']) for trace,n in zip(controlled_traces,native))
            targets=torch.tensor(list(raw[1:]),dtype=torch.long)
            def answer_nll(values):
                return -values.log_softmax(-1).gather(1,targets[:,None]).flatten()[-4:].double().sum().item()
            dumped_nll=float(np.fromfile(directory/'losses.f32',dtype='<f4')[-4:].astype('float64').sum())
            assert abs(dumped_nll-original[row]['candidate_nll'][choice]) < 1e-9
            controlled_error=abs(answer_nll(controlled)-dumped_nll)
            assert controlled_error < 3e-5
            native_logits=torch.from_numpy(np.fromfile(directory/'logits.f32',dtype='<f4').reshape(len(raw)-1,256))
            records.append(dict(row=row,choice=choice,first_spike_divergence=first,
                native_nll=dumped_nll,cpu_nll=answer_nll(logits),single_spike_override_nll=answer_nll(controlled),
                single_spike_override_nll_error=controlled_error,
                single_spike_override_max_logit_error=float((controlled-native_logits).abs().max()),
                single_spike_override_matches_all_native_spikes=True))
    assert sha(checkpoint)==identity
    result=dict(localization_passed=True,checkpoint_sha256=identity,native_repeat_results_identical=True,
        study_executable_sha256=sha(Path('build/synapticgenesis.exe')),
        diagnostic_executable_sha256=sha(Path('build/trace-diagnostic/synaptic-trace-diagnostic.exe')),
        diagnostic_source_sha256=sha(Path(__file__).with_name('learned_trace_dump.cu')),
        reference_sha256=sha(Path(__file__).with_name('probes_cli.py')),records=records,
        original_strict_cpu_score_check_passed=False,
        interpretation='A near-threshold FP32 difference changes one spike. Forcing only that decision makes '
                       'all later spike decisions agree and restores close scores. This explains the local '
                       'difference; the intervention is not an independent passing CPU oracle.',
        checkpoint_file_unchanged=True)
    (root/'result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path('runs/curriculum-order-threshold-check'))
    p.add_argument('--study',type=Path,default=Path('runs/curriculum-order-panel'))
    p.add_argument('--prepare',action='store_true',help='Create fresh inputs and run the native repeat and trace dumps first')
    a=p.parse_args()
    if a.prepare:
        prepare(a.root,a.study)
    check(a.root,a.study)
