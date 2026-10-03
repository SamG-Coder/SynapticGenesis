"""Localize a learned CPU/native discrepancy using read-only native trace dumps.

The single-spike intervention is a diagnostic, not an independent oracle pass.
The original score-tolerance failure remains a failure in the study report.
"""
import argparse
import json
from pathlib import Path
import torch

from binding_learned_oracle import development_rows, sha
from probes_cli import Reference
from native_experiment import NativeCommands
from prepare_lessons import write_probes
from trace_comparison import compare


def prepare(root,study,exe):
    data=json.loads((study/'comparison.json').read_text())
    assert sha(exe) == data['protocol']['executable_sha256']
    rows=development_rows(data['protocol'])[:4]
    root.mkdir(parents=True,exist_ok=False)
    write_probes(root/'group.sgprobe',rows,'SGPROBE2')
    checkpoint=study/'31415-ramped/checkpoint-130000.ckpt'
    native=NativeCommands(exe,root)
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


def check(root, study, exe):
    data=json.loads((study/'comparison.json').read_text())
    assert sha(exe) == data['protocol']['executable_sha256']
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
            with torch.inference_mode():
                record=compare(ref,raw,directory)
            assert record.get('first_difference_within_1e_minus_5_of_threshold')
            assert record['single_spike_override_matches_all_native_spikes']
            assert abs(record['native_nll']-original[row]['candidate_nll'][choice]) < 1e-9
            assert record['single_spike_override_nll_error'] < 3e-5
            records.append(dict(row=row,choice=choice,**record))
    assert sha(checkpoint)==identity
    result=dict(localization_passed=True,checkpoint_sha256=identity,native_repeat_results_identical=True,
        study_executable_sha256=sha(exe),
        diagnostic_executable_sha256=sha(Path('build/trace-diagnostic/synaptic-trace-diagnostic.exe')),
        diagnostic_source_sha256=sha(Path(__file__).with_name('learned_trace_dump.cu')),
        comparison_source_sha256=sha(Path(__file__).with_name('trace_comparison.py')),
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
    p.add_argument('--exe',type=Path,default=Path('build/synapticgenesis.exe'),
                   help='Preserved executable matching the study protocol')
    p.add_argument('--prepare',action='store_true',help='Create fresh inputs and run the native repeat and trace dumps first')
    a=p.parse_args()
    if a.prepare:
        prepare(a.root,a.study,a.exe)
    check(a.root,a.study,a.exe)
