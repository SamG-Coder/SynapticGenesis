"""Native adaptation admission, source exposure and exact restart checks."""
import argparse
import json
from pathlib import Path
import struct
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from adaptation_sources import fnv, prepare_schedule
from native_experiment import read, sha, write
from probes_cli import Reference
import torch


def payload(path):
    data = path.read_bytes()
    meta = struct.unpack_from('<32Q', data)
    count = meta[14] * 4
    return meta, [data[288 + i*count:288 + (i+1)*count] for i in range(3)]


def run(exe, out):
    exe, out = exe.resolve(), out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    train, validation, schedule = out / 'train.dat', out / 'validation.dat', out / 'schedule.sg'
    docs = [b'A child reads a book beneath a tree. ' * 7, b'The rain fills the stream. ' * 8]
    train.write_bytes(b'\x1e'.join(docs))
    validation.write_bytes(b'The bird sings beside a window. ' * 5 + b'\x1e' + b'Water.\n')
    selected = prepare_schedule(train, 1337, 16, schedule)
    original_inputs = {str(p): sha(p) for p in (train, validation, schedule)}
    calls, rejected = [], []

    def call(name, extra=(), changes=None, success=True):
        options = dict(train=train, validation=validation, schedule=schedule, out=out/name,
                       mode='fresh', seed=1337, lr=.0003, steps=8, endpoints='0,3,8',
                       channels=16, hidden=24, layers=2, cell='associative')
        options.update(changes or {})
        command = [str(exe), 'run']
        for key, value in options.items():
            if value is not None:
                command += ['--' + key, str(value)]
        command += list(map(str, extra))
        calls.append(command)
        write(out / 'commands.json', calls)
        result = subprocess.run(command, capture_output=True)
        (out / (name + '.log')).write_bytes(result.stdout + result.stderr)
        if success:
            assert result.returncode == 0, result.stderr.decode(errors='replace')
            return read(out / name / 'result.json')
        assert result.returncode != 0, 'Invalid probe invocation was accepted: ' + name
        assert not (out / name).exists(), 'Rejected probe created output: ' + name
        rejected.append(name)

    full = call('full')
    call('fewer-evaluations',changes=dict(endpoints='0,8'))
    assert sha(out/'full/checkpoint-8.ckpt') == sha(out/'fewer-evaluations/checkpoint-8.ckpt')
    call('split', changes=dict(steps=3))
    call('resumed', changes=dict(resume=out/'split/checkpoint-3.ckpt'))
    for suffix in ('.ckpt', '.ckpt.sgadapt'):
        assert (out/('full/checkpoint-8'+suffix)).read_bytes() == (out/('resumed/checkpoint-8'+suffix)).read_bytes()
    for step in (3, 8):
        assert read(out/f'full/evaluation-{step}.json') == read(out/f'resumed/evaluation-{step}.json')
    rows = [json.loads(line) for line in (out/'full/updates.jsonl').read_text().splitlines()]
    assert [(r['update'],r['document'],r['offset']) for r in rows] == [
        (r['update'],r['document'],r['offset']) for r in selected['windows'][:8]]
    assert full['session_presented_pairs'] == 1024
    torch.set_num_threads(1)
    reference=Reference(out/'full/checkpoint-0.ckpt')
    first=selected['windows'][0]
    text=docs[first['document']][first['offset']:first['offset']+129]
    logp=reference.logits(text[:-1]).log_softmax(-1)
    independent_loss=-sum(float(logp[i,byte]) for i,byte in enumerate(text[1:]))/128
    independent_error=abs(independent_loss-rows[0]['loss_before_update'])
    assert independent_error < 3e-5, independent_error
    for step in (0, 3, 8):
        score = read(out/f'full/evaluation-{step}.json')
        assert score['train']['pairs'] == sum(len(d)-1 for d in docs)
        assert score['validation']['pairs'] == sum(len(d)-1 for d in validation.read_bytes().split(b'\x1e'))
        assert score['weights_optimizer_unchanged']
    ancestor = out/'full/checkpoint-8.ckpt'
    ancestor_hash = sha(ancestor)
    common = dict(checkpoint=ancestor, channels=None, hidden=None, layers=None, cell=None, steps=0)
    carry = call('carry-zero', changes=dict(common, mode='carry'))
    reset = call('reset-zero', changes=dict(common, mode='reset'))
    old_meta, old_values = payload(ancestor)
    carry_meta, carry_values = payload(out/'carry-zero/checkpoint-0.ckpt')
    reset_meta, reset_values = payload(out/'reset-zero/checkpoint-0.ckpt')
    assert old_values == carry_values and carry_meta[7] == old_meta[7] == 8
    assert reset_values[0] == old_values[0] and reset_meta[7] == 0
    assert all(not any(v) for v in reset_values[1:]) and any(old_values[1]) and any(old_values[2])
    assert carry['initial_weights_hash'] == reset['initial_weights_hash'] == str(fnv(old_values[0]))
    for mode in ('carry','reset'):
        call(mode+'-full', changes=dict(common,mode=mode,steps=8))
        call(mode+'-split', changes=dict(common,mode=mode,steps=3))
        call(mode+'-resume', changes=dict(common,mode=mode,steps=8,resume=out/(mode+'-split/checkpoint-3.ckpt')))
        assert sha(out/(mode+'-full/checkpoint-8.ckpt')) == sha(out/(mode+'-resume/checkpoint-8.ckpt'))
    assert sha(ancestor) == ancestor_hash
    for name, change in [('missing-ancestor',dict(mode='carry')),
                         ('fresh-ancestor',dict(checkpoint=ancestor)),
                         ('bad-mode',dict(mode='invalid')),('bad-rate',dict(lr='nan')),
                         ('bad-budget',dict(steps=17)),('bad-endpoints',dict(endpoints='0,8,3'))]:
        call(name, changes=change, success=False)
    call('bad-flag', extra=['--fast'],success=False)
    lines = schedule.read_text().splitlines()
    for name, edit in [('bad-document','20 0 0'),('cross-document','0 99999 0'),('bad-window','0 0 0')]:
        path=out/(name+'.sg'); path.write_text('\n'.join([lines[0],edit,*lines[2:]])+'\n')
        call(name,changes=dict(schedule=path),success=False)
    restart=out/'split/checkpoint-3.ckpt'
    for name,change in [('resume-rate',dict(lr=.000075)),('resume-seed',dict(seed=2026)),
                        ('resume-backward',dict(steps=2)),('resume-mode',dict(common,mode='reset'))]:
        call(name,changes=dict(change,resume=restart),success=False)
    changed_validation=out/'changed-validation.dat'; changed_validation.write_bytes(b'A different validation story.')
    call('resume-validation',changes=dict(resume=restart,validation=changed_validation),success=False)
    for name,index in [('corrupt-payload',300),('corrupt-header',7*8)]:
        path=out/(name+'.ckpt'); raw=bytearray(restart.read_bytes()); raw[index]^=1; path.write_bytes(raw)
        Path(str(path)+'.sgadapt').write_bytes(Path(str(restart)+'.sgadapt').read_bytes())
        call(name,changes=dict(resume=path),success=False)
    path=out/'corrupt-sidecar.ckpt'; path.write_bytes(restart.read_bytes())
    Path(str(path)+'.sgadapt').write_bytes(b'bad')
    call('corrupt-sidecar',changes=dict(resume=path),success=False)
    assert all(sha(p)==expected for p,expected in original_inputs.items())
    summary=dict(passed=True, executable_sha256=sha(exe), native_commands=len(calls),
                 exact_restart_modes=['fresh','carry','reset'], exact_carry_and_reset_payloads=True,
                 all_source_exposures_match=True, original_checkpoint_unchanged=sha(ancestor)==ancestor_hash,
                 evaluation_parameter_state_unchanged=True, negative_controls=rejected,
                 evaluation_frequency_preserves_checkpoint=True,
                 independent_first_window_loss_error=independent_error,
                 source_sha256={p:sha(p) for p in ['experiments/adaptation_probe.cu','experiments/adaptation_io.cuh',
                     'scripts/adaptation_sources.py','tests/adaptation_probe.py']},
                 scope='Tiny native fixtures; not adaptation quality or independent model-math parity.')
    write(out/'result.json',summary)
    print(len(calls),'native actions pass; exact fresh/carry/reset restart and',len(rejected),'rejections.')


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--exe',type=Path,default=Path('build/adaptation-probe/synaptic-adaptation-probe.exe'))
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    run(args.exe,args.out)
