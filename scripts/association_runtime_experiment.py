"""Compare the optimized runtime against the preserved original executable.

All repetitions resume identical learned history. This is a numerical/cost
experiment; repeated runs do not count as independent learning seeds.
"""
import argparse
from pathlib import Path
import statistics
import subprocess

from native_experiment import NativeCommands, read, sha, write


def run(old, candidate, out, ancestry):
    old, candidate, ancestry = old.resolve(), candidate.resolve(), ancestry.resolve()
    recorded = read(ancestry/'comparison.json')
    assert sha(old) == recorded['protocol']['executable_sha256']
    schedule = ancestry/'curriculum.sg'
    assert sha(schedule) == recorded['protocol']['schedule_sha256']
    for record in recorded['protocol']['source_records']:
        assert sha(record['source']) == record['sha256']
    sources = {seed: ancestry/f'{seed}-associative/checkpoint-10000.ckpt' for seed in (1337,2026,31415)}
    for seed, path in sources.items():
        row = next(r for r in recorded['runs'] if r['seed']==seed and r['arm']=='associative')
        assert sha(path) == row['prerequisite_checkpoint_sha256']
        assert sha(ancestry/f'{seed}-associative/checkpoint-34000.ckpt') == row['checkpoint_sha256']
    out.mkdir(parents=True,exist_ok=False)
    exes = {'original':old,'optimized':candidate}
    calls = {}
    for name, exe in exes.items():
        (out/name).mkdir()
        calls[name] = NativeCommands(exe,out/name)
    validation = Path('data/prepared/development-v2-final/13853.txt').resolve()
    assert sha(validation) == recorded['protocol']['validation_reader_sha256']
    protocol = dict(status='declared_before_measurement',
        original_executable_sha256=sha(old),optimized_executable_sha256=sha(candidate),
        source_parent_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        changed_source_sha256={p:sha(p) for p in ['src/associative_memory.cuh','src/spike_lm.cu','src/associative_bench.cuh']},
        driver_sha256=sha(Path(__file__)),ancestry_report_sha256=sha(ancestry/'comparison.json'),
        checkpoint_sha256={str(seed):sha(path) for seed,path in sources.items()},
        schedule_sha256=sha(schedule),validation_reader_sha256=sha(validation),
        live_repeat_seed=1337,live_repeats=7,start_online_update=10000,end_online_update=12000,
        full_reproduction_seeds=[1337,2026,31415],full_reproduction_end=34000,
        decode_repeats=3,decode_rounds_per_repeat=7,decode_bytes_per_round=512,
        timing='Unprofiled sequential processes. Live elapsed time includes learning, replay, speech, logging '
               'and checkpoint writes, excluding setup and validation. Decode excludes loading and graph capture.',
        hypothesis='Same arithmetic, state and trajectory with less recurrence-kernel memory/coordination cost.',
        learning_quality_claim=False,reserved_test_evaluated=False)
    write(out/'protocol.json',protocol)
    rows=[]
    expected=None
    for repeat in range(7):
        order = ('original','optimized') if repeat%2==0 else ('optimized','original')
        for name in order:
            destination=out/name/f'live-{repeat}'
            calls[name]('live','--resume',sources[1337],'--curriculum',schedule,'--out',destination,
                '--updates',12000,'--prompt','The bird ','--validation',validation,
                '--eval-batches',32,'--log-every',1000,'--save-every',5000)
            payload=(destination/'latest.ckpt').read_bytes()
            if expected is None:
                expected=payload
            assert payload==expected,(repeat,name,'Learning history changed')
            session=read(destination/'session.json')
            rows.append(dict(repeat=repeat,variant=name,session=session,checkpoint_sha256=sha(destination/'latest.ckpt')))
            write(out/'live-partial.json',rows)
            print(f'{name} live repeat {repeat}: {session["elapsed_seconds"]:.4f}s, complete checkpoint identical',flush=True)
    full=[]
    for seed in protocol['full_reproduction_seeds']:
        destination=out/'optimized'/f'full-{seed}'
        calls['optimized']('live','--resume',sources[seed],'--curriculum',schedule,'--out',destination,
            '--updates',34000,'--prompt','The bird ','--validation',validation,
            '--eval-batches',32,'--log-every',6000,'--save-every',5000)
        previous=ancestry/f'{seed}-associative/checkpoint-34000.ckpt'
        assert (destination/'latest.ckpt').read_bytes()==previous.read_bytes(),seed
        full.append(dict(seed=seed,checkpoint_sha256=sha(previous),complete_checkpoint_identical=True,
                         session=read(destination/'session.json')))
        write(out/'full-partial.json',full)
        print(f'Seed {seed}: all 24000 later observations reproduced the original checkpoint exactly',flush=True)
    decode=[]
    checkpoint=ancestry/'1337-associative/checkpoint-34000.ckpt'
    expected_sample=None
    for repeat in range(3):
        for name in (('original','optimized') if repeat%2==0 else ('optimized','original')):
            destination=out/name/f'decode-{repeat}'
            calls[name]('decode-bench','--checkpoint',checkpoint,'--out',destination,'--tokens',512,'--rounds',7)
            sample=(destination/'sample.txt').read_bytes()
            if expected_sample is None:
                expected_sample=sample
            assert sample==expected_sample
            result=read(destination/'benchmark.json')
            assert result['generated_bytes_identical'] and result['state_logits_max_error']==0
            decode.append(dict(repeat=repeat,variant=name,result=result))
    medians={name:dict(live_seconds=statistics.median(r['session']['elapsed_seconds'] for r in rows if r['variant']==name),
                      graph_us_per_byte=statistics.median(r['result']['graph_us_per_byte'] for r in decode if r['variant']==name),
                      regular_us_per_byte=statistics.median(r['result']['regular_us_per_byte'] for r in decode if r['variant']==name))
             for name in exes}
    for name,path in exes.items():
        assert sha(path)==protocol[name+'_executable_sha256']
    for seed,path in sources.items():
        assert sha(path)==protocol['checkpoint_sha256'][str(seed)]
    result=dict(passed=True,protocol=protocol,native_commands=sum(len(n.commands) for n in calls.values()),
                live=rows,full_reproductions=full,decode=decode,medians=medians,
                complete_live_checkpoints_identical=True,generated_samples_identical=True,
                speedups={key:medians['original'][key]/medians['optimized'][key] for key in medians['original']})
    write(out/'comparison.json',result)
    print(medians,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--original',type=Path,default=Path('runs/legacy-associative-41a/synapticgenesis.exe'))
    p.add_argument('--candidate',type=Path,default=Path('build/synapticgenesis.exe'))
    p.add_argument('--ancestry',type=Path,default=Path('runs/associative-screen-panel'))
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    run(a.original,a.candidate,a.out,a.ancestry)
