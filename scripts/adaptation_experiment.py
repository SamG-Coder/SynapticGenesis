"""Run the declared native reading-adaptation pilot on temporary model copies."""
import argparse
from pathlib import Path
import shutil
import subprocess

from adaptation_sources import prepare_schedule, verified_edition
from native_experiment import NativeCommands, binding_scores, read, sha, verified_manifest, write
from narrative_experiment import assess_books


def run(args):
    proposal = read(args.design)['proposal']
    if (proposal['trajectories'] != 42 or proposal['seeds'] != [1337,2026,31415]
            or proposal['endpoints'] != [0,64,256,1024,4096]
            or proposal['learning_rates'] != [.000075,.0003]):
        raise ValueError('Expected the declared adaptation design')
    edition = Path(proposal['source_edition']['path']).resolve()
    manifest = verified_edition(edition, Path('data/sources-early-readers-v1.json'))
    if sha(edition/'manifest.json') != proposal['source_edition']['manifest_sha256']:
        raise ValueError('Declared early-reader edition changed')
    previous = read('runs/teacher-retention-panel/protocol.json')
    books, binding = Path(previous['prepared_directory']), Path(previous['binding_directory'])
    verified_manifest(binding)
    if sha(binding/'manifest.json') != previous['binding_manifest_sha256']:
        raise ValueError('Earlier binding edition changed')
    for value in previous['book_validation'].values():
        if sha(books/f"{value['id']}.txt") != value['sha256']:
            raise ValueError('Earlier book evaluation changed')
    for model in proposal['experienced_starts']:
        if sha(model['checkpoint']) != model['checkpoint_sha256']:
            raise ValueError('Declared experienced checkpoint changed')
    out, exe, main = args.out.resolve(), args.exe.resolve(), args.main_exe.resolve()
    out.mkdir(parents=True,exist_ok=False)
    inputs=out/'edition'; inputs.mkdir()
    for name in ('train.dat','validation.dat','manifest.json','source-spec.json','ATTRIBUTION.md'):
        shutil.copyfile(edition/name,inputs/name)
    seeds = [1337] if args.smoke else proposal['seeds']
    endpoints = [0,4,16] if args.smoke else proposal['endpoints']
    schedules = {}
    for seed in seeds:
        result = prepare_schedule(inputs/'train.dat',seed,endpoints[-1],inputs/f'{seed}.sg')
        schedules[str(seed)]={k:v for k,v in result.items() if k!='windows'}
        schedules[str(seed)]['window_records_sha256']=sha(inputs/f'{seed}.json')
    sources = [p.as_posix() for p in sorted(Path('src').glob('*.cu'))+sorted(Path('src').glob('*.cuh'))]
    sources += ['CMakeLists.txt','build.ps1','experiments/adaptation_probe.cu','experiments/adaptation_io.cuh',
                'scripts/adaptation_sources.py','scripts/adaptation_experiment.py',
                'scripts/native_experiment.py','scripts/narrative_experiment.py',
                'tests/adaptation_experiment.py']
    starts=['fresh','parent-carry','parent-reset','control-carry','control-reset','teacher-carry','teacher-reset']
    protocol=dict(status='smoke_only' if args.smoke else 'declared_before_training',
        design_sha256=sha(args.design), proposal=proposal, seeds=seeds, starts=starts,
        endpoints=endpoints, learning_rates=proposal['learning_rates'], trajectories=len(seeds)*14,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source_sha256={p:sha(p) for p in sources}, executable_sha256=sha(exe), main_executable_sha256=sha(main),
        schedules=schedules, source_manifest_sha256=sha(inputs/'manifest.json'),
        source_paths=dict(train=(inputs/'train.dat').as_posix(),validation=(inputs/'validation.dat').as_posix()),
        source_sha=dict(train=sha(inputs/'train.dat'),validation=sha(inputs/'validation.dat')),
        train_document_ids=[s['id'] for s in manifest['sources'] if s['split']=='train'],
        validation_document_ids=[s['id'] for s in manifest['sources'] if s['split']=='validation'],
        book_directory=books.as_posix(),book_identities=previous['book_validation'],
        binding_directory=binding.as_posix(),binding_manifest_sha256=sha(binding/'manifest.json'),
        development_probes_sha256=sha(binding/'development.sgprobe'),
        book_evaluation_batches=2 if args.smoke else 32,
        input_math='strict FP32, unweighted bytes, reset recurrence for every training/evaluation window',
        evaluation_windows='Sequential 128-target windows starting at byte 0 in each document, '
                           'with the exact final shorter tail. Every in-document next-byte pair scored once.',
        execution_order='Rotate seven starts by seed index; alternate rate order by seed plus rotated start index.',
        timing='Native synchronized update timing excludes evaluation, persistence and process startup; '
               'other desktop GPU contexts are not isolated. No intrinsic speed comparison.',
        original_checkpoints_mutated=False,reserved_tests_scored=False,generated_text_targets=False)
    write(out/'protocol.json',protocol)
    for name in ('probe-commands','assessment-commands'):
        (out/name).mkdir()
    native, assess = NativeCommands(exe,out/'probe-commands'),NativeCommands(main,out/'assessment-commands')
    records=[]
    for seed_index,seed in enumerate(seeds):
        order=starts[seed_index:]+starts[:seed_index]
        for index,start in enumerate(order):
            rates=protocol['learning_rates'] if (index+seed_index)%2==0 else protocol['learning_rates'][::-1]
            for lr in rates:
                rate_name='slow' if lr==.000075 else 'base'
                label=f'{seed}-{start}-{rate_name}'
                directory=out/label
                mode='fresh' if start=='fresh' else start.split('-')[1]
                origin=None if start=='fresh' else next(m for m in proposal['experienced_starts']
                    if m['seed']==seed and m['role']==start.split('-')[0])
                options=[] if origin is None else ['--checkpoint',origin['checkpoint']]
                native('run','--train',inputs/'train.dat','--validation',inputs/'validation.dat',
                       '--schedule',inputs/f'{seed}.sg','--out',directory,'--mode',mode,'--seed',seed,
                       '--lr',lr,'--steps',endpoints[-1],'--endpoints',','.join(map(str,endpoints)),*options)
                results=read(directory/'result.json')
                evaluations=[]
                for endpoint in endpoints:
                    checkpoint=directory/f'checkpoint-{endpoint}.ckpt'
                    value=read(directory/f'evaluation-{endpoint}.json')
                    value['checkpoint_sha256']=sha(checkpoint)
                    value['checkpoint_sidecar_sha256']=sha(Path(str(checkpoint)+'.sgadapt'))
                    if endpoint in (0,endpoints[-1]):
                        value['earlier_books']=assess_books(assess,checkpoint,books,directory,endpoint,
                                                           protocol['book_evaluation_batches'])
                        value.update(binding_scores(assess,checkpoint,binding,directory,endpoint,
                                                   splits=('development',)))
                    evaluations.append(value)
                record=dict(label=label,seed=seed,start=start,learning_rate=lr,origin=origin,
                            native_result=results,evaluations=evaluations,
                            update_journal_sha256=sha(directory/'updates.jsonl'))
                records.append(record)
                write(out/'partial.json',records)
                print(label,'complete;',len(records),'of',protocol['trajectories'],'trajectories',flush=True)
    if sha(exe)!=protocol['executable_sha256'] or sha(main)!=protocol['main_executable_sha256']:
        raise ValueError('Study executable changed')
    if any(sha(p)!=value for p,value in protocol['source_sha256'].items()):
        raise ValueError('Study implementation changed during execution')
    if any(sha(m['checkpoint'])!=m['checkpoint_sha256'] for m in proposal['experienced_starts']):
        raise ValueError('An original model changed during diagnostic execution')
    verified_edition(edition,Path('data/sources-early-readers-v1.json'))
    write(out/'comparison.json',dict(protocol=protocol,runs=records,
        native_training_commands=len(native.commands),native_assessment_commands=len(assess.commands),
        originals_unchanged=True,all_declared_trajectories_completed=True))
    print(out/'comparison.json',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--design',type=Path,default=Path('reports/plasticity-experiment-design.json'))
    parser.add_argument('--exe',type=Path,default=Path('build/adaptation-probe/synaptic-adaptation-probe.exe'))
    parser.add_argument('--main-exe',type=Path,default=Path('build/synapticgenesis.exe'))
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--smoke',action='store_true')
    run(parser.parse_args())
