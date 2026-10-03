"""Native gradual versus shuffled learning with exactly matched source exposure."""
import argparse
from pathlib import Path
import shutil
import subprocess

from experiment_checkpoint import checkpoint, distribution
from extend_curriculum import read_schedule
from native_experiment import NativeCommands, binding_scores, read, sha, verified_manifest, write
from prepare_curriculum_order import prepare
from prepare_lessons import quoted


def run(exe, out, prepared, ancestry_root, smoke=False):
    exe, prepared, ancestry_root = exe.resolve(), prepared.resolve(), ancestry_root.resolve()
    executable_sha = sha(exe)
    seeds = [1337] if smoke else [1337, 2026, 31415]
    if not smoke:
        manifest = verified_manifest(prepared)
        previous = read(ancestry_root/'comparison.json')
        assert previous['protocol']['executable_sha256'] == executable_sha
        assert sha(ancestry_root/'ancestor.sg') == previous['protocol']['ancestor_schedule_sha256']
        _, old_stages = read_schedule(ancestry_root/'ancestor.sg')
        assert len(old_stages) == 1 and old_stages[0]['end_update'] == 6000
        assert sha(old_stages[0]['source']) == sha(prepared/'reading.dat')
        for seed in seeds:
            expected = next(row for row in previous['shared_ancestors'] if row['seed'] == seed)
            assert sha(ancestry_root/f'{seed}-ancestor/latest.ckpt') == expected['checkpoint_sha256']
        assert manifest['purpose'] == 'learning_comparison'
    out.mkdir(parents=True, exist_ok=False)
    if smoke:
        prepared = (out/'source').resolve()
        manifest = prepare(Path('data/curriculum-order-v1.json'), Path('data/prepared/binding-diversity-v1'), prepared, True)
    schedules = {arm: prepared/f'{arm}.sg' for arm in ('ramped', 'shuffled')}
    sources, boundaries, observations, stage_ends = {}, {}, {}, {}
    for arm, schedule in schedules.items():
        _, stages = read_schedule(schedule)
        sources[arm] = [dict(source=s['source'].as_posix(), sha256=sha(s['source']),
                            bytes=len(s['content']), documents=len(s['content'].split(b'\x1e')),
                            end_update=s['end_update']) for s in stages]
        boundaries[arm] = [row['documents'] for row in sources[arm]]
        stage_ends[arm] = [row['end_update'] for row in sources[arm]]
        observations[arm] = [len(doc)-1 for i in range(1,4)
                            for doc in (prepared/f'{arm}-observations-{i}.dat').read_bytes().split(b'\x1e')]
    assert stage_ends['ramped'] == stage_ends['shuffled']
    common_end = manifest['prerequisite_end']
    endpoints = ([48] if smoke else [34000])+stage_ends['ramped'][2:]
    capacity, speech, save, log = (16, 16, 32, 64) if smoke else (1024, 500, 5000, 6000)
    validation = Path('data/prepared/development-v2-final/13853.txt').resolve()
    protocol = dict(status='smoke_only' if smoke else 'declared_before_training',
                    seeds=seeds, arms=['ramped', 'shuffled'], cell='selective', channels=256, hidden=512, layers=4,
                    chunk=128, tf32=True, learning_rate=.0003, later_stage_multiplier=.25, answer_emphasis=64,
                    reading_end=manifest['reading_end'], prerequisite_end=common_end,
                    online_endpoints=endpoints, replay_policy='stage', replay_slots=capacity, replay_every=4,
                    speak_every=speech, generated_bytes_per_speech=96, graph_speech=True,
                    save_every=save, log_every=log, executable_sha256=executable_sha,
                    source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                    driver_sha256=sha(Path(__file__)), shared_io_sha256=sha(Path(__file__).with_name('native_experiment.py')),
                    policy_sha256=manifest['policy_sha256'], prepared_manifest_sha256=sha(prepared/'manifest.json'),
                    source_records=sources, schedule_sha256={arm:sha(path) for arm,path in schedules.items()},
                    selected_spec_sha256=manifest['selected_spec_sha256'],
                    validation_reader_sha256=sha(validation),
                    probe_sha256={name:sha(prepared/name) for name in
                        ('train.sgprobe','development.sgprobe','test.sgprobe','expanded-train-monitor.sgprobe')},
                    final_observation_multiset_sha256=manifest['arms']['ramped']['multiset_sha256'],
                    primary_endpoint=f'Original development joint group accuracy at {endpoints[-1]} observations.',
                    secondary_endpoints='Unconstrained exact answer generation, original/expanded training monitors, '
                                        'earlier-reader loss, byte exposure and native live timing.',
                    comparison='Both arms start from the same complete 10000-observation checkpoint (32 in smoke mode). '
                               'The final observation multiset, stage boundaries, update/slot/speech budgets match. '
                               'Lesson order and consequent stage-replay composition/history differ. '
                               'Earlier measurement points may have different source byte counts.',
                    reserved_test_evaluated=False,
                    timing='Continuation segments after the common prerequisite checkpoint; includes live learning, '
                           'replay, speech, logging and checkpoint I/O; excludes setup and assessment.',
                    limits=manifest['limits'])
    if not smoke:
        protocol['reading_ancestry_comparison_sha256'] = sha(ancestry_root/'comparison.json')
        protocol['reading_ancestry_protocol_sha256'] = sha(ancestry_root/'protocol.json')
    write(out/'protocol.json', protocol)
    native = NativeCommands(exe, out)
    ancestors, results = [], []
    for seed_index, seed in enumerate(seeds):
        if smoke:
            reading_schedule = out/'reading.sg'
            reading_schedule.write_text(f'SGCURRICULUM3\n16 {quoted((prepared/"reading.dat").as_posix())} 1 all 1\n')
            reading = out/f'{seed}-reading'
            native('live','--curriculum',reading_schedule,'--out',reading,'--updates',16,
                   '--cell','selective','--channels',256,'--hidden',512,'--layers',4,'--seed',seed,
                   '--lr',.0003,'--chunk',128,'--replay','stage','--replay-capacity',capacity,
                   '--replay-every',4,'--graph','--fast','--speak-every',speech,'--tokens',96,
                   '--prompt','The bird ','--validation',validation,'--eval-batches',32,'--log-every',log,'--save-every',save)
            source = reading/'latest.ckpt'
            reading_session = read(reading/'session.json')
        else:
            reading_schedule = ancestry_root/'ancestor.sg'
            source = ancestry_root/f'{seed}-ancestor/latest.ckpt'
            reading_session = next(row['session'] for row in previous['shared_ancestors'] if row['seed'] == seed)
        reading_sha = sha(source)
        common = out/f'{seed}-common'
        native('live','--resume',source,'--curriculum',reading_schedule,'--extend-curriculum',prepared/'common.sg',
               '--out',common,'--updates',common_end,'--prompt','The bird ','--validation',validation,
               '--eval-batches',32,'--log-every',log,'--save-every',save)
        common_checkpoint = common/'latest.ckpt'
        meta, _, arrays, common_sha = checkpoint(common_checkpoint)
        assert meta[24] == common_end and tuple(meta[1:5]) == (5,256,512,4)
        common_session = read(common/'session.json')
        ancestor = dict(seed=seed, checkpoint_sha256=common_sha, reading_checkpoint_sha256=reading_sha,
                        session=common_session, reading_session=reading_session,
                        both_arms_resume_identical_checkpoint=True)
        if not smoke:
            earlier = ancestry_root/f'{seed}-diversity/stage-2.ckpt'
            assert checkpoint(earlier)[2] == arrays, 'Common prerequisite learning changed numerically'
            ancestor.update(previous_prerequisite_arrays_identical=True,
                            previous_prerequisite_checkpoint_sha256=sha(earlier))
        ancestors.append(ancestor)
        for point_index, end in enumerate(endpoints):
            arms = ('ramped','shuffled') if (seed_index+point_index)%2 == 0 else ('shuffled','ramped')
            for arm in arms:
                destination = out/f'{seed}-{arm}'
                admission = (['--resume',common_checkpoint,'--curriculum',prepared/'common.sg','--extend-curriculum',schedules[arm]]
                             if point_index == 0 else ['--resume',destination/'latest.ckpt','--curriculum',schedules[arm]])
                native('live',*admission,'--out',destination,'--updates',end,'--prompt','The bird ',
                       '--validation',validation,'--eval-batches',32,'--log-every',log,'--save-every',save)
                session = read(destination/'session.json')
                assert session['online_updates'] == end and session['gradient_reductions'] == 'ordered_v1'
                assert session['curriculum_extended'] == (point_index == 0)
                assert session['observed_pairs'] == common_session['observed_pairs']+sum(observations[arm][:end-common_end])
                snapshot = destination/f'checkpoint-{end}.ckpt'
                shutil.copyfile(destination/'latest.ckpt',snapshot)
                meta, _, _, identity = checkpoint(snapshot)
                introduced = next(i+1 for i,boundary in enumerate(stage_ends[arm]) if end <= boundary)
                expected = [capacity//introduced+int(i<capacity%introduced) for i in range(introduced)]
                expected += [0]*(5-introduced)
                slots = distribution(snapshot,boundaries[arm])
                assert slots == expected
                row = dict(seed=seed,arm=arm,online_updates=end,parameters=meta[14],checkpoint_sha256=identity,
                           ancestor_checkpoint_sha256=common_sha,stored_windows_by_source_stage=slots,session=session)
                row.update(binding_scores(native,snapshot,prepared,destination,end))
                assert sha(snapshot) == identity
                results.append(row)
                write(out/'partial.json',results)
                print(f'{seed} {arm} {end}: reader={session["final_validation_loss"]:.5f} '
                      f'dev={row["development"]["joint_accuracy"]:.4f} '
                      f'expanded-monitor={row["expanded-train-monitor"]["joint_accuracy"]:.4f}',flush=True)
            paired = [row for row in results if row['seed']==seed and row['online_updates']==end]
            fields = ['online_updates','global_updates','replay_updates','generated_bytes','learning_rate','replay_state_bytes']
            if end == endpoints[-1]:
                fields.append('observed_pairs')
            for field in fields:
                assert paired[0]['session'][field] == paired[1]['session'][field],(seed,end,field)
        assert sha(source) == reading_sha and sha(common_checkpoint) == common_sha
    assert sha(exe) == executable_sha and sha(validation) == protocol['validation_reader_sha256']
    verified_manifest(prepared)
    write(out/'comparison.json',dict(protocol=protocol,native_commands=len(native.commands),
                                     shared_ancestors=ancestors,runs=results,prepared_manifest=manifest))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--exe',type=Path,default=Path('build/synapticgenesis.exe'))
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--prepared',type=Path,default=Path('data/prepared/curriculum-order-v1'))
    p.add_argument('--ancestry-root',type=Path,default=Path('runs/binding-diversity-panel'))
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    run(a.exe,a.out,a.prepared,a.ancestry_root,a.smoke)
