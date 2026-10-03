"""Paired selected-lesson expansion, with a shared learned ancestor per seed.

Only the native C++/CUDA executable learns or generates. This driver fixes the
protocol, extends each ancestor's source schedule and collects held-out evidence.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

from experiment_checkpoint import checkpoint, distribution
from extend_curriculum import read_schedule
from native_experiment import NativeCommands, binding_scores, sha, write
from prepare_lessons import quoted


def run(exe, out, smoke=False):
    exe = exe.resolve()
    selected = dict(control=Path('data/prepared/binding-v2').resolve(),
                    diversity=Path('data/prepared/binding-diversity-v1').resolve())
    manifests = {arm: json.loads((path/'manifest.json').read_text()) for arm, path in selected.items()}
    for arm, path in selected.items():
        for name, record in manifests[arm]['files'].items():
            if sha(path/name) != record['sha256']:
                raise ValueError(f'Prepared source changed: {arm}/{name}')
    for name in ('reading.dat', 'train.sgprobe', 'development.sgprobe', 'test.sgprobe'):
        assert sha(selected['control']/name) == sha(selected['diversity']/name), name
    seeds, reading_end, prereq_end, endpoints, capacity, speech = (
        ([1337], 16, 32, [64, 96, 128], 16, 16) if smoke else
        ([1337, 2026, 31415], 6000, 10000, [34000, 67000, 130000], 1024, 500))
    out.mkdir(parents=True, exist_ok=False)
    ancestor_schedule = out/'ancestor.sg'
    ancestor_schedule.write_text('SGCURRICULUM3\n' +
        f'{reading_end} {quoted((selected["control"]/"reading.dat").as_posix())} 1 all 1\n')
    schedules, source_records, boundaries = {}, {}, {}
    for arm, path in selected.items():
        schedule = out/f'{arm}.sg'
        schedule.write_text(ancestor_schedule.read_text() +
            f'{prereq_end} {quoted((path/"through-prerequisites.dat").as_posix())} 0.25 new 64\n' +
            f'{endpoints[-1]} {quoted((path/"through-binding.dat").as_posix())} 0.25 new 64\n')
        _, stages = read_schedule(schedule)
        schedules[arm] = schedule
        source_records[arm] = [dict(source=s['source'].as_posix(), sha256=sha(s['source']),
                                    bytes=len(s['content']), documents=len(s['content'].split(b'\x1e')))
                               for s in stages]
        boundaries[arm] = [record['documents'] for record in source_records[arm]]
    validation = Path('data/prepared/development-v2-final/13853.txt').resolve()
    protocol = dict(status='smoke_only' if smoke else 'declared_before_training',
                    seeds=seeds, arms=['control', 'diversity'], cell='selective',
                    channels=256, hidden=512, layers=4, chunk=128, learning_rate=.0003,
                    later_stage_multiplier=.25, answer_emphasis=64, tf32=True,
                    replay_policy='stage', replay_slots=capacity, replay_every=4,
                    reading_end=reading_end, prerequisite_end=prereq_end, online_endpoints=endpoints,
                    generated_bytes_per_speech=96, speak_every=speech, graph_speech=True,
                    save_every=32 if smoke else 5000, log_every=64 if smoke else 6000,
                    executable_sha256=sha(exe), source_records=source_records,
                    source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                    driver_sha256=sha(Path(__file__)),
                    selected_spec_sha256={arm: m['spec_sha256'] for arm, m in manifests.items()},
                    schedule_sha256={arm: sha(path) for arm, path in schedules.items()},
                    ancestor_schedule_sha256=sha(ancestor_schedule),
                    validation_reader_sha256=sha(validation),
                    probe_sha256={name: sha(selected['diversity']/name) for name in
                        ('train.sgprobe', 'development.sgprobe', 'test.sgprobe', 'expanded-train-monitor.sgprobe')},
                    primary_endpoint=f'Original development joint accuracy at {endpoints[-1]} observations; '
                                     'each group requires both query targets under both fact assignments.',
                    secondary_endpoints='Original training fit, expanded-training monitor, unconstrained exact '
                                        'answer generation, earlier-reader loss, exposure and live runtime.',
                    reserved_test_evaluated=False,
                    comparisons='Both arms resume the same exact learned reading checkpoint per seed. '
                                'Learning updates, replay slots/updates and speech counts match. Source volume, '
                                'vocabulary, repetition and target-byte exposure can differ. No checkpoint selection.',
                    monitor='Preselected 432-group sample of expanded training; never full expanded-training accuracy.',
                    timing='Sum of live continuation segments after the common ancestor; includes replay, speech, '
                           'logging and checkpoint I/O. Excludes setup and held-out evaluation.')
    write(out/'protocol.json', protocol)
    rows, ancestors = [], []
    native = NativeCommands(exe, out)

    for seed_index, seed in enumerate(seeds):
        ancestor = out/f'{seed}-ancestor'
        native('live', '--curriculum', ancestor_schedule, '--out', ancestor, '--updates', reading_end,
               '--cell', 'selective', '--channels', 256, '--hidden', 512, '--layers', 4,
               '--seed', seed, '--lr', .0003, '--chunk', 128, '--replay', 'stage',
               '--replay-capacity', capacity, '--replay-every', 4, '--graph', '--fast',
               '--speak-every', speech, '--tokens', 96, '--prompt', 'The bird ',
               '--validation', validation, '--eval-batches', 32,
               '--log-every', protocol['log_every'], '--save-every', protocol['save_every'])
        source = ancestor/'latest.ckpt'
        meta, _, _, ancestor_sha = checkpoint(source)
        assert meta[24] == reading_end
        ancestors.append(dict(seed=seed, checkpoint_sha256=ancestor_sha,
                              session=json.loads((ancestor/'session.json').read_text()),
                              both_arms_resume_identical_checkpoint=True))
        for point_index, end in enumerate(endpoints):
            arms = ('control', 'diversity') if (seed_index+point_index) % 2 == 0 else ('diversity', 'control')
            for arm in arms:
                dest = out/f'{seed}-{arm}'
                initialization = (['--resume', dest/'latest.ckpt', '--curriculum', schedules[arm]] if point_index else
                                  ['--resume', source, '--curriculum', ancestor_schedule, '--extend-curriculum', schedules[arm]])
                native('live', '--out', dest, '--updates', end, *initialization, '--prompt', 'The bird ',
                       '--validation', validation, '--eval-batches', 32,
                       '--log-every', protocol['log_every'], '--save-every', protocol['save_every'])
                session = json.loads((dest/'session.json').read_text())
                assert session['online_updates'] == end
                assert session['gradient_reductions'] == 'ordered_v1'
                assert session['curriculum_extended'] == (point_index == 0)
                snapshot = dest/f'checkpoint-{end}.ckpt'
                shutil.copyfile(dest/'latest.ckpt', snapshot)
                meta, _, _, snapshot_sha = checkpoint(snapshot)
                slots = distribution(snapshot, boundaries[arm])
                assert slots == [capacity//3 + int(i < capacity % 3) for i in range(3)]
                row = dict(seed=seed, arm=arm, online_updates=end, parameters=meta[14],
                           checkpoint_sha256=snapshot_sha, stored_windows_by_source_stage=slots,
                           session=session, ancestor_checkpoint_sha256=ancestor_sha)
                # train continues to mean the original six-object training set.
                row.update(binding_scores(native, snapshot, selected['diversity'], dest, end))
                assert sha(snapshot) == snapshot_sha
                rows.append(row)
                write(out/'partial.json', rows)
                print(f'{seed} {arm} {end}: reader={session["final_validation_loss"]:.5f} '
                      f'train={row["train"]["joint_accuracy"]:.4f} dev={row["development"]["joint_accuracy"]:.4f} '
                      f'expanded-monitor={row["expanded-train-monitor"]["joint_accuracy"]:.4f}', flush=True)
            paired = [r for r in rows if r['seed'] == seed and r['online_updates'] == end]
            for field in ('online_updates', 'global_updates', 'replay_updates', 'generated_bytes',
                          'learning_rate', 'replay_state_bytes', 'replay_items'):
                assert paired[0]['session'][field] == paired[1]['session'][field], (seed, end, field)
        assert sha(source) == ancestor_sha
    assert sha(exe) == protocol['executable_sha256']
    assert sha(validation) == protocol['validation_reader_sha256']
    for arm, path in selected.items():
        for name, record in manifests[arm]['files'].items():
            assert sha(path/name) == record['sha256'], f'Source changed during run: {arm}/{name}'
    result = dict(protocol=protocol, native_commands=len(native.commands), shared_ancestors=ancestors, runs=rows)
    write(out/'comparison.json', result)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--exe', type=Path, default=Path('build/synapticgenesis.exe'))
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--smoke', action='store_true', help='Tiny execution/extension check; no learning-quality evidence')
    a = p.parse_args()
    run(a.exe, a.out, a.smoke)
