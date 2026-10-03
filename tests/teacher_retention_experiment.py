"""Independent exposure, source eligibility, immutable teacher and paired audit."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shlex
import struct
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from experiment_checkpoint import checkpoint, state_record, teaching
from native_experiment import NativeCommands, read, sha, write


def check(root, exe, continuation_control=False):
    data = read(root/'comparison.json')
    p = data['protocol']
    parent, previous = Path(p['parent_directory']), Path(p['previous_narrative_directory'])
    base, endpoints = p['baseline_online_updates'], p['online_endpoints']
    models = len(p['seeds'])*len(p['architectures'])
    assert sha(exe) == p['executable_sha256']
    assert len(data['runs']) == models*6
    assert len({(r['seed'], r['arm'], r['online_updates']) for r in data['runs']}) == models*6
    commands = read(root/'commands.json')
    assert len(commands) == data['native_commands'] == models*37
    assert Counter(c[1] for c in commands) == dict(live=models*4, evaluate=models*15, sample=models*10,
                                                  **{'language-probes': models*5, 'decode-bench': models*2,
                                                     'teacher-pack': models})
    expected_order = []
    for seed_index, seed in enumerate(p['seeds']):
        architectures = p['architectures']
        rotated = architectures[seed_index % 3:]+architectures[:seed_index % 3]
        for architecture in rotated:
            variants = p['variants'] if (seed_index+architectures.index(architecture)) % 2 == 0 else list(reversed(p['variants']))
            expected_order.extend(f'{seed}-{architecture}-{variant}' for variant in variants for _ in endpoints)
    actual_order = [Path(c[c.index('--out')+1]).name for c in commands if c[1] == 'live']
    assert actual_order == expected_order, 'Branch order differs from the fixed alternating policy'
    for command in commands:
        if '--data' in command:
            source = Path(command[command.index('--data')+1])
            if command[1] == 'teacher-pack':
                assert sha(source) == p['teacher_policy']['source_sha256']
            else:
                assert source.name in ('13853.txt', '12228.txt', '11757.txt')
        if '--probes' in command:
            assert Path(command[command.index('--probes')+1]).name == 'development.sgprobe'
    assert not p['reserved_test_evaluated']
    assert p['teacher_policy']['temperature'] == 2 and p['teacher_policy']['strength'] == .5
    assert sha(parent/'comparison.json') == p['parent_comparison_sha256']
    assert sha(parent/'curriculum.sg') == p['parent_schedule_sha256']
    assert sha(previous/'comparison.json') == p['previous_narrative_sha256']
    admission = p['source_admission']
    assert admission['previous_stages'] == 3 and admission['added_documents'] == 7
    assert [r['answer_scale'] for r in admission['editions']] == [1, 64, 64, 1]
    for edition in admission['editions']:
        assert sha(root/'edition'/edition['file']) == edition['sha256']
    docs = (Path(p['prepared_directory'])/'train.dat').read_bytes().split(b'\x1e')
    assert len(docs) == 7
    windows = [min(128, len(doc)-1-offset) for doc in docs for offset in range(0, len(doc)-1, 128)]
    pairs = {end: ((end-base)//len(windows))*sum(windows)+sum(windows[:(end-base) % len(windows)])
             for end in endpoints}
    checked, old_controls = [], []
    for seed in p['seeds']:
        for architecture in p['architectures']:
            name = f'{seed}-{architecture}'
            ancestor = parent/name/f'checkpoint-{base}.ckpt'
            initial = state_record(ancestor)
            assert initial['checkpoint_sha256'] == p['parent_checkpoints'][name]
            original_meta = checkpoint(ancestor)[0]
            bundle = root/(name+'-teacher-bundle')
            assert {f.name: sha(f) for f in bundle.iterdir() if f.is_file()} == data['bundles'][name]
            assert sha(bundle/'teacher-0.ckpt') == sha(ancestor)
            assert sha(bundle/'source.dat') == p['teacher_policy']['source_sha256']
            manifest_teacher = tuple(map(int, shlex.split((bundle/'teachers.sg').read_text().splitlines()[2])[1:]))
            for variant in p['variants']:
                arm = f'{architecture}-{variant}'
                directory = root/f'{seed}-{arm}'
                assert sha(directory/f'checkpoint-{base}.ckpt') == sha(ancestor)
                events = [json.loads(line) for line in (directory/'metrics.jsonl').read_text().splitlines()]
                extension = [e for e in events if e.get('event') == 'curriculum_extension']
                assert len(extension) == 1 and extension[0]['online_update'] == base
                assert extension[0]['replay_windows_preserved'] == p['replay_slots']
                policies = [e for e in events if e.get('event') == 'teacher_policy']
                assert len(policies) == int(variant == 'teacher')
                if policies:
                    assert policies[0]['admitted'] and policies[0]['online_update'] == base
                for end in [base]+endpoints:
                    path = directory/f'checkpoint-{end}.ckpt'
                    row = next(r for r in data['runs'] if r['seed'] == seed and r['arm'] == arm and r['online_updates'] == end)
                    state = state_record(path)
                    assert all(row[k] == value for k, value in state.items())
                    for i, sample in enumerate(row['samples']):
                        assert sha(directory/f'sample-{end}-{i}.txt') == sample['file_sha256']
                    if end == base:
                        continue
                    delta, replay_delta = end-base, end//4-base//4
                    assert state['global_updates'] == initial['global_updates']+delta+replay_delta
                    assert state['observed_pairs'] == initial['observed_pairs']+pairs[end]
                    assert state['replay_updates'] == initial['replay_updates']+replay_delta
                    assert state['generated_bytes'] == initial['generated_bytes']+96*(end//p['speak_every']-base//p['speak_every'])
                    assert state['curriculum_stage'] == 4
                    assert [g['stored_windows'] for g in state['replay_groups']] == p['replay_quotas']
                    assert state['replay_groups'][-1]['seen_windows'] == delta
                    for earlier, later in zip(initial['replay_groups'], state['replay_groups'][:3]):
                        assert earlier['document_end'] == later['document_end'] and earlier['seen_windows'] == later['seen_windows']
                    session = row['session']
                    assert session['shared_weights'] and session['persistent_membranes'] and not session['trains_on_generated_text']
                    assert abs(session['final_validation_loss']-row['books']['reader']['loss_nats_per_byte']) < 1e-6
                    if variant == 'control':
                        old = previous/name/f'checkpoint-{end}.ckpt'
                        assert sha(path) == sha(old), (name, end, 'Ordinary continuation changed')
                        old_controls.append(dict(seed=seed, architecture=architecture, online_updates=end,
                                                 checkpoint_sha256=sha(path), full_checkpoint_identical=True))
                    else:
                        t = teaching(path)
                        assert t[2] == t[3] == 1 and t[14] == 0
                        assert t[7] == base and t[8] == initial['replay_updates'] and t[15] == initial['replay_pairs']
                        assert t[5] == original_meta[12] and t[6] == p['source_boundaries'][2]
                        assert t[16:24] == manifest_teacher and not any(t[24:])
                        assert t[11:14] == tuple(struct.unpack('<I', struct.pack('<f', v))[0] for v in (2, .5, .5))
                        for field, word in [('replay_updates', 9), ('replay_pairs', 10)]:
                            eligible = sum(g[field] for g in state['replay_groups'][:3])-sum(g[field] for g in initial['replay_groups'])
                            assert t[word] == eligible > 0
                        assert session['teacher_updates'] == t[9] and session['teacher_pairs'] == t[10]
                        assert 0 < session['teacher_extra_gpu_bytes'] <= 512*1024**2
                    if end == endpoints[-1]:
                        assert row['decode']['generated_bytes_identical'] and row['decode']['state_logits_max_error'] == 0
                    checked.append(dict(seed=seed, arm=arm, online_updates=end, checkpoint_sha256=sha(path)))
            for end in [base]+endpoints:
                a = root/f'{seed}-{architecture}-control'/f'checkpoint-{end}.ckpt'
                b = root/f'{seed}-{architecture}-teacher'/f'checkpoint-{end}.ckpt'
                am, ae, *_ = checkpoint(a); bm, be, *_ = checkpoint(b)
                assert ae == be, 'Teacher changed replay descriptors, random draws or exposure'
                assert all(am[i] == bm[i] for i in (7, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30))
    result = dict(passed=True, smoke_only=p['status'] == 'smoke_only', native_commands=len(commands),
                  branches=models*2, checkpoint_rows=len(data['runs']), checked_endpoints=checked,
                  ordinary_controls_match_previous_study=old_controls,
                  source_exposure_and_descriptors_match_between_variants=True,
                  independent_expected_observed_pairs=pairs, selected_sources_and_parents_authenticated=True,
                  frozen_teachers_equal_own_pre_narrative_parent=True, teacher_eligibility_counters_exact=True,
                  teacher_memory_cap_observed=True, graph_generation_exact=True, reserved_tests_not_scored=True)
    controls = root/'continuation-controls'
    if continuation_control or controls.exists():
        assert result['smoke_only']
        new_controls = not controls.exists()
        if new_controls:
            controls.mkdir()
        native = NativeCommands(exe, controls) if new_controls else None
        uninterrupted = []
        if not new_controls:
            saved_commands = read(controls/'commands.json')
            assert len(saved_commands) == len(p['architectures'])*len(p['variants'])
        for architecture in p['architectures']:
            name = f'{p["seeds"][0]}-{architecture}'
            for variant in p['variants']:
                directory = controls/(name+'-'+variant)
                extra = ['--teacher-bundle', root/(name+'-teacher-bundle'), '--teacher-memory-mib', 512] if variant == 'teacher' else []
                arguments = ['live', '--resume', parent/name/f'checkpoint-{base}.ckpt', '--curriculum', parent/'curriculum.sg',
                             '--extend-curriculum', root/'edition/curriculum.sg', '--out', directory, '--updates', endpoints[-1],
                             '--prompt', 'The bird ', '--validation', Path(p['prepared_directory'])/'13853.txt',
                             '--eval-batches', p['evaluation_batches'], '--log-every', p['log_every'], '--save-every', p['save_every'], *extra]
                if native:
                    native(*arguments)
                else:
                    assert saved_commands[len(uninterrupted)] == [str(exe.resolve()), *map(str, arguments)]
                expected = root/(name+'-'+variant)/f'checkpoint-{endpoints[-1]}.ckpt'
                assert (directory/'latest.ckpt').read_bytes() == expected.read_bytes()
                uninterrupted.append(dict(architecture=architecture, variant=variant, complete_checkpoint_identical=True,
                                          checkpoint_sha256=sha(expected)))
        result['uninterrupted_controls'] = uninterrupted
        result['additional_control_commands'] = len(uninterrupted)
    write(root/'execution-check.json', result)
    print(dict(passed=True, branches=models*2, native_commands=len(commands),
               matched_old_checkpoints=len(old_controls), uninterrupted_controls=len(result.get('uninterrupted_controls', []))))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--exe', type=Path, default=Path('build/synapticgenesis.exe'))
    p.add_argument('--continuation-control', action='store_true')
    a = p.parse_args()
    check(a.root, a.exe, a.continuation_control)
