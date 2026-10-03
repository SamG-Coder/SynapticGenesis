"""Compare learned teacher continuations with the archived full-buffer runtime.

Every predeclared architecture/seed is included. This checks state and allocation
compatibility after 512 additional observations, not language quality or speed.
"""
import argparse
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from native_experiment import NativeCommands, read, sha, write
from experiment_checkpoint import state_record


def check(study, old, new, out):
    data = read(study/'comparison.json')
    p = data['protocol']
    assert p['status'] == 'declared_before_training' and len(data['runs']) == 54
    assert sha(old) == p['executable_sha256'] and sha(new) != sha(old)
    before, after = 160000, 160512
    source_files = sorted(Path('src').glob('*.cu'))+sorted(Path('src').glob('*.cuh'))+[Path(__file__)]
    source_hashes = {str(path): sha(path) for path in source_files}
    protocol = dict(old_executable_sha256=sha(old), new_executable_sha256=sha(new),
                    source_base_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                    source_sha256=source_hashes, study_sha256=sha(study/'comparison.json'),
                    checkpoints='Every frozen-self branch at observation 160000',
                    before=before, after=after, source_schedule_sha256=sha(study/'edition/curriculum.sg'),
                    scope='Complete learned checkpoint equality across 512 additional observations, including '
                          'replay and one 96-byte speech event. Explicit teacher-buffer bytes, excluding CUDA '
                          'and cuBLAS overhead. Timings are observed costs, not an isolated speed benchmark.')
    out.mkdir(parents=True, exist_ok=False)
    write(out/'protocol.json', protocol)
    native, records = NativeCommands(new, out), []
    for seed in p['seeds']:
        for architecture in p['architectures']:
            name = f'{seed}-{architecture}'
            parent = study/(name+'-teacher')/f'checkpoint-{before}.ckpt'
            row = next(r for r in data['runs'] if r['seed'] == seed and r['architecture'] == architecture
                       and r['variant'] == 'teacher' and r['online_updates'] == before)
            assert sha(parent) == row['checkpoint_sha256']
            bundle = study/(name+'-teacher-bundle')
            identities = {file.name: sha(file) for file in bundle.iterdir() if file.is_file()}
            assert identities == data['bundles'][name]
            destinations = {}
            order = [('legacy', old), ('forward-only', new)]
            if len(records) % 2:
                order.reverse()
            for label, executable in order:
                assert sha(old) == protocol['old_executable_sha256']
                assert sha(new) == protocol['new_executable_sha256']
                native.exe = executable.resolve()
                destination = out/name/label
                native('live', '--resume', parent, '--curriculum', study/'edition/curriculum.sg',
                       '--out', destination, '--updates', after, '--prompt', p['live_prompt'],
                       '--teacher-bundle', bundle, '--teacher-memory-mib', 512,
                       '--validation', Path(p['prepared_directory'])/'13853.txt', '--eval-batches', 2,
                       '--log-every', 512, '--save-every', 512)
                destinations[label] = destination
            a, b = [destinations[label] for label in ('legacy', 'forward-only')]
            identity = sha(a/'latest.ckpt')
            assert sha(b/'latest.ckpt') == identity, (name, 'Complete learned checkpoint changed')
            original, optimized = read(a/'session.json'), read(b/'session.json')
            assert original['online_updates'] == optimized['online_updates'] == after
            saved = state_record(b/'latest.ckpt')
            assert saved['online_updates'] == after
            assert saved['global_updates'] == row['global_updates']+512+128
            assert saved['replay_updates'] == row['replay_updates']+128
            assert saved['generated_bytes'] == row['generated_bytes']+96
            assert original['generated_bytes'] == optimized['generated_bytes'] == saved['generated_bytes']
            assert sha(a/'transcript.txt') == sha(b/'transcript.txt'), (name, 'Generated speech changed')
            assert original['teacher_updates'] == optimized['teacher_updates'] > row['teaching']['updates']
            assert original['teacher_pairs'] == optimized['teacher_pairs'] > row['teaching']['pairs']
            old_bytes, new_bytes = [r['teacher_extra_gpu_bytes'] for r in (original, optimized)]
            assert 0 < new_bytes < old_bytes
            assert sha(parent) == row['checkpoint_sha256']
            assert {file.name: sha(file) for file in bundle.iterdir() if file.is_file()} == identities
            records.append(dict(seed=seed, architecture=architecture, parent_sha256=row['checkpoint_sha256'],
                                complete_checkpoint_identical=True, checkpoint_sha256=identity,
                                global_updates_added=saved['global_updates']-row['global_updates'],
                                replay_updates_added=saved['replay_updates']-row['replay_updates'],
                                generated_bytes_added=saved['generated_bytes']-row['generated_bytes'],
                                transcript_sha256=sha(b/'transcript.txt'),
                                teacher_updates_added=optimized['teacher_updates']-row['teaching']['updates'],
                                teacher_pairs_added=optimized['teacher_pairs']-row['teaching']['pairs'],
                                legacy_teacher_gpu_bytes=old_bytes, forward_only_teacher_gpu_bytes=new_bytes,
                                explicit_teacher_bytes_saved=old_bytes-new_bytes,
                                observed_legacy_seconds=original['elapsed_seconds'],
                                observed_forward_only_seconds=optimized['elapsed_seconds']))
            write(out/'partial.json', records)
            print(name, 'checkpoint identical; explicit teacher bytes', old_bytes, '->', new_bytes, flush=True)
    assert all(sha(Path(path)) == identity for path, identity in source_hashes.items())
    assert sha(study/'comparison.json') == protocol['study_sha256']
    assert sha(old) == protocol['old_executable_sha256'] and sha(new) == protocol['new_executable_sha256']
    write(out/'result.json', dict(passed=True, protocol=protocol, records=records,
                                 native_commands=len(native.commands), all_study_architectures_and_seeds_checked=True,
                                 parents_and_teacher_bundles_unchanged=True, quality_or_speed_claim=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--study', type=Path, default=Path('runs/teacher-retention-panel'))
    p.add_argument('--old', type=Path, default=Path('runs/legacy-teachers-522/synapticgenesis.exe'))
    p.add_argument('--new', type=Path, default=Path('build/synapticgenesis.exe'))
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    check(a.study, a.old, a.new, a.out)
