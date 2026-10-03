"""Teacher replay at the three established model shapes; not a quality study."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from experiment_checkpoint import checkpoint, teaching
from native_experiment import read, write, sha


def check(args):
    args.out.mkdir(parents=True, exist_ok=False)
    baseline = read(args.baseline)
    executable_hash = sha(args.exe)
    commands = []
    def run(command):
        assert sha(args.exe) == executable_hash
        commands.append(command)
        write(args.out/'commands.json', commands)
        p = subprocess.run(command, capture_output=True, text=True)
        (args.out/f'command-{len(commands):02d}.log').write_text(p.stdout+p.stderr)
        assert not p.returncode, p.stderr
    parents = [Path(baseline['commands'][i][baseline['commands'][i].index('--resume')+1]) for i in (0, 2)]
    initial_hashes = [sha(p) for p in parents]
    bundle = args.out/'teachers'
    run([str(args.exe.resolve()), 'teacher-pack', '--teacher-a', str(parents[0]), '--teacher-b', str(parents[1]),
         '--data', str(args.source), '--out', str(bundle), '--temperature', '2', '--strength', '.7', '--mixture', '.3'])
    records = []
    for original, expected in zip(baseline['commands'], baseline['continuations']):
        name = expected['model']
        command = original.copy()
        command[0] = str(args.exe.resolve())
        full, split = args.out/name, args.out/(name+'-split')
        command[command.index('--out')+1] = str(full)
        command += ['--teacher-bundle', str(bundle), '--teacher-memory-mib', '512']
        run(command)
        resumed = command.copy()
        resumed[resumed.index('--out')+1] = str(split)
        resumed[resumed.index('--updates')+1] = '256'
        run(resumed)
        extended = resumed[resumed.index('--extend-curriculum')+1]
        at = resumed.index('--extend-curriculum'); del resumed[at:at+2]
        resumed[resumed.index('--curriculum')+1] = extended
        resumed[resumed.index('--resume')+1] = str(split/'latest.ckpt')
        resumed[resumed.index('--updates')+1] = '384'
        run(resumed)
        a, b = full/'latest.ckpt', split/'latest.ckpt'
        assert a.read_bytes() == b.read_bytes(), name
        meta, extra, payload, digest = checkpoint(a)
        policy = teaching(a)
        assert meta[16] == 1 and policy[3] == 2 and policy[9] > 0
        control = args.controls/name/'latest.ckpt'
        assert sha(control) == expected['checkpoint_sha256']
        cm, ce, cp, _ = checkpoint(control)
        assert extra == ce and payload != cp
        assert all(meta[i] == cm[i] for i in (7, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30))
        report = read(full/'session.json')
        assert report['teacher_math'] == 'strict_fp32' and report['teacher_extra_gpu_bytes'] <= 512*1024**2
        records.append(dict(model=name, complete_checkpoint_resume_exact=True, learner_math='tf32',
                            teacher_math='strict_fp32', channels=meta[2], hidden=meta[3], layers=meta[4],
                            parameters=meta[14], teacher_updates=policy[9], teacher_pairs=policy[10],
                            teacher_extra_gpu_bytes=report['teacher_extra_gpu_bytes'], checkpoint_sha256=digest,
                            source_replay_and_speech_counters_match_control=True, learned_buffers_differ=True))
    assert initial_hashes == [sha(p) for p in parents]
    result = dict(passed=True, executable_sha256=executable_hash, native_commands=len(commands),
                  baseline_report_sha256=sha(args.baseline), teacher_source_sha256=sha(args.source),
                  teacher_checkpoint_sha256=initial_hashes, records=records,
                  scope='128-to-384 observation mechanics smoke; no learning-quality or speed claim')
    write(args.out/'result.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--exe', type=Path, default=Path('build/synapticgenesis.exe'))
    p.add_argument('--baseline', type=Path, default=Path('reports/distillation-compatibility.json'))
    p.add_argument('--source', type=Path, default=Path('data/prepared/binding-diversity-v1/through-binding.dat'))
    p.add_argument('--controls', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    check(p.parse_args())
