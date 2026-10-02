"""Process-level resume, input rejection and v4 diagnostic checks (stdlib only).

Model computation runs in native C++/CUDA. Python reads saved results and checks
that separate process lifetimes do not change the development trajectory.
"""
import argparse
from array import array
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import subprocess


def payload(path):
    raw = path.read_bytes()
    meta = struct.unpack_from('<32Q', raw)
    hp = struct.unpack_from('<8f', raw, 256)
    count = 3 * meta[14] + meta[18]
    floats = array('f')
    floats.frombytes(raw[288:288 + 4 * count])
    offset = 288 + 4 * count
    extra = struct.unpack_from(f'<{meta[31]}Q', raw, offset)
    synapses = array('f')
    synapses.frombytes(raw[offset + 8 * meta[31]:])
    floats.extend(synapses)
    return meta, hp, extra, floats


def check(exe, out):
    exe = exe.resolve()
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    docs = [b'A child sees a bird in a tree. The bird sings to the child. ' * 3,
            b'A seed needs water and sunlight. A child waters the garden. ' * 3,
            b'Water flows down the hill. Plants live near the stream. ' * 3]
    for index in range(3):
        (out / f'{index}.dat').write_bytes(b'\x1e'.join(docs[:index + 1]))
    schedule = out / 'curriculum.sg'
    schedule.write_bytes(b'SGCURRICULUM1\n3 "0.dat" 1\n6 "1.dat" 0.5\n9 "2.dat" 0.25\n')
    calls = 0

    def run(*args, reject=None):
        nonlocal calls
        calls += 1
        result = subprocess.run([str(exe), *map(str, args)], capture_output=True, text=True)
        (out / f'command-{calls:02d}.log').write_text(result.stdout + result.stderr, encoding='utf-8')
        if reject:
            assert result.returncode != 0 and reject in result.stderr, (args, result.stdout, result.stderr)
        elif result.returncode:
            raise RuntimeError(f'{args}: {result.stdout}\n{result.stderr}')

    common = ['--channels', 8, '--hidden', 16, '--layers', 2, '--chunk', 8,
              '--replay', 'reservoir', '--replay-capacity', 16, '--replay-every', 2,
              '--lr', .0008, '--graph', '--seed', 1337, '--speak-every', 2, '--tokens', 7,
              '--prompt', 'A', '--log-every', 1, '--save-every', 100]
    spec = importlib.util.spec_from_file_location('dynamics', Path(__file__).resolve().parents[1] / 'scripts/inspect_dynamics.py')
    dynamics = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dynamics)
    max_error = 0
    cases = 0
    for cell in ('lif', 'alif'):
        for si in (False, True):
            name = f'{cell}-{int(si)}'
            policy = ['--cell', cell] + (['--consolidation', 'si', '--si-strength', .02] if si else [])
            full = out / (name + '-full')
            run('live', '--curriculum', schedule, '--out', full, *common, *policy)
            expected = payload(full / 'latest.ckpt')
            report = dynamics.inspect(full / 'latest.ckpt')
            assert report['curriculum']['stage'] == 3
            assert ('synaptic_memory' in report) == si
            transitions = [json.loads(row) for row in (full / 'metrics.jsonl').read_text().splitlines()
                           if 'curriculum_transition' in row]
            assert [r['online_update'] for r in transitions] == [3, 6]
            assert [r['new_document'] for r in transitions] == [1, 2]
            assert [r['replay_windows_preserved'] for r in transitions] == [3, 6]
            for index in range(3):
                archived = payload(full / f'stage-{index + 1}.ckpt')
                assert archived[0][24] == 3 * (index + 1) and archived[2][15] == index
            for split in (2, 3, 4):
                part = out / f'{name}-{split}'
                run('live', '--curriculum', schedule, '--out', part, '--updates', split, *common, *policy)
                run('live', '--curriculum', schedule, '--out', part, '--resume', part / 'latest.ckpt', '--prompt', 'A')
                actual = payload(part / 'latest.ckpt')
                assert expected[0][:15] == actual[0][:15] and expected[0][16:] == actual[0][16:]
                assert expected[1:3] == actual[1:3]
                error = max(abs(a-b) for a, b in zip(expected[3], actual[3]))
                assert len(expected[3]) == len(actual[3]) and error < 3e-5, error
                max_error = max(max_error, error)
                def speech(folder):
                    return [block for block in (folder / 'transcript.txt').read_bytes().split(b'\n[')
                            if block.startswith(b'online update')]
                assert speech(full) == speech(part)
                cases += 1
    # Existing v2 history can be bound to a curriculum without resetting it.
    prior = out / 'prior-v2'
    run('live', '--data', out / '0.dat', '--out', prior, '--updates', 2, *common)
    run('live', '--resume', prior / 'latest.ckpt', '--curriculum', schedule,
        '--out', prior, '--updates', 7, '--prompt', 'A', '--lr', .0002)
    state = payload(prior / 'latest.ckpt')
    assert state[0][24] == 7 and state[2][5] == 7 and state[2][15] == 2
    assert abs(state[1][0] - .00005) < 1e-10 and abs(state[1][7] - .0002) < 1e-10
    rejected = out / 'rejected'
    run('live', '--resume', prior / 'latest.ckpt', '--out', rejected,
        reject='requires --curriculum')
    (out / '2.dat').write_bytes(b'\x1e'.join(docs) + b' New future text.')
    run('live', '--resume', prior / 'latest.ckpt', '--curriculum', schedule,
        '--out', rejected, '--prompt', 'A', reject='identical curriculum')
    (out / '2.dat').write_bytes(b'\x1e'.join(docs))
    (out / '1.dat').write_bytes(b'Changed ' + b'\x1e'.join(docs[:2]))
    run('live', '--curriculum', schedule, '--out', rejected,
        reject='append complete documents')
    (out / '1.dat').write_bytes(b'\x1e'.join(docs[:2]))
    (out / 'future-heldout.dat').write_bytes(docs[2])
    run('live', '--curriculum', schedule, '--validation', out / 'future-heldout.dat', '--out', rejected,
        *common, reject='held-out document')
    assert not rejected.exists(), 'Invalid input wrote training artifacts'
    # Inference/evaluation read v4 without requiring the training schedule.
    selected = out / 'alif-1-full/latest.ckpt'
    run('sample', '--checkpoint', selected, '--prompt', 'A', '--tokens', 8)
    run('evaluate', '--checkpoint', selected, '--data', out / '0.dat', '--context', 8, '--batch', 2, '--batches', 2)
    result = {'passed': True, 'native_commands': calls, 'resume_cases': cases,
              'resume_max_error': max_error, 'identical_generated_bytes': True,
              'both_cells_optional_si': True, 'v2_binding_retains_history': True,
              'rate_override_retains_stage_scale': True, 'future_data_mutation_rejected': True,
              'changed_prefix_rejected': True, 'future_heldout_document_rejected': True,
              'invalid_input_writes_no_run': True, 'v4_read_only_commands_checked': True,
              'checkpoint_sha256': hashlib.sha256(selected.read_bytes()).hexdigest()}
    (out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exe', type=Path, default=Path('build/synapticgenesis.exe'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    check(args.exe, args.out)
