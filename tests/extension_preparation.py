"""Independent file checks and native acceptance of prepared curriculum additions."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from extend_curriculum import prepare
from curriculum_cli import payload


def check(exe, out):
    exe, out = exe.resolve(), out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    first = b'A child sees the bird in the tree. ' * 4
    second = b'The key is in the box.\nAnswer: box.'
    third = b'The cup is in the bag.\nAnswer: bag.'
    (out / 'one space.dat').write_bytes(first)
    (out / 'two.dat').write_bytes(first + b'\x1e' + second)
    source = out / 'new.dat'
    source.write_bytes(third)
    holdout = out / 'heldout.dat'
    holdout.write_bytes(b'A quiet bird rests on the garden wall. ' * 6)
    calls, error = 0, 0

    def run(*args):
        nonlocal calls
        calls += 1
        p = subprocess.run([str(exe), *map(str, args)], capture_output=True)
        (out / f'native-{calls}.log').write_bytes(p.stdout + p.stderr)
        assert p.returncode == 0, (args, p.stdout, p.stderr)

    for version in (1, 2, 3):
        old = out / f'v{version}.sg'
        suffix = ['', ' all', ' all 1'][version-1]
        # Fractional rate spelling is intentionally retained through preparation.
        old.write_bytes(f'SGCURRICULUM{version}\r\n3 "one space.dat" 1{suffix}\r\n6 "two.dat" 0.3125{suffix}\r\n'.encode())
        prepared = out / f'prepared-{version}'
        manifest = prepare(old, source, prepared, 10, answer_scale=64, holdouts=[holdout])
        assert (prepared / 'edition-1.dat').read_bytes() == first
        assert (prepared / 'edition-2.dat').read_bytes() == first + b'\x1e' + second
        assert (prepared / 'edition-3.dat').read_bytes() == first + b'\x1e' + second + b'\x1e' + third
        assert manifest['parent_schedule_sha256'] == hashlib.sha256(old.read_bytes()).hexdigest()
        for item in manifest['editions']:
            assert item['sha256'] == hashlib.sha256((prepared / item['file']).read_bytes()).hexdigest()
        common = ['--channels', 8, '--hidden', 16, '--layers', 2, '--cell', 'gated', '--chunk', 16,
                  '--replay', 'reservoir', '--replay-every', 2, '--replay-capacity', 32,
                  '--consolidation', 'si', '--si-strength', .02, '--graph', '--lr', .001,
                  '--seed', 1337, '--speak-every', 2, '--tokens', 7, '--prompt', 'A']
        full, part = out / f'full-{version}', out / f'part-{version}'
        run('live', '--curriculum', prepared / 'curriculum.sg', '--out', full, *common)
        run('live', '--curriculum', old, '--out', part, *common)
        run('live', '--curriculum', old, '--extend-curriculum', prepared / 'curriculum.sg',
            '--resume', part / 'latest.ckpt', '--out', part, '--prompt', 'A')
        expected, actual = payload(full / 'latest.ckpt'), payload(part / 'latest.ckpt')
        assert expected[0][:15] == actual[0][:15] and expected[0][16:] == actual[0][16:]
        assert expected[1:3] == actual[1:3] and len(expected[3]) == len(actual[3])
        error = max(error, max(abs(a-b) for a, b in zip(expected[3], actual[3])))
        assert error < 3e-5

    rejects = 0
    bad_out = out / 'rejected'
    def reject(message, **overrides):
        nonlocal rejects
        kwargs = dict(curriculum=old, data=source, out=bad_out, end_update=12, answer_scale=64)
        kwargs.update(overrides)
        try:
            prepare(**kwargs)
        except (ValueError, FileExistsError) as exc:
            assert message in str(exc), str(exc)
        else:
            raise AssertionError('Invalid source preparation accepted')
        assert not bad_out.exists()
        rejects += 1

    reject('later stage', end_update=6)
    reject('later stage', rate_scale=float('nan'))
    reject('held-out', holdouts=[source])
    reject('held-out', holdouts=[out / 'one space.dat'])
    source.write_bytes(second)
    reject('repeats an exact document')
    source.write_bytes(third + b'\x1e' + third)
    reject('repeats an exact document')
    source.write_bytes(b'No labelled answer here.')
    reject('Answer field')
    source.write_bytes(third + b'\x1e')
    reject('no empty separators')
    source.write_bytes(third)
    reject('exists', out=prepared)
    (out / 'two.dat').write_bytes(b'changed' + first + b'\x1e' + second)
    reject('unchanged documents')
    result = dict(passed=True, native_commands=calls, schedule_versions=[1, 2, 3],
                  source_bytes_and_sha256_verified=True, paths_with_spaces=True,
                  native_final_payload_max_error=error, rejected_invalid_preparations=rejects,
                  duplicate_and_heldout_documents_rejected=True, existing_output_not_overwritten=True)
    (out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--exe', type=Path, default=Path('build/synapticgenesis.exe'))
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    check(a.exe, a.out)
