"""Native immutable teacher admission, restart, negative controls and rejection.

These are synthetic mechanism fixtures. They make no learning-quality claim.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from experiment_checkpoint import checkpoint, teaching
from inspect_dynamics import inspect


def fnv(raw, value=14695981039346656037):
    for byte in raw:
        value = ((value ^ byte) * 1099511628211) & ((1 << 64)-1)
    return value


def rehash(raw):
    meta = struct.unpack_from('<32Q', raw)
    end = 288 + 12*meta[14] + 4*meta[18]
    struct.pack_into('<Q', raw, 15*8, 0)
    value = fnv(raw[288:end])
    for section in (raw[:256], raw[256:288], raw[end:]):
        value = fnv(section, value)
    struct.pack_into('<Q', raw, 15*8, value)
    return raw


def fingerprints(root):
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob('*') if p.is_file()}


class Fixture:
    def __init__(self, exe, out, compatible=False):
        self.exe, self.out = exe.resolve(), out.resolve()
        self.out.mkdir(parents=True, exist_ok=False)
        self.executable_sha256 = hashlib.sha256(self.exe.read_bytes()).hexdigest()
        self.calls = []
        docs = [b'A child reads a book under the tree. ' * 8,
                b'The key is in the box.\nAnswer: box.',
                b'The hat is in the bag.\nAnswer: bag.',
                b'A seed needs water.\nAnswer: water.']
        for n in range(1, 5):
            (self.out/f'{n}.dat').write_bytes(b'\x1e'.join(docs[:n]))
        rows = ['20 "1.dat" 1 all 1', '40 "2.dat" .5 new 64',
                '64 "3.dat" .25 new 64', '100 "4.dat" .25 new 64']
        for name, n in [('short', 2), ('full', 3), ('extended', 4)]:
            (self.out/f'{name}.sg').write_text('SGCURRICULUM3\n'+'\n'.join(rows[:n])+'\n')
        self.val = self.out/'validation.dat'
        self.val.write_bytes(b'The flowers grow beside a quiet path. ' * 12)
        self.parents = []
        for i, cell in enumerate(['lif', 'lif' if compatible else 'associative']):
            dest = self.out/f'parent-{i}'
            shape = 0 if compatible else i
            self.run('train', '--data', self.out/'1.dat', '--validation', self.val, '--out', dest,
                     '--channels', 8+8*shape, '--hidden', 16+8*shape, '--layers', 1+shape, '--cell', cell,
                     '--batch', 2, '--context', 16, '--steps', 4, '--warmup', 0, '--seed', 91+i,
                     '--eval-every', 4, '--eval-batches', 2)
            self.parents.append(dest/'latest.ckpt')

    def run(self, *args, reject=None):
        assert hashlib.sha256(self.exe.read_bytes()).hexdigest() == self.executable_sha256, 'Executable changed during test'
        command = [str(self.exe), *map(str, args)]
        result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', errors='replace')
        self.calls.append(dict(command=command, returncode=result.returncode, rejection_expected=reject))
        (self.out/'commands.json').write_text(json.dumps(self.calls, indent=2)+'\n')
        (self.out/f'command-{len(self.calls):03d}.log').write_text(result.stdout+result.stderr, encoding='utf-8')
        if reject:
            assert result.returncode and reject in result.stderr, (args, result.stdout, result.stderr)
        else:
            assert not result.returncode, (args, result.stdout, result.stderr)
        return result

    def pack(self, name, strength=.7, source='2.dat', single=False):
        dest = self.out/name
        other = [] if single else ['--teacher-b', self.parents[1]]
        self.run('teacher-pack', '--teacher-a', self.parents[0], *other, '--data', self.out/source,
                 '--out', dest, '--strength', strength, '--mixture', .3)
        return dest


def check(exe, out):
    f = Fixture(exe, out)
    run, out = f.run, f.out
    active, zero, single = f.pack('teachers'), f.pack('zero', 0), f.pack('single', single=True)
    bundle_before = fingerprints(active)
    full = out/'full.sg'
    common = ['--channels', 8, '--hidden', 16, '--layers', 2, '--chunk', 8,
              '--replay', 'stage', '--replay-capacity', 12, '--replay-every', 2,
              '--graph', '--speak-every', 2, '--tokens', 7, '--prompt', 'A', '--log-every', 1]
    controls, restarts = [], []
    for cell in ['lif', 'alif', 'trace', 'gated', 'selective', 'associative']:
        initial = out/(cell+'-start')
        run('live', '--curriculum', full, '--out', initial, '--cell', cell, '--updates', 7, *common)
        source = initial/'latest.ckpt'
        raw_before = source.read_bytes()
        paths = []
        for label, flags in [('control', []), ('off', ['--teacher-bundle', active, '--teaching', 'off', '--teacher-memory-mib', 0]),
                             ('zero', ['--teacher-bundle', zero, '--teacher-memory-mib', 0])]:
            dest = out/(cell+'-'+label)
            run('live', '--resume', source, '--curriculum', full, '--out', dest, '--updates', 64,
                '--prompt', 'A', *flags)
            paths.append(dest/'latest.ckpt')
        reference = checkpoint(paths[0])
        for path in paths[1:]:
            actual = checkpoint(path)
            assert reference[1:3] == actual[1:3], 'Disabled guidance changed buffers or replay history'
            for i in set(range(32)) - {15, 17}:
                assert reference[0][i] == actual[0][i], (cell, i)
            assert paths[0].read_bytes()[256:288] == path.read_bytes()[256:288]
            policy = teaching(path)
            assert policy[7:11] == (7, 3, 0, 0)
            report = json.loads((path.parent/'session.json').read_text())
            assert report['teacher_extra_gpu_bytes'] == 0 and report['session_teacher_pairs'] == 0
        assert source.read_bytes() == raw_before
        controls.append(dict(cell=cell, zero_and_off_exact=True, control_sha256=reference[3]))
        if cell in ('lif', 'associative'):
            first = out/(cell+'-taught')
            run('live', '--resume', source, '--curriculum', full, '--out', first,
                '--updates', 64, '--teacher-bundle', active, '--prompt', 'A')
            split = out/(cell+'-split')
            run('live', '--resume', source, '--curriculum', full, '--out', split,
                '--updates', 29, '--teacher-bundle', active, '--prompt', 'A')
            run('live', '--resume', split/'latest.ckpt', '--curriculum', full, '--out', split,
                '--updates', 64, '--teacher-bundle', active, '--prompt', 'A')
            assert (first/'latest.ckpt').read_bytes() == (split/'latest.ckpt').read_bytes()
            assert checkpoint(first/'latest.ckpt')[1] == reference[1], 'Teachers changed source/replay selection'
            m, e, *_ = checkpoint(first/'latest.ckpt')
            policy = teaching(first/'latest.ckpt')
            assert policy[9] == e[20]+e[25]-3 and policy[10] == e[21]+e[26]-policy[15]
            assert inspect(first/'latest.ckpt')['teaching']['updates'] == policy[9]
            assert m[17] == 6 and policy[9] > 0
            restarts.append(dict(cell=cell, sha256=checkpoint(first/'latest.ckpt')[3], updates=policy[9], pairs=policy[10]))
    source = out/'associative-taught/latest.ckpt'
    before = source.read_bytes()
    policy = teaching(source)
    badout = out/'rejected'
    def resume(*flags, reject, bundle=active):
        run('live', '--resume', source, '--curriculum', full, '--extend-curriculum', out/'extended.sg',
            '--out', badout, '--updates', 80, '--prompt', 'A',
            *(['--teacher-bundle', bundle] if bundle else []), *flags, reject=reject)
        assert not badout.exists() and source.read_bytes() == before
    resume(reject='requires --teacher-bundle', bundle=None)
    resume('--teacher-memory-mib', 0, reject='memory budget exhausted')
    resume('--teaching', 'maybe', reject='must be on or off')
    resume(reject='identical teacher bundle', bundle=single)
    # Valid bundles can still be ineligible: different selected bytes, or an
    # excerpt ending inside a selected document instead of at its boundary.
    for label, content, rejection in [('unselected', b'This source was not selected.', 'not an exact prefix'),
                                      ('partial', (out/'1.dat').read_bytes()[:-2], 'document boundaries')]:
        data = out/(label+'.dat'); data.write_bytes(content)
        package = f.pack(label+'-bundle', source=data.name)
        resume(reject=rejection, bundle=package)
    for fault in ['weights', 'source', 'trailing', 'missing']:
        dest = out/('bad-'+fault); shutil.copytree(active, dest)
        if fault == 'weights':
            path = dest/'teacher-0.ckpt'; raw = bytearray(path.read_bytes()); raw[300] ^= 1; path.write_bytes(raw)
        elif fault == 'source':
            path = dest/'source.dat'; path.write_bytes(path.read_bytes()+b' ')
        elif fault == 'trailing':
            path = dest/'teachers.sg'; path.write_bytes(path.read_bytes()+b'extra\n')
        else:
            (dest/'teacher-0.ckpt').unlink()
        expected = dict(weights='snapshot file changed', source='source identity', trailing='Unexpected teacher', missing='Cannot open teacher')[fault]
        resume(reject=expected, bundle=dest)
    # A validly checksummed malformed teacher policy must still fail loading.
    meta = checkpoint(source)[0]
    offset = 288 + 12*meta[14] + 4*meta[18] + 8*meta[31]
    assert bytes(rehash(bytearray(before))) == before
    for label, word, value in [('counter', 9, 1000000), ('scope', 14, 2), ('temperature', 11, 0),
                                ('reserved-float-bits', 11, policy[11]+2**32), ('dimensions', 19, 1),
                                ('duplicate', 24, policy[16])]:
        raw = bytearray(before); struct.pack_into('<Q', raw, offset+8*word, value)
        path = out/(label+'.ckpt'); path.write_bytes(rehash(raw))
        run('sample', '--checkpoint', path, '--tokens', 2, reject='ERROR:')
    for label, raw in [('short', before[:-1]), ('long', before+b'\x00')]:
        path = out/(label+'.ckpt'); path.write_bytes(raw)
        run('sample', '--checkpoint', path, '--tokens', 2, reject='ERROR:')
    run('sample', '--checkpoint', source, '--tokens', 8, '--prompt', 'A', '--graph')
    run('evaluate', '--checkpoint', source, '--data', f.val, '--batch', 2, '--context', 16, '--batches', 2)
    probes = out/'toy.sgprobe'
    probes.write_text('SGPROBE1 2\n"a" "pair" "location" 0 "The key is in the box." " Where is the key? " "box" "bag"\n'
                      '"b" "pair" "location" 1 "The key is in the bag." " Where is the key? " "box" "bag"\n')
    run('language-probes', '--checkpoint', source, '--probes', probes, '--output', out/'probes.json')
    off = out/'mode-off'
    run('live', '--resume', source, '--curriculum', full, '--extend-curriculum', out/'extended.sg',
        '--out', off, '--updates', 76, '--prompt', 'A', '--teacher-bundle', active, '--teaching', 'off', '--teacher-memory-mib', 0)
    assert teaching(off/'latest.ckpt')[9:11] == policy[9:11] and not teaching(off/'latest.ckpt')[2]
    run('live', '--resume', off/'latest.ckpt', '--curriculum', out/'extended.sg', '--out', off,
        '--updates', 84, '--prompt', 'A', '--teacher-bundle', active)
    assert teaching(off/'latest.ckpt')[9:11] == policy[9:11] and not teaching(off/'latest.ckpt')[2]
    run('live', '--resume', off/'latest.ckpt', '--curriculum', out/'extended.sg', '--out', off,
        '--updates', 100, '--prompt', 'A', '--teacher-bundle', active, '--teaching', 'on')
    assert teaching(off/'latest.ckpt')[9] > policy[9]
    run('live', '--curriculum', full, '--out', out/'one-teacher', '--cell', 'lif', '--updates', 5,
        '--teacher-bundle', single, *common)
    # Existing registered archives can be evaluated and inherited without
    # loading their teachers. A child starts with its own empty live history.
    population = out/'inheritance'
    for i in range(2):
        run('population-add', '--population', population, '--id', f'ancestor-{i}',
            '--checkpoint', out/'lif-taught/latest.ckpt', '--data', f.val, '--max-score', 20,
            *(['--batch', 2, '--context', 16, '--batches', 2] if i == 0 else []))
    run('evolve', '--population', population, '--data', f.val, '--round', 'child', '--children', 1)
    newborn = population/'child-child-0/latest.ckpt'
    assert newborn.exists() and checkpoint(newborn)[0][17] == 0 and teaching(newborn) is None
    run('teacher-pack', '--teacher-a', f.parents[0], '--teacher-b', f.parents[0], '--data', out/'1.dat',
        '--out', badout, reject='must be distinct')
    run('teacher-pack', '--teacher-a', f.parents[0], '--data', out/'1.dat', '--out', active, reject='already exists')
    assert not badout.exists() and source.read_bytes() == before and fingerprints(active) == bundle_before
    report = dict(passed=True, executable_sha256=f.executable_sha256,
                  native_commands=len(f.calls), disabled_controls=controls, exact_restarts=restarts,
                  one_and_two_teachers=True, immutable_bundle=True, invalid_resume_writes_nothing=True,
                  checksum_valid_invalid_policy_rejected=True, memory_limit_enforced=True,
                  modes_persist_counters=True, append_only_curriculum_extension=True,
                  all_read_only_commands_accept_v6=True, source_replay_selection_unchanged=True,
                  unselected_and_partial_source_rejected=True, v6_inheritance_starts_empty_child_history=True)
    (out/'result.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--exe', type=Path, default=Path('build/synapticgenesis.exe'))
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    check(args.exe, args.out)
