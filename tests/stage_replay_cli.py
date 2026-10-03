"""Native stage-memory admission, read-only tools and corrupt-policy rejection."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys

from curriculum_cli import payload


def check(exe, out):
    exe, out = exe.resolve(), out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    docs = [b'A child reads a book under a tree. ' * 12,
            b'The key is in the box.\nAnswer: box.',
            b'The hat is in the bag.\nAnswer: bag.']
    for n in range(1, 4):
        (out / f'{n}.dat').write_bytes(b'\x1e'.join(docs[:n]))
    short, full = out / 'short.sg', out / 'full.sg'
    rows = ['20 "1.dat" 1 all 1', '40 "2.dat" .5 new 64', '60 "3.dat" .25 all 64']
    short.write_text('SGCURRICULUM3\n' + '\n'.join(rows[:2]) + '\n')
    full.write_text('SGCURRICULUM3\n' + '\n'.join(rows) + '\n')
    validation = out / 'validation.dat'
    validation.write_bytes(b'The flowers grow beside a quiet path. ' * 8)
    calls = 0

    def run(*args, reject=None):
        nonlocal calls
        calls += 1
        result = subprocess.run([str(exe), *map(str, args)], capture_output=True, text=True)
        (out / f'command-{calls:02d}.log').write_text(result.stdout + result.stderr)
        if reject:
            assert result.returncode and reject in result.stderr, result.stderr
        else:
            assert result.returncode == 0, result.stderr

    common = ['--cell', 'selective', '--channels', 8, '--hidden', 16, '--layers', 2,
              '--chunk', 16, '--speak-every', 0, '--replay', 'stage', '--replay-capacity', 6]
    dest = out / 'live'
    run('live', '--curriculum', full, '--out', dest, *common, '--validation', validation, '--eval-batches', 2)
    source = dest / 'latest.ckpt'
    before = source.read_bytes()
    meta, hp, extra, _ = payload(source)
    assert meta[17] == 5 and extra[1] == 3 and extra[16] == 3
    assert len(extra) == 17 + 5*3 + 3*6
    assert [extra[17+5*g+2] for g in range(3)] == [2, 2, 2]
    assert sum(extra[17+5*g+1] for g in range(3)) == 60
    session = json.loads((dest / 'session.json').read_text())
    assert session['replay_state_bytes'] == 8*len(extra)
    assert sum(g['replay_updates'] for g in session['replay_groups']) == session['replay_updates']
    assert sum(g['replay_pairs'] for g in session['replay_groups']) == session['replay_pairs']
    assert session['replay_groups'][0]['seen_windows'] > 20, 'All-scope stage did not revisit the earlier source'
    run('sample', '--checkpoint', source, '--tokens', 8, '--prompt', 'A', '--graph')
    run('evaluate', '--checkpoint', source, '--data', validation, '--batch', 2, '--context', 16, '--batches', 2)
    sys.path.insert(0, str(Path('scripts').resolve()))
    from inspect_dynamics import inspect
    assert inspect(source)['curriculum']['stage'] == 3
    assert source.read_bytes() == before
    badout = out / 'rejected'
    run('live', '--data', out / '1.dat', '--out', badout, *common, reject='requires --curriculum')
    low = common.copy(); low[-1] = 2
    run('live', '--curriculum', full, '--out', badout, *low, reject='capacity must cover')
    assert not badout.exists()
    bounded = out / 'bounded'
    run('live', '--curriculum', short, '--out', bounded, *low)
    checkpoint = bounded / 'latest.ckpt'
    unchanged = checkpoint.read_bytes()
    run('live', '--resume', checkpoint, '--curriculum', short, '--extend-curriculum', full,
        '--out', badout, reject='capacity must cover')
    run('live', '--resume', source, '--curriculum', full, '--out', badout, '--replay', 'reservoir',
        reject='preserves replay')
    run('live', '--resume', source, '--out', badout, reject='requires --curriculum')
    assert checkpoint.read_bytes() == unchanged and not badout.exists()
    # Recompute the full checksum to prove semantic validation, not just byte corruption detection.
    def fnv(data, value=14695981039346656037):
        for x in data:
            value = ((value ^ x) * 1099511628211) & ((1 << 64)-1)
        return value
    offset = 288 + 12*meta[14] + 4*meta[18]
    original_header = bytearray(before[:256])
    struct.pack_into('<Q',original_header,15*8,0)
    check_hash = fnv(before[288:offset])
    check_hash = fnv(original_header,check_hash)
    check_hash = fnv(before[256:288],check_hash)
    check_hash = fnv(before[offset:],check_hash)
    assert check_hash == meta[15], 'Independent checksum does not match a valid native checkpoint'
    for fault, word, value in [('range', 17, 0), ('seen', 18, 123), ('layout', 16, 4097)]:
        raw = bytearray(before)
        struct.pack_into('<Q', raw, offset + 8*word, value)
        struct.pack_into('<Q', raw, 15*8, 0)
        h = fnv(raw[288:offset])
        h = fnv(raw[:256], h); h = fnv(raw[256:288], h); h = fnv(raw[offset:], h)
        struct.pack_into('<Q', raw, 15*8, h)
        corrupt = out / f'{fault}.ckpt'; corrupt.write_bytes(raw)
        run('sample', '--checkpoint', corrupt, '--tokens', 2, reject='stage replay')
    # Convert while still inside the first source stage and preserve its exact
    # reservoir. Continuing in that stage must match the legacy learner.
    legacy_args = common.copy(); legacy_args[legacy_args.index('stage')] = 'reservoir'
    ancestor = out/'ancestor'
    run('live','--curriculum',full,'--out',ancestor,'--updates',10,*legacy_args)
    origin = ancestor/'latest.ckpt'
    original = origin.read_bytes()
    for policy in ('reservoir','stage'):
        dest = out/f'conversion-{policy}'
        options = ['--replay','stage'] if policy=='stage' else []
        run('live','--resume',origin,'--curriculum',full,'--out',dest,'--updates',18,*options)
    old = payload(out/'conversion-reservoir/latest.ckpt')
    new = payload(out/'conversion-stage/latest.ckpt')
    conversion_error = max(abs(a-b) for a,b in zip(old[3],new[3]))
    assert old[1] == new[1] and conversion_error < 3e-5
    assert old[2][4:14] == new[2][4:14] and old[2][16:] == new[2][22:]
    assert original == origin.read_bytes()
    events = [json.loads(line) for line in (out/'conversion-stage/metrics.jsonl').read_text().splitlines()]
    assert events[0]['event'] == 'replay_policy_conversion' and events[0]['online_update']==10
    assert events[0]['replay_windows_preserved']==6
    run('live','--resume',ancestor/'stage-1.ckpt' if (ancestor/'stage-1.ckpt').exists() else origin,
        '--curriculum',full,'--out',out/'past-first','--updates',40)
    too_late = out/'past-first/latest.ckpt'
    run('live','--resume',too_late,'--curriculum',full,'--out',badout,'--replay','stage',
        reject='first-stage curriculum reservoir')
    assert not badout.exists()
    result = dict(passed=True, native_commands=calls, persistent_budget_slots=6,
                  source_groups_checked=True, returning_source_seen_count_checked=True,
                  replay_exposure_totals_checked=True, read_only_tools_checked=True,
                  invalid_policies_write_no_run=True, over_capacity_extension_transactional=True,
                  checksum_valid_invalid_group_payload_rejected=True,
                  first_stage_conversion_max_error=conversion_error,
                  first_stage_conversion_preserves_observed_history=True,
                  late_conversion_rejected=True,
                  checkpoint_sha256=hashlib.sha256(before).hexdigest())
    (out / 'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--exe', type=Path, default=Path('build/synapticgenesis.exe'))
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    check(a.exe, a.out)
