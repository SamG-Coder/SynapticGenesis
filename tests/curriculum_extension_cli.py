"""Append unplanned lessons without resetting live learning or population age.

Compare against the same complete schedule declared from birth, including
weighted replay, SI and graph speech. All model computation is native CUDA.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from curriculum_cli import payload


def check(exe, out, replay='reservoir'):
    exe, out = exe.resolve(), out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    docs = [b'A child sees a bird in a tree. The bird sings to the child. ' * 3,
            b'The key is in the box. The cup is in the bag.\nAnswer: box.',
            b'The hat is in the car. The mug is in the bed.\nAnswer: car.',
            b'The pen is in the bag. The toy is in the box.\nAnswer: bag.']
    for i in range(4):
        (out / f'{i}.dat').write_bytes(b'\x1e'.join(docs[:i+1]))
    rows = ['3 "0.dat" 1 all 1', '6 "1.dat" 0.5 new 64',
            '10 "2.dat" 0.25 new 16', '14 "3.dat" 0.25 new 32']
    schedules = []
    for n in (2, 3, 4):
        path = out / f'{n}-stages.sg'
        path.write_text('SGCURRICULUM3\n' + '\n'.join(rows[:n]) + '\n', encoding='utf-8')
        schedules.append(path)
    old, new, newest = schedules
    val = out / 'validation.dat'
    val.write_bytes(b'A little bird rests near a quiet garden. The seed needs water. ' * 10)
    calls, cases, max_error = 0, 0, 0

    def run(*args, reject=None):
        nonlocal calls
        calls += 1
        p = subprocess.run([str(exe), *map(str, args)], capture_output=True)
        (out / f'command-{calls:03d}.log').write_bytes(p.stdout + p.stderr)
        if reject:
            assert p.returncode and reject.encode() in p.stderr, (args, p.stdout, p.stderr)
        elif p.returncode:
            raise RuntimeError(f'{args}: {p.stdout}\n{p.stderr}')

    def compare(expected, actual):
        nonlocal max_error
        assert expected[0][:15] == actual[0][:15] and expected[0][16:] == actual[0][16:]
        assert expected[1:3] == actual[1:3]
        assert len(expected[3]) == len(actual[3])
        error = max(abs(a-b) for a, b in zip(expected[3], actual[3]))
        assert error < 3e-5, error
        max_error = max(max_error, error)

    def speech(folder):
        return [block for block in (folder / 'transcript.txt').read_bytes().split(b'\n[')
                if block.startswith(b'online update')]

    common = ['--channels', 8, '--hidden', 16, '--layers', 2, '--chunk', 16,
              '--replay', replay, '--replay-capacity', 32, '--replay-every', 2,
              '--lr', .0008, '--graph', '--seed', 1337, '--speak-every', 2, '--tokens', 7,
              '--prompt', 'A', '--log-every', 1, '--save-every', 100]
    for cell in ('lif', 'alif', 'trace', 'gated', 'selective', 'associative'):
        for si in (False, True):
            policy = ['--cell', cell] + (['--consolidation', 'si', '--si-strength', .02] if si else [])
            full = out / f'{cell}-{int(si)}-full'
            run('live', '--curriculum', new, '--out', full, *common, *policy)
            expected = payload(full / 'latest.ckpt')
            for split in (2, 3, 4, 6):
                part = out / f'{cell}-{int(si)}-{split}'
                run('live', '--curriculum', old, '--out', part, '--updates', split, *common, *policy)
                before = payload(part / 'latest.ckpt')
                initial = (part / 'initial.ckpt').read_bytes()
                archives = {p.name: p.read_bytes() for p in part.glob('stage-*.ckpt')}
                run('live', '--resume', part / 'latest.ckpt', '--curriculum', old,
                    '--extend-curriculum', new, '--out', part, '--updates', 8, '--prompt', 'A')
                assert json.loads((part / 'session.json').read_text())['curriculum_extended']
                extended = [json.loads(s) for s in (part / 'metrics.jsonl').read_text().splitlines()
                            if '"event":"curriculum_extension"' in s]
                assert len(extended) == 1
                event = extended[0]
                assert event['online_update'] == before[0][24]
                assert event['global_update'] == before[0][7]
                assert event['document'] == before[0][19] and event['byte_offset'] == before[0][20]
                assert event['generated_bytes'] == before[0][30]
                assert event['replay_updates'] == before[2][6]
                assert event['replay_windows_preserved'] == split
                assert event['previous_stages'] == 2 and event['stages'] == 3
                assert str(event['previous_curriculum_hash']) == str(before[2][14])
                assert (part / 'initial.ckpt').read_bytes() == initial
                assert all((part / k).read_bytes() == v for k, v in archives.items())
                run('live', '--resume', part / 'latest.ckpt', '--curriculum', new,
                    '--out', part, '--prompt', 'A')
                compare(expected, payload(part / 'latest.ckpt'))
                assert speech(full) == speech(part)
                assert not json.loads((part / 'session.json').read_text())['curriculum_extended']
                cases += 1

    # Repeated extension is still equivalent to a schedule declared from birth.
    selected = out / 'gated-1-6'
    full = out / 'gated-repeated-full'
    run('live', '--curriculum', newest, '--out', full, *common,
        '--cell', 'gated', '--consolidation', 'si', '--si-strength', .02)
    run('live', '--resume', selected / 'latest.ckpt', '--curriculum', new,
        '--extend-curriculum', newest, '--out', selected, '--prompt', 'A')
    compare(payload(full / 'latest.ckpt'), payload(selected / 'latest.ckpt'))
    assert speech(full) == speech(selected)

    # Invalid extensions must leave checkpoints and logs untouched, including
    # rejecting a changed old schedule even if the new prefix matches it.
    source = out / 'gated-1-4/stage-2.ckpt'
    source_bytes = source.read_bytes()
    rejected = out / 'rejected'
    def reject_extension(previous, following, message, *extra):
        run('live', '--resume', source, '--curriculum', previous, '--extend-curriculum', following,
            '--out', rejected, '--prompt', 'A', *extra, reject=message)
        assert source.read_bytes() == source_bytes and not rejected.exists()

    # stage-2 in this split has already adopted the extended schedule identity.
    reject_extension(old, newest, 'identical curriculum')
    for old_row, replacement in [
            (rows[0], '4 "0.dat" 1 all 1'),
            (rows[1], '6 "1.dat" 0.4 new 64'),
            (rows[1], '6 "1.dat" 0.5 all 64'),
            (rows[1], '6 "1.dat" 0.5 new 32')]:
        bad = out / 'changed.sg'
        bad.write_text(newest.read_text().replace(old_row, replacement), encoding='utf-8')
        reject_extension(new, bad, 'preserve every existing stage')
    reject_extension(new, new, 'append at least one stage')
    reject_extension(new, old, 'append at least one stage')
    (out / 'changed-old.sg').write_text(new.read_text() + '\n', encoding='utf-8')
    reject_extension(out / 'changed-old.sg', newest, 'identical curriculum')
    # New corpus bytes are checked even when an old stage is still in progress.
    original = (out / '3.dat').read_bytes()
    (out / '3.dat').write_bytes(b'changed' + original)
    reject_extension(new, newest, 'append complete documents')
    (out / '3.dat').write_bytes((out / '2.dat').read_bytes() + b'\x1eUnannotated lesson.')
    reject_extension(new, newest, 'Answer field')
    (out / '3.dat').write_bytes(original)
    (out / 'heldout.dat').write_bytes(docs[3])
    reject_extension(new, newest, 'held-out document', '--validation', out / 'heldout.dat')
    reject_extension(new, newest, 'omit --data', '--data', out / '0.dat')
    run('live', '--curriculum', new, '--extend-curriculum', newest, '--out', rejected,
        reject='requires --resume')
    run('live', '--resume', source, '--extend-curriculum', newest, '--out', rejected,
        reject='requires --curriculum')
    # Ordinary resume continues to reject unannounced schedule changes.
    run('live', '--resume', source, '--curriculum', newest, '--out', rejected, '--prompt', 'A',
        reject='identical curriculum')
    assert not rejected.exists() and source.read_bytes() == source_bytes

    # Register an existing live individual, age it, extend its lessons, then
    # verify that extension cannot resurrect it after the next aging round.
    pop = out / 'population'
    run('population-add', '--population', pop, '--id', 'founder', '--checkpoint', source,
        '--data', val, '--batch', 2, '--context', 16, '--batches', 2, '--lifespan', 2, '--max-score', 20)
    run('evolve', '--population', pop, '--data', val, '--round', 'age-one', '--food-mib', 0)
    member = pop / 'founder'
    lineage = (member / 'member.sg').read_bytes()
    clock = (pop / 'population.sg').read_bytes()
    run('population-live', '--population', pop, '--id', 'founder', '--curriculum', new,
        '--extend-curriculum', newest, '--validation', val, '--prompt', 'A', '--eval-batches', 2)
    assert (member / 'member.sg').read_bytes() == lineage and (pop / 'population.sg').read_bytes() == clock
    assert not (member / 'live/latest.ckpt').exists()
    journal = json.loads((member / 'live/population.jsonl').read_text().splitlines()[-1])
    assert journal['age'] == 1 and journal['born_tick'] == 0 and journal['saved_online_updates'] == 14
    learned = (member / 'latest.ckpt').read_bytes()
    run('evolve', '--population', pop, '--data', val, '--round', 'age-two', '--food-mib', 0)
    deceased = json.loads((pop / 'age-two.json').read_text())['members'][0]
    assert not deceased['alive'] and not deceased['eligible'] and deceased['death_reason'] == 'old_age'
    run('population-live', '--population', pop, '--id', 'founder', '--curriculum', new,
        '--extend-curriculum', newest, '--validation', val, '--prompt', 'A', reject='died of old age')
    assert (member / 'latest.ckpt').read_bytes() == learned
    run('sample', '--checkpoint', member / 'latest.ckpt', '--tokens', 8, '--prompt', 'A')
    result = dict(passed=True, replay_policy=replay, native_commands=calls, extension_cases=cases,
                  cells=['lif', 'alif', 'trace', 'gated', 'selective', 'associative'], optional_si=True, graph_speech_identical=True,
                  final_payload_max_error=max_error, repeated_extension_matches_full_schedule=True,
                  prefix_policy_and_all_source_editions_preserved=True,
                  invalid_extension_writes_no_run=True, heldout_new_document_rejected=True,
                  initial_and_completed_stage_archives_preserved=True,
                  population_age_and_lineage_preserved=True, dead_member_extension_rejected=True,
                  deceased_archive_readable=True, reserved_language_test_unused=True,
                  source_sha256=hashlib.sha256(source_bytes).hexdigest())
    (out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--exe', type=Path, default=Path('build/synapticgenesis.exe'))
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--replay', choices=['reservoir','stage'], default='reservoir')
    a = p.parse_args()
    check(a.exe, a.out, a.replay)
