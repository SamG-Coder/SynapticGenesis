"""Population learning, canonical state, lifespan and writer ownership checks."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess


def check(exe, out):
    exe = exe.resolve()
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    train, later, val = out / 'train.dat', out / 'later.dat', out / 'validation.dat'
    train.write_bytes(b'A child sees the bird in a tree. The bird sings. ' * 12)
    later.write_bytes(train.read_bytes() + b'\x1e' + b'A seed needs water and sunlight. The child waters the garden. ' * 12)
    val.write_bytes(b'The child rests near a tree. A bird sees the garden. ' * 12)
    schedule = out / 'curriculum.sg'
    schedule.write_bytes(b'SGCURRICULUM1\n3 "train.dat" 1\n8 "later.dat" 0.5\n')
    population = out / 'population'
    calls = 0

    def run(*args, reject=None):
        nonlocal calls
        calls += 1
        p = subprocess.run([str(exe), *map(str, args)], capture_output=True, text=True)
        (out / f'command-{calls:02d}.log').write_text(p.stdout + p.stderr, encoding='utf-8')
        if reject:
            assert p.returncode != 0 and reject in p.stderr, (args, p.stdout, p.stderr)
        elif p.returncode:
            raise RuntimeError(f'{args}: {p.stdout}\n{p.stderr}')

    for index in range(2):
        source = out / f'founder-{index}'
        run('train', '--data', train, '--validation', val, '--out', source,
            '--channels', 8, '--hidden', 16, '--layers', 2, '--batch', 2, '--context', 16,
            '--steps', 100, '--warmup', 0, '--lr', (.00073571 if index == 0 else .00041239), '--seed', 11 + index,
            '--eval-every', 100, '--eval-batches', 2)
        fixed = ['--batch', 2, '--context', 16, '--batches', 2] if index == 0 else []
        run('population-add', '--population', population, '--id', f'founder-{index}',
            '--checkpoint', source / 'latest.ckpt', '--data', val, '--max-score', 20,
            '--lifespan', 2, '--growth-chance', 1, '--setting-mutation-chance', 1, *fixed)
        run('population-live', '--population', population, '--id', f'founder-{index}',
            '--curriculum', schedule, '--validation', val, '--updates', 8, '--chunk', 8,
            '--speak-every', 0, '--eval-batches', 2)
    run('evolve', '--population', population, '--data', val, '--round', 'birth', '--children', 1, '--seed', 7)
    child = population / 'birth-child-0'
    birth = json.loads((population / 'birth.json').read_text())['children'][0]
    assert birth['grew'] and birth['lifespan_ticks'] == 2
    for label in ('a', 'b'):
        parent = population / birth[f'parent_{label}'] / 'latest.ckpt'
        hp = struct.unpack_from('<8f', parent.read_bytes(), 256)
        assert hp[0] == hp[7] * .5
        assert abs(birth[f'parent_{label}_base_learning_rate'] - hp[7]) < 1e-12
    before = (child / 'latest.ckpt').read_bytes()
    initial_hash = hashlib.sha256((child / 'initial.ckpt').read_bytes()).hexdigest()
    lineage = (child / 'member.sg').read_bytes()
    clock = (population / 'population.sg').read_bytes()
    base_rate = struct.unpack_from('<8f', before, 256)[0]
    run('population-live', '--population', population, '--id', child.name, '--curriculum', schedule,
        '--validation', val, '--updates', 3, '--chunk', 8, '--replay', 'reservoir', '--replay-capacity', 16,
        '--replay-every', 2, '--graph', '--speak-every', 2, '--tokens', 7, '--prompt', 'A',
        '--consolidation', 'si', '--si-strength', .001, '--eval-batches', 2)
    first = (child / 'latest.ckpt').read_bytes()
    assert first != before and struct.unpack_from('<32Q', first)[24] == 3
    assert struct.unpack_from('<8f', first, 256)[7] == base_rate, 'Inherited rate was rounded or replaced'
    assert not (child / 'live/latest.ckpt').exists(), 'Two competing latest checkpoint authorities'
    assert (child / 'member.sg').read_bytes() == lineage and (population / 'population.sg').read_bytes() == clock
    assert hashlib.sha256((child / 'initial.ckpt').read_bytes()).hexdigest() == initial_hash
    run('population-live', '--population', population, '--id', child.name, '--curriculum', schedule,
        '--validation', val, '--updates', 8, '--prompt', 'A', '--eval-batches', 2)
    learned = (child / 'latest.ckpt').read_bytes()
    meta = struct.unpack_from('<32Q', learned)
    extras = struct.unpack_from(f'<{meta[31]}Q', learned, 288 + 12 * meta[14] + 4 * meta[18])
    assert meta[24] == 8 and extras[5] == 8 and extras[6] == 4 and extras[15] == 1
    assert struct.unpack_from('<8f', learned, 256)[0] == base_rate * .5
    events = [json.loads(s) for s in (child / 'live/population.jsonl').read_text().splitlines()]
    assert len(events) == 2 and events[1]['starting_online_updates'] == 3
    assert all(e['age'] == 0 and e['generation'] == 1 and not e['automatic_fitness_promotion'] for e in events)
    run('population-live', '--population', population, '--id', child.name, '--curriculum', schedule,
        '--validation', train, reject='Population evaluation corpus changed')
    run('population-live', '--population', population, '--id', child.name, '--out', out / 'override',
        reject='Population owns out')
    # A separate process cannot age, replace or learn a member while owned.
    lock = population / '.population-lock'
    lock.mkdir()
    run('population-live', '--population', population, '--id', child.name, reject='Population is locked')
    run('evolve', '--population', population, '--data', val, '--round', 'locked', reject='Population is locked')
    assert lock.is_dir(), 'Failed acquisition removed another writer lock'
    lock.rmdir()
    assert (child / 'latest.ckpt').read_bytes() == learned
    assert (population / 'population.sg').read_bytes() == clock
    run('evolve', '--population', population, '--data', val, '--round', 'parents-die', '--food-mib', 0)
    round2 = json.loads((population / 'parents-die.json').read_text())
    selected = next(m for m in round2['members'] if m['id'] == child.name)
    assert selected['alive'] and selected['generation'] == 1
    # Exact checkpoint hash in parent selection proves it read the published learning result.
    assert str(selected['checkpoint_payload_hash']) == str(meta[15])
    run('evolve', '--population', population, '--data', val, '--round', 'child-dies', '--food-mib', 0)
    round3 = json.loads((population / 'child-dies.json').read_text())
    deceased = next(m for m in round3['members'] if m['id'] == child.name)
    assert not deceased['alive'] and not deceased['eligible'] and deceased['death_reason'] == 'old_age'
    logs = (child / 'live/metrics.jsonl').read_bytes()
    run('population-live', '--population', population, '--id', child.name, '--curriculum', schedule,
        '--validation', val, '--updates', 9, '--prompt', 'A', reject='died of old age')
    assert (child / 'latest.ckpt').read_bytes() == learned and (child / 'live/metrics.jsonl').read_bytes() == logs
    assert (child / 'member.sg').read_bytes() == lineage and not lock.exists()
    run('sample', '--checkpoint', child / 'latest.ckpt', '--tokens', 8, '--prompt', 'A')
    report = {'passed': True, 'native_commands': calls, 'canonical_checkpoint_updated': True,
              'evolution_reads_exact_learned_payload': True, 'inherited_rate_preserved_exactly': True,
              'mature_parent_base_rates_inherited': True,
              'live_curriculum_and_si_resume': True, 'learning_does_not_reset_age': True,
              'lineage_and_birth_checkpoint_preserved': True, 'wrong_evaluation_corpus_rejected': True,
              'writer_lock_blocks_learning_and_aging': True, 'dead_member_cannot_resume_learning': True,
              'dead_archive_remains_readable': True, 'learned_checkpoint_sha256': hashlib.sha256(learned).hexdigest()}
    (out / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exe', type=Path, default=Path('build/synapticgenesis.exe'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    check(args.exe, args.out)
