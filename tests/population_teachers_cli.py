"""Registered parent teaching, immutable ancestry, lifespan and ownership checks."""
import argparse
import json
from pathlib import Path

from teacher_replay_cli import Fixture, checkpoint, teaching, fingerprints, inspect


def check(exe, out):
    f = Fixture(exe, out, compatible=True)
    run, out = f.run, f.out
    population, bundle = out/'population', out/'parents'
    for i, source in enumerate(f.parents):
        run('population-add', '--population', population, '--id', f'parent-{i}', '--checkpoint', source,
            '--data', f.val, '--max-score', 20, '--lifespan', 2, '--growth-chance', 1,
            '--setting-mutation-chance', 1, *(['--batch', 2, '--context', 16, '--batches', 2] if i == 0 else []))
    run('evolve', '--population', population, '--data', f.val, '--round', 'birth', '--children', 2, '--seed', 7)
    births = json.loads((population/'birth.json').read_text())['children']
    assert len(births) == 2 and all(b['grew'] for b in births)
    run('teacher-pack', '--population', population, '--teacher-a', 'parent-0', '--teacher-b', 'parent-1',
        '--data', out/'2.dat', '--out', bundle, '--strength', .7)
    frozen = fingerprints(bundle)
    schedule = out/'full.sg'
    initial = ['--chunk', 8, '--replay', 'stage', '--replay-capacity', 12, '--replay-every', 2,
               '--graph', '--speak-every', 2, '--tokens', 7, '--prompt', 'A', '--consolidation', 'si', '--si-strength', .02]
    child = population/'birth-child-0'
    child_before = fingerprints(child)
    parent_files = [fingerprints(population/f'parent-{i}') for i in range(2)]
    run('population-live', '--population', population, '--id', child.name, '--curriculum', schedule,
        '--validation', f.val, '--updates', 15, '--teacher-bundle', bundle, *initial)
    policy = teaching(child/'latest.ckpt')
    assert policy[14] == 1 and policy[9] > 0
    diagnostic = inspect(child/'latest.ckpt')
    assert diagnostic['teaching']['updates'] == policy[9]
    assert diagnostic['synaptic_memory']['boundaries'] == checkpoint(child/'latest.ckpt')[1][12]
    for file in ['initial.ckpt', 'member.sg']:
        assert fingerprints(child)[file] == child_before[file]
    assert all(fingerprints(population/f'parent-{i}') == parent_files[i] for i in range(2))
    # A teacher can itself learn; attached snapshots remain frozen. A different
    # child cannot newly attach that now-outdated snapshot as its current parent.
    run('population-live', '--population', population, '--id', 'parent-0', '--curriculum', schedule,
        '--validation', f.val, '--updates', 8, *initial)
    second = population/'birth-child-1'
    untouched = fingerprints(second)
    run('population-live', '--population', population, '--id', second.name, '--curriculum', schedule,
        '--validation', f.val, '--updates', 15, '--teacher-bundle', bundle, *initial,
        reject='current registered checkpoint')
    assert fingerprints(second) == untouched
    run('population-live', '--population', population, '--id', child.name, '--curriculum', schedule,
        '--validation', f.val, '--updates', 28, '--teacher-bundle', bundle, '--prompt', 'A')
    assert teaching(child/'latest.ckpt')[9] > policy[9] and fingerprints(bundle) == frozen
    assert inspect(child/'latest.ckpt')['synaptic_memory']['boundaries'] > 0
    saved = fingerprints(child)
    run('live', '--resume', child/'latest.ckpt', '--curriculum', schedule, '--updates', 32,
        '--out', out/'standalone', '--prompt', 'A', '--teacher-bundle', bundle,
        reject='requires population-live')
    assert not (out/'standalone').exists()
    standalone = f.pack('standalone-bundle')
    run('population-live', '--population', population, '--id', second.name, '--curriculum', schedule,
        '--validation', f.val, '--updates', 15, '--teacher-bundle', standalone, *initial,
        reject='registered teacher origins')
    lock = population/'.population-lock'
    lock.mkdir()
    run('teacher-pack', '--population', population, '--teacher-a', 'parent-0', '--data', out/'2.dat',
        '--out', out/'locked-bundle', reject='Population is locked')
    run('population-live', '--population', population, '--id', child.name,
        reject='Population is locked')
    assert lock.is_dir() and not (out/'locked-bundle').exists()
    lock.rmdir()
    assert fingerprints(child) == saved and fingerprints(second) == untouched
    # At tick two parents die, while their children are age one of two.
    run('evolve', '--population', population, '--data', f.val, '--round', 'parents-die', '--food-mib', 0)
    status = json.loads((population/'parents-die.json').read_text())
    selected = next(m for m in status['members'] if m['id'] == child.name)
    assert selected['alive'] and str(selected['checkpoint_payload_hash']) == str(checkpoint(child/'latest.ckpt')[0][15])
    run('teacher-pack', '--population', population, '--teacher-a', 'parent-0', '--data', out/'2.dat',
        '--out', out/'dead-bundle', reject='died of old age')
    run('population-live', '--population', population, '--id', child.name, '--curriculum', schedule,
        '--validation', f.val, '--updates', 40, '--teacher-bundle', bundle, '--prompt', 'A',
        reject='Teacher died of old age')
    assert fingerprints(child) == saved and not (out/'dead-bundle').exists()
    policy = teaching(child/'latest.ckpt')
    run('population-live', '--population', population, '--id', child.name, '--curriculum', schedule,
        '--validation', f.val, '--updates', 40, '--teacher-bundle', bundle, '--prompt', 'A',
        '--teaching', 'off', '--teacher-memory-mib', 0)
    assert teaching(child/'latest.ckpt')[9:11] == policy[9:11]
    run('population-live', '--population', population, '--id', child.name, '--curriculum', schedule,
        '--validation', f.val, '--updates', 48, '--teacher-bundle', bundle, '--prompt', 'A')
    assert teaching(child/'latest.ckpt')[2] == 0 and teaching(child/'latest.ckpt')[9:11] == policy[9:11]
    current = fingerprints(child)
    run('population-live', '--population', population, '--id', child.name, '--curriculum', schedule,
        '--validation', f.val, '--updates', 52, '--teacher-bundle', bundle, '--prompt', 'A',
        '--teaching', 'on', reject='Teacher died of old age')
    assert fingerprints(child) == current
    run('evolve', '--population', population, '--data', f.val, '--round', 'children-die', '--food-mib', 0)
    dead = next(m for m in json.loads((population/'children-die.json').read_text())['members'] if m['id'] == child.name)
    assert not dead['alive'] and not dead['eligible'] and dead['death_reason'] == 'old_age'
    run('population-live', '--population', population, '--id', child.name, '--curriculum', schedule,
        '--validation', f.val, '--updates', 52, '--teacher-bundle', bundle, '--prompt', 'A',
        '--teaching', 'off', reject='Member died of old age')
    run('sample', '--checkpoint', child/'latest.ckpt', '--tokens', 8, '--prompt', 'A', '--graph')
    run('evaluate', '--checkpoint', child/'latest.ckpt', '--data', f.val, '--batch', 2, '--context', 16, '--batches', 2)
    assert fingerprints(child) == current and fingerprints(bundle) == frozen and not lock.exists()
    result = dict(passed=True, executable_sha256=f.executable_sha256,
                  native_commands=len(f.calls), two_registered_parents=True,
                  inherited_width_growth=True, lineage_and_initial_checkpoint_unchanged=True,
                  frozen_snapshots_survive_parent_learning=True, stale_new_attachment_rejected=True,
                  scope_and_lock_enforced=True, rejected_actions_change_no_member_files=True,
                  dead_parent_cannot_teach=True, alive_child_can_disable_teaching=True,
                  dead_child_cannot_learn=True, archived_v6_readable=True,
                  evolution_evaluates_exact_learned_v6=True, teaching_counter_history_preserved=True)
    (out/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--exe', type=Path, default=Path('build/synapticgenesis.exe'))
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    check(args.exe, args.out)
