"""Exercise native births, scarcity, newborn gates, old age and restart persistence.

Python only launches the C++/CUDA commands and checks their artifacts. All model
learning, scoring, mutation, resource accounting and selection are native.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess


def check(exe, out, cell='lif'):
    exe = exe.resolve()
    out.mkdir(parents=True, exist_ok=False)
    train, val = out / 'train.dat', out / 'validation.dat'
    train.write_bytes(b'A child sees the bird. The bird rests in a tree. ' * 32)
    val.write_bytes(b'The child rests. A bird sees the tree. ' * 29)
    population = out / 'population'
    calls = 0

    def run(*args, reject=None):
        nonlocal calls
        calls += 1
        result = subprocess.run([str(exe), *map(str, args)], capture_output=True, text=True)
        (out / f'command-{calls:02d}.log').write_text(result.stdout + result.stderr, encoding='utf-8')
        if reject is not None:
            assert result.returncode != 0 and reject in result.stderr, (args, result.stdout, result.stderr)
        elif result.returncode:
            raise RuntimeError(f'{args}: {result.stdout}\n{result.stderr}')

    for index in range(2):
        source = out / f'founder-{index}'
        run('train', '--data', train, '--validation', val, '--out', source, '--steps', 100,
            '--channels', 8, '--hidden', 16, '--layers', 2, '--cell', cell, '--batch', 2, '--context', 16,
            '--eval-batches', 2, '--eval-every', 100, '--warmup', 0, '--seed', 41 + index)
        extra = ['--batch', 2, '--context', 16, '--batches', 2] if index == 0 else []
        run('population-add', '--population', population, '--id', f'founder-{index}',
            '--checkpoint', source / 'latest.ckpt', '--data', val, '--max-score', 20,
            '--growth-chance', 1, '--setting-mutation-chance', 0, '--lifespan', 4, *extra)

    def evolve(name, children, budget):
        run('evolve', '--population', population, '--data', val, '--round', name,
            '--children', children, '--seed', 13, '--food-mib', budget)
        return json.loads((population / (name + '.json')).read_text())

    first = evolve('birth', 2, 64)
    assert first['tick'] == 1 and first['births'] == 2
    assert all(c['grew'] and c['hidden'] == 24 and c['lifespan_ticks'] == 4
               and c['generation'] == 1 and not c['eligible_at_birth'] for c in first['children'])
    scarce = evolve('scarce', 32, 1)
    assert scarce['tick'] == 2 and 0 < scarce['births'] < 32
    assert scarce['stop_reason'] == 'resource_limit'
    assert scarce['food_used_after_bytes'] <= scarce['food_capacity_bytes']
    assert scarce['effective_elite_fraction'] < first['effective_elite_fraction']
    assert scarce['children'][0]['required_improvement'] > first['children'][0]['required_improvement']
    assert all(not m['eligible'] for m in scarce['members'] if m['generation'] == 1)
    assert all(c['parent_a'].startswith('founder-') and c['parent_b'].startswith('founder-')
               for c in scarce['children'])
    empty = evolve('empty', 1, 0)
    assert empty['tick'] == 3 and empty['births'] == 0 and empty['stop_reason'] == 'resource_limit'
    assert empty['food_capacity_bytes'] == 0 and not (population / 'empty-child-0').exists()
    deaths = evolve('old-age', 1, 64)
    assert deaths['tick'] == 4 and deaths['births'] == 0 and deaths['stop_reason'] == 'no_eligible_pairs'
    founders = [m for m in deaths['members'] if m['id'].startswith('founder-')]
    assert len(founders) == 2 and all(not m['alive'] and not m['eligible']
                                    and m['death_reason'] == 'old_age' for m in founders)
    expected_food = 0
    for member in deaths['members']:
        ckpt = population / member['id'] / 'latest.ckpt'
        assert ckpt.is_file(), 'Old-age death must preserve the archived checkpoint'
        meta = struct.unpack_from('<32Q', ckpt.read_bytes())
        if member['alive']:
            expected_food += 20 * meta[14] + 4 * meta[4] * meta[3] * (2 if meta[1] in (2, 3, 4, 5) else 1)
    assert deaths['food_used_before_bytes'] == expected_food
    old = (population / 'population.sg').read_bytes()
    run('evolve', '--population', population, '--data', train, '--round', 'invalid',
        reject='Population evaluation corpus changed')
    assert old == (population / 'population.sg').read_bytes()
    run('evolve', '--population', population, '--data', val, '--round', 'birth', reject='Round already exists')
    assert old == (population / 'population.sg').read_bytes()
    run('population-add', '--population', population, '--id', '../outside',
        '--checkpoint', out / 'founder-0/latest.ckpt', '--data', val, reject='Invalid population member')
    assert not (out / 'outside').exists()
    child_run = population / 'birth-child-0'
    run('train', '--resume', child_run / 'latest.ckpt', '--allow-new-corpus',
        '--data', train, '--validation', val, '--out', child_run, '--steps', 100,
        '--eval-batches', 2, '--eval-every', 100)
    start = json.loads((child_run / 'metrics.jsonl').read_text().splitlines()[0])
    assert start['resumed'] and not start['from_random_initialization']
    report = {'passed': True, 'native_commands': calls, 'ticks': deaths['tick'],
              'growth_at_birth_checked': True, 'newborn_parent_rejection_checked': True,
              'budget_limited_births': scarce['births'], 'zero_budget_births': empty['births'],
              'scarcity_tightens_selection': True, 'old_age_deaths': len(founders),
              'dead_checkpoints_preserved': True, 'dead_population_food_released': True,
              'clock_persisted_across_processes': True, 'changed_evaluation_corpus_rejected': True,
              'duplicate_round_rejected': True, 'path_traversal_id_rejected': True,
              'inherited_child_training_origin_recorded': True,
              'final_population_config_sha256': hashlib.sha256(old).hexdigest()}
    (out / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exe', type=Path, default=Path('build/synapticgenesis.exe'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cell', choices=['lif', 'alif', 'trace', 'gated', 'selective'], default='lif')
    args = parser.parse_args()
    check(args.exe, args.out, args.cell)
