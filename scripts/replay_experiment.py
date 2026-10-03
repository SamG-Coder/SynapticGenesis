"""Paired native live learning with a stream reservoir or stage-balanced memory.

Three declared seeds, one learned ancestor per seed, fixed online/update/slot budgets.
Replay target bytes can differ with window lengths; report them and measured cost.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from audit_binding import audit
from experiment_checkpoint import checkpoint, distribution
from extend_curriculum import read_schedule
from prepare_lessons import quoted


def run(exe, out, seeds):
    exe = exe.resolve()
    executable_sha = hashlib.sha256(exe.read_bytes()).hexdigest()
    out.mkdir(parents=True, exist_ok=False)
    selected = Path('data/prepared/binding-v2')
    manifest = json.loads((selected / 'manifest.json').read_text())
    for name, record in manifest['files'].items():
        assert hashlib.sha256((selected / name).read_bytes()).hexdigest() == record['sha256'], name
    _, stages = read_schedule(selected / 'curriculum.sg')
    assert [s['end_update'] for s in stages] == [6000, 10000, 34000]
    stages[-1]['end_update'] = 130000
    schedule = out / 'curriculum.sg'
    schedule.write_text('SGCURRICULUM3\n' + '\n'.join(
        f'{s["end_update"]} {quoted(s["source"].as_posix())} {s["rate"]} {s["scope"]} {s["answer"]}'
        for s in stages) + '\n')
    boundaries = [sum(len(d) >= 2 for d in s['source'].read_bytes().split(b'\x1e')) for s in stages]
    rows, ancestors, decode = [], [], {}
    calls = 0
    protocol = dict(cell='selective', seeds=seeds, online_endpoints=[34000, 67000, 130000],
                    replay_policies=['reservoir', 'stage'], slots=1024, replay_every=4,
                    learning_rate=.0003, later_stage_multiplier=.25, answer_emphasis=64,
                    generated_bytes_per_500_observations=96, save_every=5000,
                    source_spec_sha256=manifest['spec_sha256'], reserved_test_evaluated=False,
                    executable_sha256=executable_sha,
                    primary_endpoint='Earlier-reader loss at 130000, accompanied by new lesson training/development scores.',
                    comparisons='Same exact learned first-stage checkpoint and online targets; equal replay updates and descriptor slots. '
                                'Replay target identities and byte counts intentionally differ; report runtime cost. '
                                'No checkpoint selection based on development scores.')
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')

    def native(*args):
        nonlocal calls
        calls += 1
        result = subprocess.run([str(exe), *map(str, args)], capture_output=True)
        (out / f'command-{calls:03d}.log').write_bytes(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError(f'{args}: {result.stdout}\n{result.stderr}')

    for si, seed in enumerate(seeds):
        ancestor = out / f'{seed}-ancestor'
        native('live', '--curriculum', schedule, '--out', ancestor, '--updates', 6000,
               '--cell', 'selective', '--seed', seed, '--lr', .0003, '--chunk', 128,
               '--replay', 'reservoir', '--replay-capacity', 1024, '--replay-every', 4,
               '--graph', '--speak-every', 500, '--tokens', 96, '--fast', '--prompt', 'The bird ',
               '--validation', 'data/prepared/development-v2-final/13853.txt', '--eval-batches', 32,
               '--log-every', 6000, '--save-every', 5000)
        source = ancestor / 'latest.ckpt'
        meta, _, _, ancestor_sha = checkpoint(source)
        assert meta[24] == 6000
        ancestors.append(dict(seed=seed, checkpoint_sha256=ancestor_sha,
                              session=json.loads((ancestor/'session.json').read_text()),
                              both_policies_resume_identical_checkpoint=True))
        for index, end in enumerate(protocol['online_endpoints']):
            policies = ('reservoir', 'stage') if (si+index) % 2 == 0 else ('stage', 'reservoir')
            for policy in policies:
                dest = out / f'{seed}-{policy}'
                initialization = ['--resume', dest/'latest.ckpt' if index else source]
                if index == 0 and policy == 'stage':
                    initialization += ['--replay','stage']
                native('live', '--curriculum', schedule, '--out', dest, '--updates', end, *initialization,
                       '--prompt', 'The bird ', '--validation', 'data/prepared/development-v2-final/13853.txt',
                       '--eval-batches', 32, '--log-every', 6000, '--save-every', 5000)
                session = json.loads((dest/'session.json').read_text())
                assert session['online_updates'] == end, 'Run stopped before the declared measurement point'
                snapshot = dest / f'checkpoint-{end}.ckpt'
                shutil.copyfile(dest / 'latest.ckpt', snapshot)
                meta, extra, _, sha = checkpoint(snapshot)
                row = dict(seed=seed, policy=policy, online_updates=end, parameters=meta[14],
                           checkpoint_sha256=sha, stored_windows_by_source_stage=distribution(snapshot, boundaries),
                           session=session, ancestor_checkpoint_sha256=ancestor_sha)
                if policy == 'stage':
                    assert row['stored_windows_by_source_stage'] == [342,341,341]
                for split in ('train','development'):
                    report = dest / f'{split}-{end}.json'
                    native('language-probes', '--checkpoint', snapshot, '--probes', selected / f'{split}.sgprobe',
                           '--output', report)
                    row[split] = {k:v for k,v in json.loads(report.read_text()).items() if k != 'results'}
                    row[split+'_audit'] = audit(report, split)
                rows.append(row)
                (out / 'partial.json').write_text(json.dumps(rows, indent=2)+'\n')
                print(f'{seed} {policy} {end}: reader={row["session"]["final_validation_loss"]:.5f} '
                      f'train={row["train"]["joint_accuracy"]:.4f} dev={row["development"]["joint_accuracy"]:.4f} '
                      f'stored={row["stored_windows_by_source_stage"]}', flush=True)
            paired = [r for r in rows if r['seed']==seed and r['online_updates']==end]
            for key in ('online_updates','global_updates','observed_pairs','replay_updates','generated_bytes',
                        'learning_rate','curriculum_hash'):
                assert paired[0]['session'][key] == paired[1]['session'][key], (seed,end,key)
        assert checkpoint(source)[3] == ancestor_sha, 'Shared ancestor was modified'
        if si == 0:
            for policy in ('reservoir','stage'):
                dest = out / f'{seed}-{policy}'
                native('decode-bench', '--checkpoint', dest/'latest.ckpt', '--tokens', 1024,
                       '--rounds', 7, '--out', dest/'decode')
                decode[policy] = json.loads((dest/'decode/benchmark.json').read_text())
    assert hashlib.sha256(exe.read_bytes()).hexdigest() == executable_sha
    result = dict(protocol=protocol, native_commands=calls, shared_ancestors=ancestors,
                  runs=rows, first_seed_decode=decode)
    (out/'comparison.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--exe', type=Path, default=Path('build/synapticgenesis.exe'))
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--seeds', nargs='+', type=int, default=[1337,2026,31415])
    a = p.parse_args()
    if len(set(a.seeds)) != len(a.seeds) or min(a.seeds) < 1:
        p.error('Seeds must be unique positive integers')
    run(a.exe,a.out,a.seeds)
