"""Sequence the declared size controls and assessments after a live founder exits."""
import argparse
from pathlib import Path
import struct
import subprocess

from native_experiment import read, sha
from process_gate import ProcessGate
from prose_evaluation import assess
from prose_founder import file_hash, run as train_founder, write


def counters(path):
    """Small diagnostic header view; native assessment validates the full checkpoint."""
    with path.open('rb') as stream:
        meta = struct.unpack('<32Q', stream.read(256))
        if meta[17] != 5 or not 17 <= meta[31] <= 16384:
            raise ValueError('Expected a bounded grouped-replay checkpoint')
        stream.seek(288 + 12 * meta[14] + 4 * meta[18])
        extra = struct.unpack(f'<{meta[31]}Q', stream.read(8 * meta[31]))
    if extra[1] != 3:
        raise ValueError('Unexpected grouped-replay policy')
    return dict(online_updates=meta[24], global_updates=meta[7], observed_pairs=meta[22],
                generated_bytes=meta[30], replay_updates=extra[6], replay_pairs=extra[7],
                curriculum_stage=extra[15] + 1, parameters=meta[14],
                channels=meta[2], hidden=meta[3], layers=meta[4])


def require_complete(founder, plan):
    spec, exposure = plan['specification'], plan['exposure']
    final = counters(founder / 'stage-4.ckpt')
    expected = dict(online_updates=spec['stage_endpoints'][-1], observed_pairs=exposure['source_pairs'],
                    replay_updates=spec['stage_endpoints'][-1] // 4,
                    generated_bytes=(spec['stage_endpoints'][-1] // 500) * 96)
    if any(final[k] != v for k, v in expected.items()):
        raise ValueError('Founder has not completed the declared source exposure')
    session = read(founder / 'session.json')
    if any(session[k] != final[k] for k in expected):
        raise ValueError('Final session and checkpoint counters differ')
    if file_hash(founder / 'latest.ckpt') != file_hash(founder / 'stage-4.ckpt'):
        raise ValueError('Latest checkpoint is not the completed immutable endpoint')
    return final


def summarize(rows):
    summaries = []
    for row in rows:
        profile, stage = row['profile'], row['stage']
        baseline = next(r for r in rows if r['profile'] == '2m' and r['stage'] == stage)
        early = next(r for r in rows if r['profile'] == profile and r['stage'] == 1)
        first = {r['book']: r['loss_nats_per_byte'] for r in early['books']}
        summaries.append(dict(profile=profile, stage=stage, parameters=row['parameters'],
            source_observations=row['online_updates'],
            validation_mean_nats_per_byte=row['validation_mean_nats_per_byte'],
            validation_difference_from_2m=row['validation_mean_nats_per_byte'] - baseline['validation_mean_nats_per_byte'],
            training_retention=[dict(book=r['book'], loss_nats_per_byte=r['loss_nats_per_byte'],
                change_from_stage_one=r['loss_nats_per_byte'] - first[r['book']])
                for r in row['books'] if r['role'] == 'training_retention']))
    return summaries


def run(founder, plan_path, out, wait_pid):
    plan = read(plan_path)
    assert plan['status'] == 'declared_before_extended_assessment'
    for name, digest in plan['authenticated_inputs'].items():
        if file_hash(name) != digest:
            raise ValueError(f'Declared assessment input changed: {name}')
    out.mkdir(parents=True, exist_ok=False)
    gate = ProcessGate(wait_pid, 'build/synapticgenesis.exe') if wait_pid else None
    try:
        inputs = {p.as_posix(): file_hash(p) for p in [plan_path, Path(__file__),
                    Path('scripts/process_gate.py'), founder / 'protocol.json', founder / 'continuation-8176.json']}
        protocol = dict(status='declared_before_size_controls', assessment_plan=str(plan_path),
            assessment_plan_sha256=sha(plan_path), original_founder=str(founder),
            native_predecessor=gate.identity if gate else None,
            source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            authenticated_inputs=inputs, profiles_in_order=['105m', '2m', '27m'],
            stage_order=[1, 2, 3, 4], seed=1337,
            policy='Finish the original 105M process, verify its final immutable checkpoint and session, '
                   'then assess it; train and assess each smaller random control sequentially. '
                   'No simultaneous GPU experiments and no restart of the original process.',
            timing_limit='Full-run timing is not a controlled speed comparison: the original 105M '
                         'continuation saves every 8192 observations, controls save every 2048. '
                         'Use the separate bounded-capacity benchmark for speed.',
            quality_limit=plan['specification']['limits'])
        write(out / 'protocol.json', protocol)
        if gate:
            print('Waiting on verified native process handle:', gate.identity, flush=True)
            code = gate.wait()
            write(out / 'predecessor-exit.json', dict(**gate.identity, exit_code=code))
            print('Original native process exited successfully; checking final source counters.', flush=True)
        initial_counters = require_complete(founder, plan)
        write(out / 'founder-completion.json', dict(checkpoint=str(founder / 'stage-4.ckpt'),
            checkpoint_sha256=file_hash(founder / 'stage-4.ckpt'), counters=initial_counters,
            session=read(founder / 'session.json')))
        rows = []
        for profile in protocol['profiles_in_order']:
            directory = founder if profile == '105m' else out / f'founder-{profile}'
            if profile != '105m':
                print('Starting matched random founder', profile, flush=True)
                train_founder(directory, plan['exposure']['online_updates'], profile)
            require_complete(directory, plan)
            for stage in protocol['stage_order']:
                destination = out / f'{profile}-stage-{stage}'
                print('Assessing', profile, 'stage', stage, flush=True)
                assess(plan_path, directory, stage, destination)
                row = read(destination / 'result.json')
                row['exposure_counters'] = counters(directory / f'stage-{stage}.ckpt')
                rows.append(row)
                write(out / 'partial.json', rows)
        fields = ('online_updates', 'global_updates', 'observed_pairs', 'generated_bytes', 'replay_updates', 'replay_pairs')
        for stage in protocol['stage_order']:
            matched = [r['exposure_counters'] for r in rows if r['stage'] == stage]
            if any(any(record[k] != matched[0][k] for k in fields) for record in matched[1:]):
                raise ValueError('Model-size controls did not receive the same exposure')
        if any(file_hash(name) != digest for name, digest in inputs.items()):
            raise ValueError('Comparison inputs changed during execution')
        write(out / 'comparison.json', dict(complete=True, protocol=protocol, rows=rows,
                    summary=summarize(rows), matched_source_replay_and_speech_counters=True,
                    reserved_tests_scored=False, reproduction_admitted=False))
        print('Completed all three sizes at all four declared endpoints.', flush=True)
    finally:
        if gate:
            gate.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--founder', type=Path, default=Path('runs/prose-105m-founder'))
    parser.add_argument('--plan', type=Path, default=Path('runs/prose-evaluation-declared/protocol.json'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--wait-pid', type=int, default=0)
    args = parser.parse_args()
    run(args.founder, args.plan, args.out, args.wait_pid)
