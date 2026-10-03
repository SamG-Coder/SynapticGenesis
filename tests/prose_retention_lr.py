"""CPU preflight on real prose evidence plus rejection before any native work."""
import argparse
import copy
from pathlib import Path
import struct
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from native_experiment import read
from prose_founder import file_hash, write
from prose_retention_inputs import (ROOT, SPEC, admit, authenticate, matched, predecessors,
                                   selection, state)
from prose_retention_lr import check_endpoint, execute, learning_args


def audit(out, report):
    out.mkdir(parents=True, exist_ok=False)
    spec = read(SPEC)
    plan = admit(spec)
    parent_path = Path(spec['checkpoint'])
    original = plan['original_completion']
    final = check_endpoint(plan, Path(original['checkpoint']), spec['endpoints'][-1], .0003,
                           original['session'])
    assert final['counters'] == original['counters']
    # Independent bytewise traversal of the actual stage-four documents.
    source = read(plan['evaluation']['source_spec'])
    prepared = Path(plan['evaluation']['prepared'])
    observations, targets = spec['starting_observations'], 12182873
    reached = {}
    for row in source['sources']:
        if row['split'] != 'train' or row['stage'] != 4:
            continue
        raw = (prepared / f'{row["id"]}.txt').read_bytes()
        cursor = 0
        while cursor < len(raw) - 1:
            end = min(cursor + 128, len(raw) - 1)
            targets += len(raw[cursor + 1:end + 1])
            observations += 1
            if observations in spec['endpoints']:
                reached[str(observations)] = targets
            cursor = end
    assert observations == 216289 and targets == 27680129
    assert reached == {k: plan['expected_exposure'][k]['observed_pairs'] for k in reached}
    assert reached == {'155748': 19931542, '216289': 27680129}
    assert plan['additional_books'] == [43, 1257]
    for arm in spec['arms']:
        for end in spec['endpoints']:
            args = list(map(str, learning_args(spec, parent_path, out / 'unused', end, arm['learning_rate'])))
            assert args[0] == 'live' and args[1::2] == [
                '--resume', '--curriculum', '--out', '--updates', '--lr', '--prompt', '--log-every', '--save-every']
            assert args[args.index('--resume') + 1] == str(parent_path)
    rejected = []

    def rejects(name, action, expected):
        try:
            action()
        except ValueError as error:
            assert expected in str(error), str(error)
            rejected.append(name)
        else:
            raise AssertionError(f'{name} was accepted')

    bad = copy.deepcopy(spec)
    bad['checkpoint_sha256'] = '0' * 64
    rejects('unadmitted checkpoint identity', lambda: admit(bad), 'not uniquely admitted')
    policy = copy.deepcopy(selection())
    policy['prose_diagnostic_bases'] = []
    with patch('prose_retention_inputs.selection', return_value=policy):
        rejects('unadmitted continuation base', lambda: admit(spec), 'not uniquely admitted')
    bad_rate = copy.deepcopy(spec)
    bad_rate['arms'][1]['learning_rate'] = .003
    rejects('undeclared rate', lambda: admit(bad_rate), 'comparison arms')
    changed = out / 'changed-input.txt'
    changed.write_bytes(b'original')
    pin = {str(changed): file_hash(changed)}
    changed.write_bytes(b'altered')
    rejects('changed pinned input', lambda: authenticate(pin), 'input changed')
    matched(final, final)
    for key in ('replay_payload_sha256', 'cursor_and_rng', 'speech_policy'):
        altered = copy.deepcopy(final)
        altered[key] = 'changed'
        rejects('changed ' + key, lambda: matched(final, altered), 'differs')
    altered = copy.deepcopy(final)
    altered['counters']['replay_pairs'] += 1
    rejects('changed replay exposure', lambda: matched(final, altered), 'counters differ')
    rejects('wrong requested endpoint', lambda: check_endpoint(plan, parent_path, 155748, .0003),
            'declared exposure')
    rejects('wrong native rate', lambda: check_endpoint(plan, parent_path, 95207, .000075),
            'rate differs')

    comparison_path, spike_path = out / 'comparison.json', out / 'spike.json'
    comparison = dict(complete=True, matched_source_replay_and_speech_counters=True, reserved_tests_scored=False,
                      rows=[dict(profile=p, stage=s) for p in ('2m', '27m', '105m') for s in (1, 2, 3, 4)])
    write(comparison_path, comparison)
    spike = dict(complete=True, compatibility_trace_byte_identical=True, all_checkpoints_unchanged=True,
                 learning_updates=0, reserved_tests_scored=False, native_commands=73,
                 comparison_sha256=file_hash(comparison_path),
                 records=[dict(profile=p, stage=s) for p in ('2m', '27m', '105m') for s in (0, 1, 4)])
    write(spike_path, spike)
    predecessors(comparison_path, spike_path)
    spike['compatibility_trace_byte_identical'] = False
    write(spike_path, spike)
    rejects('failed native trace compatibility', lambda: predecessors(comparison_path, spike_path), 'panel is incomplete')
    comparison['complete'] = False
    write(comparison_path, comparison)
    rejects('incomplete size comparison', lambda: predecessors(comparison_path, spike_path), 'comparison is incomplete')

    execution = out / 'incomplete-predecessor-run'
    execution.mkdir()
    failed_plan = dict(plan, status='declared_before_learning_rate_comparison',
                       predecessors=[str(comparison_path), str(spike_path)])
    write(execution / 'protocol.json', failed_plan)
    with patch('prose_retention_lr.NativeCommands', side_effect=AssertionError('Unexpected native work')) as native:
        rejects('native work before predecessors finish', lambda: execute(execution, 0, None), 'comparison is incomplete')
        native.assert_not_called()
    failure = read(execution / 'failure.json')
    assert failure['attempted_native_commands'] == 0 and failure['partial_artifacts_preserved']
    # This header view never reads 1.3 GB model arrays into Python at once.
    with parent_path.open('rb') as stream:
        header = struct.unpack('<32Q', stream.read(256))
    assert state(parent_path)['counters']['parameters'] == header[14] == 104851472
    result = dict(passed=True, parent_checkpoint_sha256=file_hash(parent_path),
        original_final_checkpoint_sha256=file_hash(original['checkpoint']),
        expected_exposure=plan['expected_exposure'], additional_training_books=plan['additional_books'],
        independent_source_traversal=True, native_session_and_checkpoint_counters_match=True,
        resume_only_arguments=True, rejected=rejected, new_native_commands=0,
        resumed_control_identity_verified=False, lower_rate_learning_verified=False,
        code_sha256={p.relative_to(ROOT).as_posix(): file_hash(p) for p in
                     [ROOT / 'scripts/prose_retention_inputs.py', ROOT / 'scripts/prose_retention_lr.py', Path(__file__)]},
        limits='CPU preflight only. Authenticity, exposure and rejection checks do not establish CUDA restart identity '
               'or an improvement in language quality. Those remain required gates in the native execution.')
    write(report, result)
    print('PASS: real parent/final evidence, independent full exposure, two rates and', len(rejected),
          'negative cases; no CUDA work.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    audit(args.out, args.report)
