"""Host checks for larger founder admission, inherited exposure and CLI compatibility."""
import argparse
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from large_founder_inputs import SPEC, admit, facts, first_stage_pairs, validate
from native_experiment import read
from prose_founder import file_hash, live_arguments, write
from prose_large_founder import same_exposure


def check(out, preserved):
    assert preserved.resolve() != (ROOT / 'scripts/prose_founder.py').resolve(), 'Use the preserved earlier helper'
    out.mkdir(parents=True, exist_ok=False)
    module_spec = importlib.util.spec_from_file_location('preserved_founder', preserved)
    old = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(old)
    controls = []
    for profile in ('2m', '27m', '105m'):
        for capacity in (1024, 16384):
            args = (Path('unused'), Path('schedule'), 8176, profile, capacity)
            assert live_arguments(*args) == old.live_arguments(*args)
            controls.append(dict(profile=profile, replay_capacity=capacity, native_arguments_unchanged=True))
    large = list(map(str, live_arguments(Path('unused'), Path('schedule'), 216289, '411m', 16384,
                                        save_every=8192)))
    for key, value in [('--channels', '2048'), ('--hidden', '8192'), ('--layers', '8'),
                       ('--save-every', '8192'), ('--replay-capacity', '16384')]:
        assert large[large.index(key) + 1] == value
    rejected = 0

    def fails(action):
        nonlocal rejected
        try:
            action()
        except (ValueError, AssertionError):
            rejected += 1
        else:
            raise AssertionError('Corrupt large-founder policy was accepted')

    spec = read(SPEC)
    for key, value in [('shape', [1024, 4096, 8]), ('parameters', 411028497), ('spiking_neurons', 65535),
                       ('seed', 42), ('source_observations', 216288), ('learning_rate', .000075),
                       ('core_scale', .25), ('replay_capacity', 1024), ('replay_every', 5), ('save_every', 2048),
                       ('native_preflight', dict(spec['native_preflight'], evaluation_batch=1))]:
        altered = deepcopy(spec); altered[key] = value
        fails(lambda: validate(altered))
    for value in (0, -1, 1000000001):
        fails(lambda: live_arguments(Path('unused'), Path('schedule'), 8176, '411m', save_every=value))
    admitted = admit()
    source = admitted['exposure']['stages'][0]['cumulative_source']
    # The first source has 14,481 next-byte targets: 113 full windows and 17 bytes.
    expected_pairs = {113: 14464, 114: 14481, 128: 16273, 129: 16401}
    assert all(first_stage_pairs(source, count) == pairs for count, pairs in expected_pairs.items())
    baseline = read(spec['baseline_result'])
    checked = []
    for stage in range(1, 5):
        a, b = [next(r for r in baseline['rows'] if r['stage'] == stage and r['profile'] == profile)
                for profile in ('2m', '105m')]
        left, right = facts(a['checkpoint']), facts(b['checkpoint'])
        same_exposure(left, right)
        checked.append(dict(stage=stage, replay_payload_sha256=left['replay_payload_sha256'],
                            different_shapes_same_source_replay_speech=True))
    for key in ('online_updates', 'global_updates', 'observed_pairs', 'generated_bytes',
                'replay_updates', 'replay_pairs', 'curriculum_stage'):
        altered = deepcopy(right); altered['counters'][key] += 1
        fails(lambda: same_exposure(left, altered))
    for key in ('seed', 'batch', 'chunk', 'fast', 'graph', 'replay_every', 'replay_capacity'):
        altered = deepcopy(right); altered[key] += 1
        fails(lambda: same_exposure(left, altered))
    for key in ('hyperparameters', 'cursor_and_rng', 'speech_policy'):
        altered = deepcopy(right); altered[key][0] += 1
        fails(lambda: same_exposure(left, altered))
    altered = deepcopy(right); altered['replay_payload_sha256'] = '0' * 64
    fails(lambda: same_exposure(left, altered))
    result = dict(passed=True, cuda_work_executed=False, preserved_helper=str(preserved),
                  preserved_helper_sha256=file_hash(preserved), legacy_argument_controls=controls,
                  actual_baseline_endpoints=checked, source_tail_pair_counts=expected_pairs,
                  malformed_policy_cases_rejected=rejected,
                  admitted_input_count=len(admitted['authenticated_inputs']),
                  source_sha256={p.as_posix(): file_hash(p) for p in
                      [SPEC, ROOT / 'scripts/prose_founder.py', ROOT / 'scripts/large_founder_inputs.py',
                       ROOT / 'scripts/prose_large_founder.py', Path(__file__)]})
    write(out / 'result.json', result)
    print('CPU checks passed:', len(controls), 'legacy argument cases, four real replay endpoints,',
          rejected, 'rejected policy changes; no CUDA work.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--preserved-helper', type=Path, default=Path('scripts/prose_founder.py'))
    args = parser.parse_args()
    check(args.out, args.preserved_helper)
