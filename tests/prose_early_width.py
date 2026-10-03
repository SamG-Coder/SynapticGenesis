"""CPU preflight for the compiled observer and guarded study orchestration."""
import argparse
import copy
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from early_width_observations import audit
from prose_early_width import PROBE, SPEC, arguments, initial_match, match_policy, validate
from prose_founder import file_hash, write
from prose_projection_snapshot import layout
from prose_retention_inputs import state
from native_experiment import read
import prose_early_width as study


def fixture(out):
    """Synthetic arrays exercise artifact checks; this is not a learned model."""
    out.mkdir()
    case = dict(channels=8, hidden=32, layers=2, core_scale=1.)
    blocks, count = layout(8, 32, 2)
    fields = [(0, 2048, (256, 8)), *(f for block in blocks for f in block),
              (count - 2312, count - 2304, (8,)), (count - 2304, count - 256, (256, 8)), (count - 256, count, (256,))]
    roles, offsets = [], []
    for index, (first, last, _) in enumerate(fields):
        n = min(64, last - first)
        selected = [first + (i * (last - first - 1) // (n - 1) if n > 1 else 0) for i in range(n)]
        roles.append(dict(name=f'fixture-role-{index}', tensor_begin=first, tensor_size=last - first,
                          sample_begin=len(offsets), sample_size=n))
        offsets.extend(selected)
    write(out / 'coordinates.json', dict(sample_limit_per_tensor=64, roles=roles, offsets=offsets))
    initial = (np.arange(count, dtype=np.float32) % 17 - 8) / np.float32(256)
    meta = [0] * 32
    meta[14] = count
    (out / 'initial.ckpt').write_bytes(struct.pack('<32Q', *meta) + bytes(32) + initial.tobytes())
    initial[offsets].tofile(out / 'initial-coordinates.f32')
    source = out / 'source.dat'
    source.write_bytes(b'a' * 61 + b'\x1e' + b'b' * 98)
    points, rows = [1, 4, 7], []
    widths = [60, 97, 60, 97, 60, 97, 60]

    def moments(values):
        vals = list(map(float, values))
        return dict(count=len(vals), mean_absolute=math.fsum(abs(v) for v in vals) / len(vals),
                    rms=math.sqrt(math.fsum(v * v for v in vals) / len(vals)), maximum_absolute=max(map(abs, vals)))

    for at in points:
        before = (initial + np.float32(at / 4096))[offsets]
        after = before + np.float32(1 / 4096)
        gradient = np.zeros(len(offsets), dtype=np.float32)
        gradient[0] = .25
        for name, data in [('before', before), ('after', after), ('clipped-gradient', gradient)]:
            data.tofile(out / f'source-{at}-{name}.f32')
        measured = []
        for role in roles:
            s = slice(role['sample_begin'], role['sample_begin'] + role['sample_size'])
            measured.append(dict(name=role['name'], weights_before=moments(before[s]), weights_after=moments(after[s]),
                source_update_delta=moments(after[s].astype(float) - before[s]),
                displacement_from_initial=moments(after[s].astype(float) - initial[offsets][s]),
                clipped_gradient=moments(gradient[s])))
        rows.append(dict(source_update=at, source_global_update=at + (at - 1) // 4,
            global_updates_after_tick=at + at // 4, document=(at - 1) % 2, byte_offset=0,
            target_bytes=widths[at - 1], source_loss=2., source_gradient_norm_before_clip=2., source_clip_factor=.5,
            generated_bytes_this_tick=0, roles=measured,
            pre_update_forward_layers=[dict(neuron_positions=widths[at - 1] * 32, spike_event_fraction=.25,
                zero_local_spike_surrogate_fraction=.5) for _ in range(2)]))
    (out / 'observations.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    final = initial.copy()
    final[offsets] = after
    (out / 'latest.ckpt').write_bytes(struct.pack('<32Q', *meta) + bytes(32) + final.tobytes())
    (out / 'speech.bin').write_bytes(b'')
    write(out / 'result.json', dict(complete=True, shared_production_live_tick=True,
        generated_text_targets=False, reserved_tests_scored=False, **case, parameters=count, seed=1337,
        source_updates=7, source_pairs=sum(widths), global_updates=8, replay_updates=1, generated_bytes=0,
        observations=3, source_gradient_norm_before_clip=dict(count=7), source_updates_clipped=7))
    return case, points, source, rows


def main(out, report):
    out.mkdir(parents=True, exist_ok=False)
    spec = read(SPEC)
    validate(spec)
    source = Path('runs/prose-scale-curriculum/curriculum.sg')
    archived = []
    for profile, directory, case in [('27m', Path('runs/prose-size-panel/founder-27m'), spec['cases'][0]),
                                     ('105m', Path('runs/prose-105m-founder'), spec['cases'][1])]:
        original = read(directory / 'commands/commands.json')[0][1:]
        expected = list(original)
        expected[expected.index('--out') + 1] = str(out / profile)
        expected[expected.index('--updates') + 1] = '8176'
        expected += ['--core-scale', '1.0']
        actual = list(map(str, arguments(case, 1337, out / profile, source, 8176)))
        assert actual == expected
        archived.append(profile)
    probe_arguments = arguments(spec['cases'][-1], 2027, out / 'candidate', source, 8176, spec['observation_points'])
    assert probe_arguments[0] == 'run' and '--cell' not in probe_arguments and '--save-every' not in probe_arguments
    assert probe_arguments[probe_arguments.index('--core-scale') + 1] == .25
    assert probe_arguments[probe_arguments.index('--seed') + 1] == 2027
    tiny = out / 'fixture'
    case, points, text, rows = fixture(tiny)
    accepted = audit(tiny, case, 1337, 7, points, text)
    original = (tiny / 'observations.jsonl').read_bytes()
    rejected = []

    def reject(name, action):
        try:
            action()
        except (ValueError, AssertionError):
            rejected.append(name)
        else:
            raise AssertionError('Malformed input accepted: ' + name)

    for key, value in [('source_global_update', 99), ('byte_offset', 1), ('target_bytes', 59),
                       ('source_clip_factor', 1.), ('generated_bytes_this_tick', 96)]:
        changed = copy.deepcopy(rows)
        changed[0][key] = value
        (tiny / 'observations.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in changed))
        reject(key, lambda: audit(tiny, case, 1337, 7, points, text))
    changed = copy.deepcopy(rows)
    changed[0]['roles'][0]['source_update_delta']['rms'] *= 2
    (tiny / 'observations.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in changed))
    reject('fabricated_update_statistic', lambda: audit(tiny, case, 1337, 7, points, text))
    (tiny / 'observations.jsonl').write_bytes(original)
    gradient = tiny / 'source-1-clipped-gradient.f32'
    raw = gradient.read_bytes()
    gradient.write_bytes(raw[:-4])
    reject('truncated_gradient', lambda: audit(tiny, case, 1337, 7, points, text))
    gradient.write_bytes(struct.pack('<f', 2.) + raw[4:])
    reject('impossible_clipped_gradient', lambda: audit(tiny, case, 1337, 7, points, text))
    gradient.write_bytes(raw)
    altered = copy.deepcopy(spec)
    altered['cases'][0]['layers'] = 4
    reject('changed_depth', lambda: validate(altered))
    altered = copy.deepcopy(spec)
    altered['cases'][-1]['core_scale'] = .5
    reject('changed_rate_control', lambda: validate(altered))
    left, right = state('runs/prose-size-panel/founder-27m/stage-1.ckpt'), state('runs/prose-105m-founder/stage-1.ckpt')
    match_policy(left, right)
    right['hyperparameters'][6] = .25
    match_policy(left, right)
    right['counters']['observed_pairs'] += 1
    reject('changed_source_exposure', lambda: match_policy(left, right))
    original_initial = tiny / 'initial.ckpt'
    candidate = out / 'candidate-initial.ckpt'
    raw = original_initial.read_bytes()
    changed = bytearray(raw)
    changed[280:284] = struct.pack('<f', .25)
    candidate.write_bytes(changed)
    initial_match(original_initial, candidate)
    changed[-1] ^= 1
    candidate.write_bytes(changed)
    reject('changed_initial_weight', lambda: initial_match(original_initial, candidate))
    blocked = out / 'incomplete-predecessor'
    blocked.mkdir()
    write(blocked / 'protocol.json', dict(status='declared_before_early_width_learning',
                                        specification=spec, authenticated_inputs={}))
    original_read = study.read

    def unfinished(path):
        if str(path) == spec['learning_rate_result']:
            return dict(complete=True, control_checkpoint_byte_identical=True, native_commands=64,
                        reserved_tests_scored=False)
        if str(path) == spec['predecessor_result']:
            return dict(complete=False)
        return original_read(path)

    with patch.object(study, 'read', side_effect=unfinished), patch.object(study, 'NativeCommands') as runner:
        try:
            study.execute(blocked)
        except ValueError as error:
            assert str(error) == 'Replay-capacity predecessor is incomplete'
            rejected.append('incomplete_predecessor_before_native_runner')
        else:
            raise AssertionError('Incomplete predecessor was accepted')
        runner.assert_not_called()
    host = subprocess.run([str(PROBE), 'host-test'], capture_output=True, check=True)
    (out / 'host-test.log').write_bytes(host.stdout + host.stderr)
    host_result = json.loads(host.stdout)
    assert host_result['passed'] and host_result['gpu_work'] is False and host_result['rejected_cases'] == 11
    selector = subprocess.run(['powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File',
        str(ROOT / 'build.ps1'), '-CapacityProbe', '-EarlyLearningProbe'], capture_output=True)
    (out / 'build-selector.log').write_bytes(selector.stdout + selector.stderr)
    assert selector.returncode != 0 and b'Select one diagnostic build per invocation.' in selector.stderr
    assert len(rejected) == 13
    write(report, dict(passed=True, original_native_command_policies_preserved=archived,
        compiled_host_test=host_result, rejected_cpu_cases=rejected, incompatible_build_targets_rejected=True,
        accepted_synthetic_coordinate_fixture=accepted,
        diagnostic_executable_sha256=file_hash(PROBE), production_executable_sha256=file_hash('build/synapticgenesis.exe'),
        code_sha256={p.relative_to(ROOT).as_posix(): file_hash(p) for p in
            (ROOT / 'experiments/early_learning_probe.cu', ROOT / 'experiments/learning_scale_observer.cuh',
             ROOT / 'scripts/early_width_observations.py', ROOT / 'scripts/prose_early_width.py', Path(__file__))},
        native_host_commands=1, gpu_model_commands=0, real_instrumented_learning_verified=False,
        limits='Compiled CPU diagnostics, archived argument checks and synthetic artifact guard tests only. '
               'Native observer neutrality, CUDA coordinate gathering and early-learning outcomes remain pending.'))
    print('Compiled host tests, 13 CPU rejections and the build selector guard passed; no new CUDA model calls.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    main(args.out, args.report)
