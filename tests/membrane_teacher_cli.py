"""Regularized live learning with two frozen teachers, SI and process restart.

All texts and models are disposable mechanism fixtures, never admitted sources.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct

from teacher_replay_cli import Fixture


def snapshot(path):
    raw = Path(path).read_bytes()
    meta = struct.unpack_from('<32Q', raw)
    policy = struct.unpack_from('<8Q', raw, 288) if meta[17] == 7 else None
    base, prefix = (policy[2], 352) if policy else (meta[17], 288)
    assert base == 6 and meta[1] == 6, 'Expected an associative teacher checkpoint'
    at = prefix + 12 * meta[14] + 4 * meta[18]
    extra = struct.unpack_from(f'<{meta[31]}Q', raw, at)
    teacher = struct.unpack_from('<32Q', raw, at + 8 * meta[31])
    assert extra[9] in (0, 1)
    assert len(raw) == at + 8 * meta[31] + 256 + (12 * meta[14] if extra[9] else 0)
    return dict(sha256=hashlib.sha256(raw).hexdigest(), meta=meta, policy=policy, extra=extra,
                teacher=teacher, weight_sha256=hashlib.sha256(raw[prefix:prefix + 4 * meta[14]]).hexdigest())


def run(exe, out):
    fixture = Fixture(exe, out)
    out = fixture.out
    bundle = out / 'two-teachers'
    fixture.run('teacher-pack', '--teacher-a', fixture.parents[0], '--teacher-b', fixture.parents[1],
                '--data', out / '1.dat', '--out', bundle, '--temperature', 2, '--strength', .4, '--mixture', .5)
    common = ['--curriculum', out / 'full.sg', '--cell', 'associative', '--channels', 16,
              '--hidden', 24, '--layers', 2, '--chunk', 16, '--seed', 1337, '--lr', .0005,
              '--replay', 'stage', '--replay-every', 2, '--replay-capacity', 32,
              '--graph', '--speak-every', 7, '--tokens', 9, '--prompt', 'A',
              '--consolidation', 'si', '--si-strength', .01, '--si-damping', .001,
              '--teacher-bundle', bundle, '--teacher-memory-mib', 64, '--log-every', 1, '--save-every', 1000]
    control, full, split = (out / name for name in ('disabled', 'full', 'split'))
    fixture.run('live', *common, '--out', control, '--updates', 64)
    regularizer = ['--membrane-cost', .02, '--membrane-band', 1.5]
    fixture.run('live', *common, *regularizer, '--out', full, '--updates', 64)
    fixture.run('live', *common, *regularizer, '--out', split, '--updates', 23)
    fixture.run('live', '--resume', split / 'latest.ckpt', '--curriculum', out / 'full.sg',
                '--out', split, '--updates', 64, '--prompt', 'A', '--teacher-bundle', bundle, '--teacher-memory-mib', 64)
    expected, actual, disabled = (snapshot(p / 'latest.ckpt') for p in (full, split, control))
    assert expected == actual and expected['weight_sha256'] != disabled['weight_sha256']
    assert (full / 'transcript.txt').read_bytes() == (split / 'transcript.txt').read_bytes()
    policy = expected['policy']
    assert policy == (0x314d454d504753, 1, 6, struct.unpack('<I', struct.pack('<f', .02))[0],
                       struct.unpack('<I', struct.pack('<f', 1.5))[0], 0, 0, 0)
    meta, extra, teacher = (expected[k] for k in ('meta', 'extra', 'teacher'))
    assert meta[24] == 64 and meta[30] == 81 and extra[15] == 2 and extra[9] == 1
    assert teacher[2] == 1 and teacher[3] == 2 and teacher[9] > 0 and teacher[10] > 0 and extra[12] > 0
    assert meta[7] == meta[24] + extra[6] == extra[13]
    sample, evaluation = out / 'sample.bin', out / 'evaluation.json'
    fixture.run('sample', '--checkpoint', full / 'latest.ckpt', '--prompt', 'A', '--tokens', 24,
                '--seed', 42, '--graph', '--output', sample)
    fixture.run('evaluate', '--checkpoint', full / 'latest.ckpt', '--data', fixture.val,
                '--batch', 2, '--context', 16, '--batches', 2, '--output', evaluation)
    assert snapshot(full / 'latest.ckpt') == expected
    assert sample.read_bytes().startswith(b'A') and len(sample.read_bytes()) == 25
    score = json.loads(evaluation.read_text())
    assert score['step'] == meta[7] and score['evaluated_bytes'] == 64 and math.isfinite(score['loss_nats_per_byte'])
    fixture.run('live', '--resume', full / 'latest.ckpt', '--curriculum', out / 'full.sg',
                '--out', out / 'invalid-resume', '--updates', 65, '--membrane-band', 1.6,
                reject='Live resume preserves membrane-band')
    fixture.run('teacher-pack', '--teacher-a', full / 'latest.ckpt', '--data', out / '1.dat',
                '--out', out / 'invalid-teacher', reject='Experimental membrane policy is not admitted as a teacher')
    assert not (out / 'invalid-resume').exists() and not (out / 'invalid-teacher').exists()
    report = dict(passed=True, native_commands=len(fixture.calls), regularized_checkpoint_exact_on_restart=True,
        speech_exact_on_restart=True, enabled_cost_changes_learned_weights=True, two_teachers_active=True,
        optional_si_exercised=True, source_updates=64, teacher_updates=teacher[9], teacher_pairs=teacher[10],
        full_checkpoint_sha256=expected['sha256'], regularized_weights_sha256=expected['weight_sha256'],
        disabled_weights_sha256=disabled['weight_sha256'], read_only_commands_preserve_checkpoint=True,
        forbidden_resume_override_and_teacher_promotion_rejected=True, language_quality_established=False,
        new_sources_admitted=False)
    (out / 'membrane-result.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exe', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    run(args.exe, args.out)
