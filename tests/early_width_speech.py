"""Check prompt-inclusive binary speech against the preserved real native guard."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from early_width_observations import SPEECH_PROMPT, audit, match_guard_speech, speech_records
from native_experiment import read
from prose_early_width import PROBE, arguments
from prose_founder import file_hash, write
from prose_retention_inputs import authenticate
from unittest.mock import patch


def check(out, failed, report_path):
    out.mkdir(parents=True, exist_ok=False)
    accepted, rejected = [], []
    # Include newlines, a zero byte, non-UTF8 bytes and text resembling a prompt.
    generated = (bytes(range(60)) + b'The bird \n\x00\xff' + b'x' * 96)[:96]
    frame = SPEECH_PROMPT + generated
    for end in (0, 499, 500, 511, 1000, 8176):
        raw = frame * (end // 500)
        assert speech_records(raw, end) == [frame] * (end // 500)
        accepted.append(end)

    def reject(label, action):
        try:
            action()
        except ValueError:
            rejected.append(label)
        else:
            raise AssertionError('Altered speech accepted: ' + label)

    for name, raw, end in [('missing_prompt', generated, 500), ('truncated_record', frame[:-1], 500),
                           ('extra_byte', frame + b'\n', 500), ('wrong_prompt', b'X' + frame[1:], 500),
                           ('wrong_second_prompt', frame + b'X' + frame[1:], 1000),
                           ('missing_record', frame, 1000), ('early_record', frame, 499)]:
        reject(name, lambda: speech_records(raw, end))

    control, observed = out / 'control', out / 'observed'
    control.mkdir()
    observed.mkdir()
    (observed / 'speech.bin').write_bytes(frame * 2)
    transcript = (b'\n[session starts at online update 0]\n'
                  b'\n[online update 500; global update 625]\n' + frame + b'\n'
                  b'\n[online update 1000; global update 1250]\n' + frame + b'\n')
    (control / 'transcript.txt').write_bytes(transcript)
    assert match_guard_speech(control, observed, 1000)['raw_transcript_byte_identical']
    for name, changed in [('changed_generated_byte', transcript[:-2] + b'z\n'),
                          ('changed_global_update', transcript.replace(b'1250', b'1251')),
                          ('extra_transcript_byte', transcript + b'\n'),
                          ('missing_transcript_newline', transcript[:-1])]:
        (control / 'transcript.txt').write_bytes(changed)
        reject(name, lambda: match_guard_speech(control, observed, 1000))

    # The script's assumptions are checked against actual founder arguments.
    import prose_early_width as study
    base = study.live_arguments
    tiny = dict(channels=8, hidden=32, layers=2, core_scale=1.)
    for option, value in [('--prompt', 'Other '), ('--tokens', 95), ('--speak-every', 499)]:
        def changed_arguments(*args, **kwargs):
            args = list(base(*args, **kwargs))
            args[args.index(option) + 1] = value
            return args
        with patch.object(study, 'live_arguments', side_effect=changed_arguments):
            reject('changed_policy_' + option[2:], lambda: arguments(tiny, 1337, out, Path('schedule.sg'), 511))

    protocol_path = failed / 'protocol.json'
    protocol, failure = read(protocol_path), read(failed / 'failure.json')
    assert failure['error'] == 'Speech output extent differs' and failure['completed_cases'] == 0
    authenticate(protocol['authenticated_inputs'])
    before = {p.as_posix(): file_hash(p) for folder in ('guard-control', 'guard-observed')
              for p in (failed / folder).iterdir() if p.is_file()}
    native_audit = audit(failed / 'guard-observed', tiny, 1337, 511, [1, 128, 511],
                        'data/prepared/prose-scale-v1-pinned/through-stage-1.dat')
    native_speech = match_guard_speech(failed / 'guard-control', failed / 'guard-observed', 511)
    checkpoints = {}
    for filename in ('initial.ckpt', 'latest.ckpt'):
        left, right = [file_hash(failed / folder / filename) for folder in ('guard-control', 'guard-observed')]
        assert left == right
        checkpoints[filename] = left
    assert file_hash(PROBE) == file_hash(protocol['diagnostic_executable'])
    original = Path(protocol['source_checkout'])
    native_files = [p.relative_to(ROOT) for pattern in ('*.cu', '*.cuh') for p in (ROOT / 'src').glob(pattern)]
    # Both source trees are pinned; no native binary or numerical source changed.
    native_files += [Path('experiments/early_learning_probe.cu'), Path('experiments/learning_scale_observer.cuh')]
    for relative in native_files:
        assert (ROOT / relative).read_bytes() == (original / relative).read_bytes(), relative
    assert before == {name: file_hash(name) for name in before}
    result = dict(passed=True, accepted_speech_boundaries=accepted, rejected_cases=rejected,
        binary_payload_with_newlines_zero_and_non_utf8_preserved=True,
        prompt_bytes_per_record=9, generated_bytes_per_record=96,
        original_failure_preserved=True, failed_protocol_sha256=file_hash(protocol_path),
        failed_status_sha256=file_hash(failed / 'failure.json'), completed_study_cases_in_failed_run=0,
        existing_native_guard=native_audit, existing_native_speech=native_speech,
        byte_identical_guard_checkpoints=checkpoints, unchanged_diagnostic_executable_sha256=file_hash(PROBE),
        native_source_files_unchanged=len(native_files), original_guard_artifacts_sha256=before,
        implementation_sha256={p.relative_to(ROOT).as_posix(): file_hash(p) for p in
            (ROOT / 'scripts/early_width_observations.py', ROOT / 'scripts/prose_early_width.py', Path(__file__))},
        new_native_commands=0, new_models_trained=False,
        limits='CPU verification of the preserved two-command native guard, not completion of the 15-case width study. '
               'The failure remains recorded under its original protocol; a rerun needs a new declared output directory.')
    write(report_path, result)
    print('Prompt-inclusive speech passed six boundaries and', len(rejected), 'rejections; preserved native guard verified.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--failed', type=Path, default=Path('runs/prose-early-width'))
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    assert not args.report.exists(), 'Use a fresh report'
    check(args.out, args.failed, args.report)
