"""Audit the actual failed restart transcripts and reject altered speech."""
import argparse
import hashlib
import json
from pathlib import Path

from membrane_teacher_cli import match_restart_speech


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check(failed, out):
    assert not out.exists(), 'Use a fresh report path'
    full_path, split_path = (failed / name / 'transcript.txt' for name in ('full', 'split'))
    full, split = full_path.read_bytes(), split_path.read_bytes()
    result = match_restart_speech(full, split)
    assert full != split, 'Fixture no longer reproduces the original assertion failure'
    checkpoint_paths = [failed / name / 'latest.ckpt' for name in ('full', 'split')]
    assert sha(checkpoint_paths[0]) == sha(checkpoint_paths[1])
    resume = b'\n[session starts at online update 23]\n'
    header = lambda n: f'\n[online update {n}; global update {n + n // 2}]\n'.encode('ascii')
    body_start = split.index(header(7)) + len(header(7)) + 1
    changed = bytearray(split)
    changed[body_start] ^= 1
    without_resume = split.replace(resume, b'', 1)
    p14, p21, p28 = (split.index(header(n)) for n in (14, 21, 28))
    faults = {
        'generated_byte': bytes(changed),
        'lossy_utf8': split.decode('utf-8', errors='replace').encode('utf-8'),
        'missing_resume_marker': without_resume,
        'wrong_resume_update': split.replace(b'update 23]', b'update 24]', 1),
        'duplicate_resume_marker': split.replace(resume, resume + resume, 1),
        'misplaced_resume_marker': without_resume.replace(header(21), resume + header(21), 1),
        'wrong_global_update': split.replace(b'update 28; global update 42', b'update 28; global update 43', 1),
        'missing_speech_record': split[:p14] + split[p21:],
        'reordered_speech_records': split[:p14] + split[p21:p28] + split[p14:p21] + split[p28:],
        'extra_byte': split + b'x',
        'missing_final_newline': split[:-1],
    }
    rejected = []
    for name, raw in faults.items():
        try:
            match_restart_speech(full, raw)
        except AssertionError:
            rejected.append(name)
        else:
            raise AssertionError('Altered transcript accepted: ' + name)
    inputs = [full_path, split_path, *checkpoint_paths, Path(__file__),
              Path(__file__).with_name('membrane_teacher_cli.py')]
    record = dict(passed=True, original_assertion_failure_reproduced=True,
        full_checkpoint_exact_on_restart=True, restart_speech=result, rejected=rejected,
        inputs_sha256={str(p.resolve()): sha(p) for p in inputs}, new_native_commands=0,
        limits='CPU audit of the preserved failed fixture and transcript comparison. '
               'The remaining native CLI checks require a fresh execution.')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2) + '\n', encoding='utf8', newline='\n')
    print('Actual checkpoint/speech match; rejected', len(rejected), 'altered transcripts.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--failed', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    check(args.failed.resolve(), args.out.resolve())
