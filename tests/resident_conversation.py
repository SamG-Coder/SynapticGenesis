"""Native checkpoint/answer equivalence and real stdin session checks.

The caller serializes access to the GPU before invoking run(). Technical
fixtures are original, tiny state-contract tests, not language-quality models.
"""
from pathlib import Path
import json
import math
import queue
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from native_experiment import NativeCommands, read
from prose_founder import file_hash, write


def require(condition, why):
    if not condition:
        raise ValueError(why)


def quoted(value):
    return '"' + value.replace('\\', '\\\\').replace('"', '\\"') + '"'


def session_file(path, lines):
    path.write_bytes(('SGCONVERSATION1\n' + '\n'.join(lines) + '\n').encode('utf-8'))


def check_events(directory, questions):
    records = [json.loads(line) for line in (directory / 'events.jsonl').read_text().splitlines()]
    start, end = records[0], records[-1]
    require(start['single_parameter_workspace'] and start['independent_question_recurrence'] and
            start['question_math'] == 'strict FP32', 'Resident view contract differs')
    answers = [r for r in records if r['event'] == 'answer']
    require(end['event'] == 'complete' and end['questions'] == len(answers) == len(questions) and
            not end['query_outputs_are_training_targets'], 'Incomplete resident conversation')
    require([r['id'] for r in answers] == list(questions), 'Resident answer order differs')
    require([r['captured_now'] for r in answers] == [True] + [False] * (len(answers) - 1),
            'Graph should be captured only for the first question')
    for row in answers:
        # Native JSON escapes individual UTF-8 bytes as U+00xx. Preserve bytes.
        prefix = ('Question: ' + questions[row['id']] + '\nAnswer: ').encode('utf-8')
        raw = (directory / row['raw_file']).read_bytes()
        answer = row['answer'].encode('latin-1')
        require(raw == prefix + answer and len(answer) == row['generated_bytes'], 'Raw native answer differs')
        require(all(math.isfinite(row[key]) and row[key] > 0 for key in
                    ('decode_seconds', 'resident_decode_bytes_per_second')),
                'Invalid resident generation timing')
    return records, answers


def interactive(candidate, args, directory):
    """Send later turns only after observing the earlier reply from the same PID."""
    lines, output = queue.Queue(), []
    command = [str(candidate), 'run', *map(str, args), '--script', '-']
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    def consume():
        for line in iter(process.stdout.readline, b''):
            output.append(line); lines.put(line)
        lines.put(None)
    thread = threading.Thread(target=consume, daemon=True); thread.start()
    def send(value):
        require(process.poll() is None, 'Resident stdin process ended between turns')
        process.stdin.write(value.encode()); process.stdin.flush()
    def until(prefix):
        deadline = time.monotonic() + 60
        while True:
            remaining = deadline - time.monotonic()
            require(remaining > 0, 'Timed out awaiting resident response')
            item = lines.get(timeout=remaining)
            require(item is not None, 'Resident process closed stdout unexpectedly')
            if item.startswith(prefix):
                require(process.poll() is None, 'Expected the same live process for the next turn')
                return
    try:
        send('SGCONVERSATION1\n')
        until(b'ready parameters=')
        send('ask before "What is water?"\n')
        until(b'answer before: ')
        send('learn 29\n')
        until(b'learned online_updates=29 ')
        send('not-a-command\n')
        until(b'rejected: ')
        send('ask after "What is water?"\n')
        until(b'answer after: ')
        send('quit\n'); process.stdin.close()
        code = process.wait(timeout=60); thread.join(timeout=5)
        require(code == 0 and not thread.is_alive(), 'Interactive resident process failed')
    finally:
        if process.poll() is None:
            process.kill(); process.wait(timeout=10)
        (directory / 'interactive-process.log').write_bytes(b''.join(output))
        write(directory / 'interactive-process.json', dict(command=command, pid=process.pid,
            exit_code=process.returncode, stdout_checked_between_turns=True))
    return dict(pid=process.pid, process_stayed_live_between_turns=True, bad_command_did_not_discard_model=True)


def run(out, candidate, baseline):
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    (out / 'first.dat').write_bytes(b'Alpha sees a bright blue ball.\n\x1eBeta opens the small red box.\n')
    corrections = b'Question: What is two plus three?\nAnswer: Five.\n\x1eQuestion: What is water made of?\nAnswer: Hydrogen and oxygen.\n'
    (out / 'second.dat').write_bytes((out / 'first.dat').read_bytes() + b'\x1e' + corrections)
    old, new = out / 'original.sg', out / 'extended.sg'
    old.write_bytes(b'SGCURRICULUM3\n11 "first.dat" 1 new 1\n')
    new.write_bytes(b'SGCURRICULUM3\n11 "first.dat" 1 new 1\n29 "second.dat" 0.25 new 8\n')
    old_journal, candidate_journal = out / 'legacy-commands', out / 'resident-commands'
    old_journal.mkdir(); candidate_journal.mkdir()
    legacy, resident = NativeCommands(baseline, old_journal), NativeCommands(candidate, candidate_journal)
    results = []
    for cost in (0, .001):
        for fast in (False, True):
            label = ('membrane' if cost else 'ordinary') + ('-tf32' if fast else '-fp32')
            directory = out / label; directory.mkdir()
            base, control, middle, split, measured = [directory / name for name in ('base', 'control', 'middle', 'split', 'resident')]
            options = ['--curriculum', old, '--cell', 'associative', '--channels', 32, '--hidden', 64,
                '--layers', 2, '--seed', 1337, '--chunk', 16, '--lr', .0003, '--replay', 'stage',
                '--replay-every', 4, '--replay-capacity', 32, '--speak-every', 7, '--tokens', 9,
                '--prompt', 'A', '--graph', '--log-every', 1, '--save-every', 100000000]
            if cost: options += ['--membrane-cost', cost]
            if fast: options += ['--fast']
            legacy('live', '--out', base, '--updates', 11, *options)
            legacy('live', '--resume', base / 'latest.ckpt', '--curriculum', old, '--extend-curriculum', new,
                   '--out', control, '--updates', 29, '--prompt', 'A')
            legacy('live', '--resume', base / 'latest.ckpt', '--curriculum', old, '--extend-curriculum', new,
                   '--out', middle, '--updates', 17, '--prompt', 'A')
            legacy('live', '--resume', middle / 'latest.ckpt', '--curriculum', new,
                   '--out', split, '--updates', 29, '--prompt', 'A')
            plan = directory / 'session.sgconversation'
            session_file(plan, ['save start0', 'ask before "What is water?"', 'save start1', 'learn 17',
                'save middle0', 'ask middle "What is water?"', 'save middle1', 'learn 29',
                'save end0', 'ask after "What is water?"', 'save end1', 'quit'])
            resident('run', '--resume', base / 'latest.ckpt', '--curriculum', old, '--extend-curriculum', new,
                     '--script', plan, '--out', measured, '--prompt', 'A', '--answer-bytes', 64,
                     '--answer-top-k', 40, '--answer-temperature', .8)
            for point in ('start', 'middle', 'end'):
                require(file_hash(measured / f'checkpoint-{point}0.ckpt') == file_hash(measured / f'checkpoint-{point}1.ckpt'),
                        'A resident question altered full learned, recurrent or optimizer state')
            require(file_hash(measured / 'checkpoint-middle0.ckpt') == file_hash(middle / 'latest.ckpt'),
                    'Resident intermediate learning differs from the preserved runtime')
            final_hash = file_hash(control / 'latest.ckpt')
            require(all(file_hash(path) == final_hash for path in (split / 'latest.ckpt',
                    measured / 'checkpoint-end0.ckpt', measured / 'final.ckpt')),
                    'Resident final learning or full restart differs from the preserved runtime')
            for name, checkpoint in (('before', base / 'latest.ckpt'), ('middle', middle / 'latest.ckpt'), ('after', control / 'latest.ckpt')):
                sample = directory / f'legacy-{name}.bin'
                legacy('sample', '--checkpoint', checkpoint, '--prompt', 'Question: What is water?\nAnswer: ',
                    '--tokens', 64, '--seed', 42, '--top-k', 40, '--temperature', .8, '--graph', '--output', sample)
                require(sample.read_bytes() == (measured / f'answer-{name}.bin').read_bytes(),
                        'Resident raw generation differs from the preserved runtime')
            records, answers = check_events(measured, dict.fromkeys(('before', 'middle', 'after'), 'What is water?'))
            stream = None
            if not cost and not fast:
                streamed = directory / 'stdin-session'
                stream = interactive(candidate, ['--resume', middle / 'latest.ckpt', '--curriculum', new,
                    '--out', streamed, '--prompt', 'A', '--answer-bytes', 64, '--answer-top-k', 40,
                    '--answer-temperature', .8], directory)
                require(file_hash(streamed / 'final.ckpt') == final_hash, 'Interactive learning differs')
                check_events(streamed, dict.fromkeys(('before', 'after'), 'What is water?'))
                require((streamed / 'answer-before.bin').read_bytes() == (directory / 'legacy-middle.bin').read_bytes() and
                        (streamed / 'answer-after.bin').read_bytes() == (directory / 'legacy-after.bin').read_bytes(),
                        'Interactive raw answers differ')
            record = dict(case=label, parameters=records[0]['parameters'], full_checkpoint_exact=True,
                questions_preserve_full_checkpoint=True, raw_generations_exact=True,
                final_checkpoint_sha256=final_hash, interactive=stream, answer_metrics=answers)
            results.append(record)
            print('Resident native parity passed:', label, flush=True)
    result = dict(passed=True, cases=results, legacy_commands=len(legacy.commands),
        resident_file_commands=len(resident.commands), interactive_commands=1,
        candidate_sha256=file_hash(candidate), baseline_sha256=file_hash(baseline),
        language_quality_tested=False)
    write(out / 'result.json', result)
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    args = parser.parse_args()
    run(args.out, args.candidate, args.baseline)
