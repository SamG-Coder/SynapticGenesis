"""Replay archived native outputs to verify the assessment extraction on CPU."""
import argparse
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from checkpoint_assessment import measure
from native_experiment import read, sha
from prose_founder import write


class ArchivedCommands:
    def __init__(self, archive, corrupt_sample=False):
        self.commands = read(archive / 'commands.json')
        self.index = 0
        self.corrupt_sample = corrupt_sample

    def __call__(self, *args):
        actual = list(map(str, args))
        expected = self.commands[self.index][1:]
        # Only the destination changes. Every model/input/sampling argument
        # must still equal the command that produced the archived result.
        assert actual[:-1] == expected[:-1] and actual[-2] == '--output'
        shutil.copyfile(expected[-1], actual[-1])
        if self.corrupt_sample and actual[0] == 'sample':
            Path(actual[-1]).write_bytes(b'wrong prompt and length')
        self.index += 1


def audit(out, report):
    out.mkdir(parents=True, exist_ok=False)
    spec = read('data/prose-evaluation-v1.json')
    cases = []
    for stage in (1, 4):
        archive = Path(f'runs/prose-size-panel/105m-stage-{stage}')
        original = read(archive / 'result.json')
        destination = out / str(stage)
        destination.mkdir()
        replay = ArchivedCommands(archive)
        result = measure(replay, Path(original['checkpoint']), spec, destination)
        assert replay.index == len(replay.commands) == 10
        assert result == {k: original[k] for k in result}
        cases.append(dict(stage=stage, command_journal_sha256=sha(archive / 'commands.json'),
                          result_sha256=sha(archive / 'result.json'), replayed_commands=10))
    rejected = []
    try:
        measure(None, None, spec, out, [('duplicate', [spec['validation_books'][0]])])
    except ValueError:
        rejected.append('duplicate book roles')
    else:
        raise AssertionError('Duplicate role was accepted')
    bad = out / 'bad-sample'
    bad.mkdir()
    replay = ArchivedCommands(Path('runs/prose-size-panel/105m-stage-1'), corrupt_sample=True)
    try:
        measure(replay, Path('runs/prose-105m-founder/stage-1.ckpt'), spec, bad)
    except AssertionError:
        assert replay.index == 7
        rejected.append('malformed raw generation')
    else:
        raise AssertionError('Malformed generation was accepted')
    result = dict(passed=True, cases=cases, rejected=rejected,
                  existing_native_commands_replayed=20, new_native_commands=0,
                  exact_command_arguments=True, exact_book_and_sample_results=True,
                  helper_sha256=sha(Path(__file__).resolve().parents[1] / 'scripts/checkpoint_assessment.py'),
                  limitation='CPU replay of existing native artifacts, not fresh CUDA execution.')
    write(report, result)
    print('PASS: 20 archived commands and all book/sample fields preserved; two bad cases rejected.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    audit(args.out, args.report)
