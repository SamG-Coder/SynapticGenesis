"""Bounded live workspace must match the preserved runtime across ragged streams."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from native_experiment import NativeCommands, sha


def check(out):
    out.mkdir(parents=True, exist_ok=False)
    old_dir, new_dir = out / 'old', out / 'new'
    old_dir.mkdir(); new_dir.mkdir()
    old = NativeCommands('build/pre-bounded-live/synapticgenesis.exe', old_dir)
    new = NativeCommands('build/synapticgenesis.exe', new_dir)
    data = out / 'ragged.dat'
    phrase = b'The red ball is in the box. A cat watches the ball. '
    data.write_bytes(b'\x1e'.join((phrase * 3)[:33 + i] for i in range(32)))
    cases = []
    for cell in ('lif', 'alif', 'trace', 'gated', 'selective', 'associative'):
        for fast in (False, True):
            label = cell + ('-tf32' if fast else '-fp32')
            options = ['--data', data, '--cell', cell, '--channels', 32, '--hidden', 64, '--layers', 2,
                       '--chunk', 16, '--seed', 1337, '--lr', .0003, '--replay', 'reservoir',
                       '--replay-every', 4, '--replay-capacity', 32, '--speak-every', 31, '--tokens', 8,
                       '--prompt', 'A', '--graph'] + (['--fast'] if fast else [])
            control, measured, split, resumed = [out / (label + '-' + p) for p in ('old', 'new', 'split', 'resumed')]
            old('live', *options, '--out', control, '--updates', 384)
            new('live', *options, '--out', measured, '--updates', 384)
            assert sha(control / 'latest.ckpt') == sha(measured / 'latest.ckpt'), label
            assert (control / 'transcript.txt').read_bytes() == (measured / 'transcript.txt').read_bytes(), label
            new('live', *options, '--out', split, '--updates', 173)
            new('live', '--data', data, '--resume', split / 'latest.ckpt', '--out', resumed,
                '--updates', 384, '--prompt', 'A')
            assert sha(measured / 'latest.ckpt') == sha(resumed / 'latest.ckpt'), label
            cases.append(dict(cell=cell, tf32=fast, complete_old_checkpoint_exact=True,
                              old_speech_exact=True, complete_restart_exact=True))
            print(label, 'old runtime and restart exact', flush=True)
    result = dict(passed=True, cases=cases, old_sha256=sha(old.exe), new_sha256=sha(new.exe),
                  fixture_sha256=sha(data), document_lengths=list(range(33, 65)), native_commands=48)
    (out / 'result.json').write_bytes((json.dumps(result, indent=2) + '\n').encode())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    check(parser.parse_args().out)
