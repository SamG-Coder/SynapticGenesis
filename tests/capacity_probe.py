"""Capacity telemetry must leave the production learning trajectory unchanged."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from native_experiment import NativeCommands, read, sha


def check(out):
    out.mkdir(parents=True, exist_ok=False)
    journal = out / 'commands'
    journal.mkdir()
    probe = NativeCommands('build/capacity-probe/synaptic-capacity-probe.exe', journal)
    native_dir = out / 'native'
    native_dir.mkdir()
    native = NativeCommands('build/synapticgenesis.exe', native_dir)
    data = out / 'fixture.dat'
    data.write_bytes(b'The cat sits on the mat.\n\x1eThe boat floats on the pond.\n')
    common = ['--data', data, '--cell', 'associative', '--channels', 32, '--hidden', 64, '--layers', 2,
              '--chunk', 16, '--seed', 1337, '--lr', .0003, '--replay', 'reservoir', '--replay-every', 4,
              '--replay-capacity', 16, '--speak-every', 8, '--tokens', 8, '--prompt', 'A', '--graph']
    cases = []
    for math in ('fp32', 'tf32'):
        options = common + (['--fast'] if math == 'tf32' else [])
        control = out / (math + '-control')
        measured = out / (math + '-probe')
        native('live', *options, '--out', control, '--updates', 72)
        probe('run', *options, '--out', measured, '--warmup', 8, '--updates', 32, '--rounds', 2,
              '--decode-bytes', 32, '--save', 1)
        assert sha(control / 'latest.ckpt') == sha(measured / 'latest.ckpt')
        result = read(measured / 'result.json')
        assert result['complete'] and result['decode_repeated_bytes_exact']
        cases.append(dict(math=math, complete_checkpoint_exact=True, parameters=result['parameters']))
    import json
    (out / 'result.json').write_bytes((json.dumps(dict(passed=True, cases=cases,
        probe_sha256=sha(probe.exe), native_sha256=sha(native.exe)), indent=2) + '\n').encode())
    print('Capacity telemetry preserves complete FP32 and TF32 production checkpoints.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    check(parser.parse_args().out)
