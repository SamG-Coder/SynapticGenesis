"""Authenticate older fixtures and repeat their ordinary native continuations."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(exe, baseline, out):
    out.mkdir(parents=True, exist_ok=False)
    evidence = json.loads(baseline.read_text())
    matched = []
    for name, expected in evidence['baseline_fixture_sha256'].items():
        assert sha(Path(name)) == expected, name
        matched.append(name)
    records, commands = [], []
    for command, expected in zip(evidence['commands'], evidence['continuations']):
        command[0] = str(exe.resolve())
        dest = out/expected['model']
        command[command.index('--out')+1] = str(dest)
        commands.append(command)
        (out/'commands.json').write_text(json.dumps(commands, indent=2)+'\n')
        result = subprocess.run(command, capture_output=True, text=True)
        (out/(expected['model']+'.log')).write_text(result.stdout+result.stderr)
        assert not result.returncode, result.stderr
        actual = sha(dest/'latest.ckpt')
        assert actual == expected['checkpoint_sha256'], (expected['model'], actual)
        records.append(dict(model=expected['model'], checkpoint_sha256=actual, complete_checkpoint_identical=True))
    result = dict(passed=True, executable_sha256=sha(exe), baseline_report_sha256=sha(baseline),
                  numerical_fixture_files_identical=len(matched), fixture_paths=matched, continuations=records)
    (out/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--exe', type=Path, default=Path('build/synapticgenesis.exe'))
    p.add_argument('--baseline', type=Path, default=Path('reports/distillation-compatibility.json'))
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    check(a.exe, a.baseline, a.out)
