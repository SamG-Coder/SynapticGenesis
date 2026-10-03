"""Shared native command journals, source authentication and binding assessment."""
import hashlib
import json
from pathlib import Path
import subprocess

from audit_binding import audit


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8')


def verified_manifest(path):
    path = Path(path).resolve()
    manifest = read(path/'manifest.json')
    for name, record in manifest['files'].items():
        source = (path/name).resolve()
        if not source.is_relative_to(path) or sha(source) != record['sha256']:
            raise ValueError(f'Prepared source changed: {name}')
    return manifest


class NativeCommands:
    """Sequential commands with their arguments saved before execution."""
    def __init__(self, exe, out):
        self.exe, self.out, self.commands = Path(exe).resolve(), Path(out), []
        if not self.out.is_dir() or (self.out/'commands.json').exists():
            raise ValueError('Native journal needs a fresh existing experiment directory')

    def __call__(self, *args):
        command = [str(self.exe), *map(str, args)]
        self.commands.append(command)
        write(self.out/'commands.json', self.commands)
        result = subprocess.run(command, capture_output=True)
        log = self.out/f'command-{len(self.commands):03d}.log'
        log.write_bytes(result.stdout+result.stderr)
        if result.returncode:
            raise RuntimeError(f'Native command failed; inspect {log}')


def binding_scores(native, checkpoint, selected, out, end):
    result = {}
    for split in ('train', 'development', 'expanded-train-monitor'):
        report = out/f'{split}-{end}.json'
        native('language-probes', '--checkpoint', checkpoint,
               '--probes', selected/f'{split}.sgprobe', '--output', report)
        result[split] = {k:v for k,v in read(report).items() if k != 'results'}
        if split != 'expanded-train-monitor':
            result[split+'_audit'] = audit(report, split)
    return result
