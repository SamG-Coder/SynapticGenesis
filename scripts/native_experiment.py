"""Shared native command journals, source authentication and binding assessment."""
import hashlib
import json
from pathlib import Path
import subprocess

from audit_binding import audit


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


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


def verified_book_manifest(path, specification):
    """Authenticate a selected-book edition without scoring its reserved files."""
    path, specification = Path(path).resolve(), Path(specification)
    manifest, selection = read(path / 'manifest.json'), read(specification)
    assert sha(specification) == manifest['source_spec_sha256']
    assert (path / 'source-spec.json').read_bytes() == specification.read_bytes()
    expected = {(r['id'], r['split'], r['stage']) for r in selection['sources']}
    assert {(r['id'], r['split'], r['stage']) for r in manifest['sources']} == expected
    assert len(expected) == len(manifest['sources'])
    for record in manifest['sources']:
        book = path / f'{int(record["id"])}.txt'
        assert sha(book) == record['clean_sha256'] and book.stat().st_size == record['clean_bytes']
    for split, record in manifest['outputs'].items():
        assert split in ('train', 'validation', 'test')
        books = [(path / f'{r["id"]}.txt').read_bytes() for r in manifest['sources'] if r['split'] == split]
        assert b'\x1e'.join(books) == (path / f'{split}.dat').read_bytes()
        assert sha(path / f'{split}.dat') == record['sha256']
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


def binding_scores(native, checkpoint, selected, out, end,
                   splits=('train', 'development', 'expanded-train-monitor')):
    result = {}
    for split in splits:
        report = out/f'{split}-{end}.json'
        native('language-probes', '--checkpoint', checkpoint,
               '--probes', selected/f'{split}.sgprobe', '--output', report)
        result[split] = {k:v for k,v in read(report).items() if k != 'results'}
        if split != 'expanded-train-monitor':
            result[split+'_audit'] = audit(report, split)
    return result
