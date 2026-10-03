"""Native resident process adapter. All model computation remains C++/CUDA."""
import json
import subprocess
import time
from pathlib import Path


def quoted(value):
    return '"' + value.replace('\\', '\\\\').replace('"', '\\"') + '"'


class Resident:
    def __init__(self, executable, checkpoint, curriculum, directory, extension=None):
        self.directory = Path(directory)
        self.position = 0
        self.command_count = 0
        self.directory.parent.mkdir(parents=True, exist_ok=True)
        self.log = open(str(directory) + '.log', 'wb')
        args = [str(executable), 'run', '--resume', str(checkpoint), '--curriculum', str(curriculum),
                '--out', str(directory), '--script', '-', '--answer-bytes', '192']
        if extension:
            args += ['--extend-curriculum', str(extension)]
        self.process = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=self.log, stderr=subprocess.STDOUT)
        self.send('SGCONVERSATION1')
        try:
            self.initial = self.wait('start')
        except BaseException:
            self.abort()
            raise

    def send(self, line):
        if self.process.poll() is not None:
            raise RuntimeError('Native process exited. See session log.')
        self.process.stdin.write((line + '\n').encode('utf-8'))
        self.process.stdin.flush()
        self.command_count += 1

    def wait(self, event, timeout=600):
        deadline = time.monotonic() + timeout
        path = self.directory / 'events.jsonl'
        while time.monotonic() < deadline:
            if path.exists():
                with path.open('rb') as stream:
                    stream.seek(self.position)
                    for line in stream:
                        if not line.endswith(b'\n'):
                            break
                        self.position += len(line)
                        row = json.loads(line)
                        if row.get('event') == event:
                            return row
            if self.process.poll() is not None:
                raise RuntimeError('Native process exited: ' + Path(self.log.name).read_text(errors='replace')[-1200:])
            time.sleep(.05)
        raise TimeoutError('Native operation timed out; inspect the session log before retrying.')

    def ask(self, question, ident):
        self.send('ask ' + ident + ' ' + quoted(question))
        row = self.wait('answer')
        raw = row.pop('answer').encode('latin-1')
        row['text'] = raw.decode('utf-8', errors='replace')
        row['raw_hex'] = raw.hex()
        try:
            raw.decode('utf-8'); row['valid_utf8'] = True
        except UnicodeDecodeError:
            row['valid_utf8'] = False
        return row

    def learn(self, endpoint):
        self.send('learn ' + str(endpoint))
        return self.wait('learned')

    def save(self, label):
        self.send('save ' + label)
        self.wait('saved')
        return self.directory / ('checkpoint-' + label + '.ckpt')

    def close(self):
        if self.process.poll() is None:
            self.send('quit')
            self.process.stdin.close()
            self.wait('complete')
            self.process.wait(timeout=60)
        self.log.close()
        return self.directory / 'final.ckpt'

    def abort(self):
        if self.process.poll() is None:
            self.process.kill(); self.process.wait()
        self.log.close()
