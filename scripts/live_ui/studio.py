"""Serialized local sessions, explicit feedback, append-only training editions."""
import hashlib
import json
import math
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from extend_curriculum import prepare, read_schedule
from live_ui.native import Resident


def write(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=True), encoding='utf-8')
    temp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def text(value, limit):
    if not isinstance(value, str) or not value.strip() or len(value.encode('utf-8')) > limit:
        raise ValueError('Text is empty or exceeds the byte limit.')
    if any(ord(c) < 32 for c in value) or '\nAnswer: ' in value:
        raise ValueError('Use a single line without control characters.')
    return value.strip()


class Studio:
    def __init__(self, workspace, catalog, executable):
        self.workspace = Path(workspace).resolve()
        self.root = self.workspace / 'runs/live-studio'
        self.root.mkdir(parents=True, exist_ok=True)
        self.catalog = json.loads(Path(catalog).read_text(encoding='utf-8-sig'))['models']
        self.executable = Path(executable).resolve()
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.jobs = {}; self.lock = threading.Lock(); self.resident = None; self.session = None
        self.status = 'Ready to load a model'

    def models(self):
        return [dict(id=m['id'], name=m['name'], description=m['description'], parameters=m['parameters'],
            available=(self.workspace / m['checkpoint']).is_file() and (self.workspace / m['curriculum']).is_file()) for m in self.catalog]

    def submit(self, action, payload):
        with self.lock:
            if any(j['state'] == 'running' for j in self.jobs.values()):
                raise ValueError('Wait for the current operation to finish.')
            ident = uuid.uuid4().hex
            if len(self.jobs) > 100:
                self.jobs = {}
            self.jobs[ident] = dict(state='running')
        def run():
            try:
                value = getattr(self, action)(payload)
                self.jobs[ident] = dict(state='complete', result=value)
                self.status = 'Ready'
            except Exception as error:
                self.jobs[ident] = dict(state='failed', error=str(error))
                self.status = 'Operation failed'
                if self.resident and not isinstance(error, ValueError):
                    self.resident.abort(); self.resident = None
        self.pool.submit(run)
        return ident

    def persist(self):
        write(Path(self.session['directory']) / 'session.json', self.session)

    def release(self):
        if self.resident:
            self.status = 'Saving current session'
            self.session['checkpoint'] = str(self.resident.close())
            self.resident = None; self.persist()

    def start(self, checkpoint, curriculum, extension=None):
        path = Path(self.session['directory']) / ('native-' + uuid.uuid4().hex[:10])
        self.resident = Resident(self.executable, checkpoint, curriculum, path, extension)

    def load(self, payload):
        model = next((m for m in self.catalog if m['id'] == payload.get('model')), None)
        if model is None:
            raise ValueError('Unknown model')
        self.release(); self.status = 'Verifying and loading model'
        cp = self.workspace / model['checkpoint']; curriculum = self.workspace / model['curriculum']
        if digest(cp) != model['sha256'] or digest(curriculum) != model['curriculum_sha256']:
            raise ValueError('Model or curriculum differs from the selected catalog.')
        directory = self.root / uuid.uuid4().hex
        directory.mkdir()
        self.session = dict(id=directory.name, directory=str(directory), model=model['name'],
            checkpoint=str(cp), curriculum=str(curriculum), parent_sha256=model['sha256'],
            created_utc=datetime.now(timezone.utc).isoformat(), messages=[], corrections=[])
        self.start(cp, curriculum); self.persist()
        return self.view()

    def resume(self, payload):
        ident = payload.get('session', '')
        if not isinstance(ident, str) or len(ident) != 32 or any(c not in '0123456789abcdef' for c in ident):
            raise ValueError('Invalid session')
        self.release()
        data = json.loads((self.root / ident / 'session.json').read_text())
        self.session = data
        self.start(data['checkpoint'], data['curriculum'])
        return self.view()

    def ensure(self):
        if not self.resident:
            raise ValueError('Load or resume a model first.')
        # Bound the native command/snapshot limits for long browser sessions.
        if self.resident.command_count >= 40:
            checkpoint = self.resident.close(); self.resident = None
            self.session['checkpoint'] = str(checkpoint); self.persist()
            self.start(checkpoint, self.session['curriculum'])

    def ask(self, payload):
        self.ensure(); question = text(payload.get('question'), 1024)
        self.status = 'Generating answer'
        row = self.resident.ask(question, 'q' + uuid.uuid4().hex[:16])
        row['question'] = question
        self.session['messages'].append(row); self.persist()
        return self.view()

    def teach(self, payload):
        self.ensure()
        question = text(payload.get('question'), 1024); answer = text(payload.get('answer'), 1024)
        passes = payload.get('passes', 16)
        if type(passes) is not int or not 1 <= passes <= 256:
            raise ValueError('Practice count must be between 1 and 256.')
        raw = ('Question: ' + question + '\nAnswer: ' + answer + '\n').encode('utf-8')
        if len(raw) > 2048:
            raise ValueError('Combined correction exceeds 2048 bytes.')
        old = Path(self.session['curriculum']); _, stages = read_schedule(old)
        # New stage starts after the previous schedule. Catalog models are completed endpoints.
        if self.resident.initial['online_updates'] != stages[-1]['end_update']:
            raise ValueError('Finish the existing curriculum before appending corrections.')
        windows = math.ceil((len(raw)-1)/128)
        endpoint = stages[-1]['end_update'] + passes * windows
        folder = Path(self.session['directory']) / ('correction-' + uuid.uuid4().hex[:10]); folder.mkdir()
        target = folder / 'target.txt'; target.write_bytes(raw)
        prepare(old, target, folder / 'edition', endpoint, .25, 'new', 8)
        record = dict(question=question, answer=answer, passes=passes, source='explicit-user-feedback',
            target_sha256=digest(target), path=str(target), state='pending',
            created_utc=datetime.now(timezone.utc).isoformat())
        self.session['corrections'].append(record); self.persist()
        self.status = 'Saving and admitting correction'
        checkpoint = self.resident.close(); self.resident = None
        self.session['checkpoint'] = str(checkpoint); self.persist()
        curriculum = folder / 'edition/curriculum.sg'
        self.start(checkpoint, old, curriculum)
        self.status = 'Learning correction'
        metrics = self.resident.learn(endpoint)
        saved = self.resident.save('learned')
        self.session['checkpoint'] = str(saved); self.session['curriculum'] = str(curriculum)
        self.resident.initial['online_updates'] = endpoint
        record.update(state='learned', metrics=metrics)
        self.persist()
        self.status = 'Testing corrected answer'
        return self.ask(dict(question=question))

    def save(self, payload):
        self.ensure()
        self.session['checkpoint'] = str(self.resident.save('s' + uuid.uuid4().hex[:12]))
        self.persist(); return self.view()

    def view(self):
        return self.session

    def sessions(self):
        rows = []
        for path in sorted(self.root.glob('*/session.json'), key=lambda p:p.stat().st_mtime, reverse=True):
            data = json.loads(path.read_text())
            rows.append(dict(id=data['id'], model=data['model'], created_utc=data['created_utc'], messages=len(data['messages'])))
        return rows[:30]
