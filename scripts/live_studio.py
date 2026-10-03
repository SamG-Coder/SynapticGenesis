"""Loopback HTML/API host; Python only orchestrates the native CUDA process."""
import argparse
import json
import secrets
import signal
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from live_ui.studio import Studio

ROOT = Path(__file__).resolve().parents[1]


def serve(studio, port):
    token = secrets.token_urlsafe(32)
    origin = 'http://127.0.0.1:' + str(port)
    class Handler(BaseHTTPRequestHandler):
        def response(self, code, value, kind='application/json'):
            raw = json.dumps(value).encode() if kind == 'application/json' else value
            self.send_response(code); self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(raw))); self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Security-Policy', "default-src 'self'; style-src 'self'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'")
            self.end_headers(); self.wfile.write(raw)
        def safe(self):
            return self.headers.get('Host') == '127.0.0.1:' + str(port)
        def do_GET(self):
            if not self.safe():
                return self.response(403, dict(error='Use the loopback URL.'))
            if self.path == '/api/state':
                return self.response(200, dict(token=token, status=studio.status, models=studio.models(),
                    sessions=studio.sessions(), session=studio.view()))
            if self.path.startswith('/api/jobs/'):
                return self.response(200, studio.jobs.get(self.path.rsplit('/',1)[-1], dict(state='failed',error='Unknown job')))
            files = {'/':'index.html', '/index.html':'index.html', '/app.js':'app.js', '/styles.css':'styles.css'}
            if self.path not in files:
                return self.response(404, dict(error='Not found'))
            name = files[self.path]
            self.response(200, (ROOT/'ui'/name).read_bytes(), {'html':'text/html; charset=utf-8','js':'text/javascript; charset=utf-8','css':'text/css; charset=utf-8'}[name.rsplit('.',1)[-1]])
        def do_POST(self):
            if not self.safe() or self.headers.get('Origin') != origin or self.headers.get('X-Studio-Token') != token:
                return self.response(403, dict(error='Local session authorization required.'))
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 16384:
                    raise ValueError('Invalid request size')
                data = json.loads(self.rfile.read(size))
                action = self.path.removeprefix('/api/')
                if self.path != '/api/' + action or action not in ('load','resume','ask','teach','save') or not isinstance(data, dict):
                    raise ValueError('Unknown operation')
                self.response(202, dict(job=studio.submit(action, data)))
            except (ValueError, TypeError) as error:
                self.response(400, dict(error=str(error)))
        def log_message(self, fmt, *args):
            pass
    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    print('SynapticGenesis Studio: ' + origin, flush=True)
    try:
        server.serve_forever()
    finally:
        studio.pool.shutdown(wait=True)
        studio.release()
        server.server_close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=Path, default=ROOT)
    parser.add_argument('--catalog', type=Path, default=ROOT/'data/studio-models.json')
    parser.add_argument('--runtime', type=Path, default=ROOT/'build/resident-conversation/synaptic-resident-conversation.exe')
    parser.add_argument('--port', type=int, default=8765)
    a = parser.parse_args()
    if not a.runtime.is_file():
        parser.error('Build the resident runtime first or pass --runtime.')
    serve(Studio(a.workspace, a.catalog, a.runtime), a.port)
