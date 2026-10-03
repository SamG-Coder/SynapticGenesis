"""Start a larger random founder on the admitted complete-book curriculum."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess

from corpus.selection import require_training_spec
from native_experiment import NativeCommands, read, sha, verified_book_manifest


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    path.write_bytes((json.dumps(value, indent=2) + '\n').encode())


def run(out, updates, profile):
    spec = Path('data/sources-prose-scale-v1.json')
    prepared = Path('data/prepared/prose-scale-v1-pinned')
    schedule = Path('runs/prose-scale-curriculum/curriculum.sg')
    plan = read(schedule.parent / 'protocol.json')
    require_training_spec(spec)
    verified_book_manifest(prepared, spec)
    assert sha(schedule) == plan['schedule_sha256']
    assert sha(prepared / 'manifest.json') == plan['prepared_manifest_sha256']
    assert 0 < updates <= plan['online_updates']
    shapes = {'2m': (256, 512, 4), '27m': (512, 2048, 8), '105m': (1024, 4096, 8)}
    channels, hidden, layers = shapes[profile]
    inputs = {str(p): sha(p) for p in (spec, prepared / 'manifest.json', schedule,
              Path('build/synapticgenesis.exe'), *(Path(r['cumulative_source']) for r in plan['stages']))}
    out.mkdir(parents=True, exist_ok=False)
    protocol = dict(status='declared_before_learning', source_commit=subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], text=True).strip(), profile=profile, channels=channels,
        neurons_per_layer=hidden, layers=layers, cell='associative', seed=1337,
        random_initialization=True, imported_weights=False, updates=updates, full_schedule_updates=plan['online_updates'],
        source_schedule=plan, authenticated_inputs=inputs, learning_rate=.0003,
        replay='ordinary stage-balanced, every four source observations, 1024 stored descriptors',
        graph_speech_every=500, generated_bytes_per_speech=96, generated_text_targets=False,
        reserved_tests_scored=False, reproduction_admitted=False,
        scope='Initial learning checkpoint at a declared exposure; not a language-quality or scaling comparison.')
    write(out / 'protocol.json', protocol)
    commands = out / 'commands'
    commands.mkdir()
    native = NativeCommands('build/synapticgenesis.exe', commands)
    native('live', '--out', out, '--curriculum', schedule, '--cell', 'associative', '--channels', channels,
           '--hidden', hidden, '--layers', layers, '--seed', 1337, '--chunk', 128, '--lr', .0003,
           '--updates', updates, '--replay', 'stage', '--replay-every', 4, '--replay-capacity', 1024,
           '--graph', '--fast', '--speak-every', 500, '--tokens', 96, '--prompt', 'The bird ',
           '--log-every', 512, '--save-every', 2048)
    assert all(sha(p) == digest for p, digest in inputs.items())
    completed_stage = next((r['stage'] for r in plan['stages'] if r['end_update'] == updates), None)
    checkpoint = out / (f'stage-{completed_stage}.ckpt' if completed_stage else 'latest.ckpt')
    with checkpoint.open('rb') as stream:
        meta = struct.unpack('<32Q', stream.read(256))
    assert meta[24] == updates and meta[2:5] == (channels, hidden, layers) and meta[17] == 5
    result = dict(complete=True, protocol=protocol, session=read(out / 'session.json'),
                  checkpoint=str(checkpoint), checkpoint_sha256=file_hash(checkpoint),
                  parameters=meta[14], spiking_neurons=hidden * layers,
                  quality_evaluated=False, full_corpus_completed=updates == plan['online_updates'])
    write(out / 'result.json', result)
    print('Saved',meta[14], 'parameter founder at',updates,'source observations.',flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--updates', type=int, default=8176)
    parser.add_argument('--profile', choices=('2m', '27m', '105m'), default='105m')
    args = parser.parse_args()
    run(args.out, args.updates, args.profile)
