"""Declare and run fixed development checks on immutable curriculum checkpoints."""
import argparse
from pathlib import Path
import struct
import subprocess

from corpus.selection import require_training_spec
from native_experiment import NativeCommands, read, sha, verified_book_manifest
from prose_founder import file_hash, write


SPEC = Path('data/prose-evaluation-v1.json')


def declare(out):
    spec = read(SPEC)
    source = Path(spec['source_spec'])
    require_training_spec(source)
    prepared = Path(spec['prepared'])
    manifest = verified_book_manifest(prepared, source)
    schedule = Path(spec['schedule'])
    exposure = read(schedule.parent / 'protocol.json')
    assert [r['end_update'] for r in exposure['stages']] == spec['stage_endpoints']
    assert sha(schedule) == exposure['schedule_sha256']
    assert sha(prepared / 'manifest.json') == exposure['prepared_manifest_sha256']
    splits = {r['id']: r['split'] for r in manifest['sources']}
    assert all(splits[n] == 'validation' for n in spec['validation_books'])
    assert all(splits[n] == 'train' for n in spec['training_retention_books'])
    assert len(set(spec['validation_books'] + spec['training_retention_books'])) == 6
    e = spec['evaluation']
    assert e['batch'] * e['context'] * e['batches'] == e['target_bytes_per_book']
    files = [SPEC, source, prepared / 'manifest.json', schedule,
             Path('build/synapticgenesis.exe'), Path(__file__),
             Path('scripts/prose_founder.py'), Path('scripts/native_experiment.py'),
             *(prepared / f'{n}.txt' for n in splits),
             *(Path(r['cumulative_source']) for r in exposure['stages'])]
    plan = dict(status='declared_before_extended_assessment', specification=spec,
                source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                authenticated_inputs={p.as_posix(): file_hash(p) for p in files},
                exposure=exposure, reserved_tests_scored=False, native_work_started=False)
    out.mkdir(parents=True, exist_ok=False)
    write(out / 'protocol.json', plan)
    print('Declared four validation books, two training monitors, four prompts and four endpoints.')


def assess(plan_path, run, stage, out):
    plan = read(plan_path)
    assert plan['status'] == 'declared_before_extended_assessment'
    spec = plan['specification']
    require_training_spec(spec['source_spec'])
    for name, digest in plan['authenticated_inputs'].items():
        if file_hash(name) != digest:
            raise ValueError(f'Assessment input changed: {name}')
    founder = read(run / 'protocol.json')
    assert founder['profile'] in spec['profiles'] and founder['random_initialization']
    assert founder['seed'] == spec['seed'] and not founder['imported_weights']
    assert founder['cell'] == 'associative' and founder['learning_rate'] == .0003
    assert founder['source_schedule'] == plan['exposure']
    assert not founder['generated_text_targets'] and not founder['reserved_tests_scored']
    assert (founder['graph_speech_every'], founder['generated_bytes_per_speech']) == (500, 96)
    for name, digest in founder['authenticated_inputs'].items():
        assert digest == plan['authenticated_inputs'][Path(name).as_posix()]
    checkpoint = run / f'stage-{stage}.ckpt'
    with checkpoint.open('rb') as stream:
        meta = struct.unpack('<32Q', stream.read(256))
    assert list(meta[2:5]) == spec['profiles'][founder['profile']]
    assert meta[24] == spec['stage_endpoints'][stage - 1] and meta[17] == 5 and meta[1] == 6
    before = file_hash(checkpoint)
    out.mkdir(parents=True, exist_ok=False)
    identity = dict(plan_sha256=sha(plan_path), founder_protocol_sha256=sha(run / 'protocol.json'),
                    checkpoint=checkpoint.as_posix(), checkpoint_sha256=before,
                    stage=stage, online_updates=meta[24], profile=founder['profile'],
                    parameters=meta[14], spiking_neurons=meta[3] * meta[4])
    write(out / 'protocol.json', identity)
    native = NativeCommands('build/synapticgenesis.exe', out)
    prepared = Path(spec['prepared'])
    book_rows, generated = [], []
    e, g = spec['evaluation'], spec['generation']
    for role, ids in [('validation', spec['validation_books']),
                      ('training_retention', spec['training_retention_books'])]:
        for ident in ids:
            report = out / f'book-{ident}.json'
            native('evaluate', '--checkpoint', checkpoint, '--data', prepared / f'{ident}.txt',
                   '--batch', e['batch'], '--context', e['context'], '--batches', e['batches'],
                   '--output', report)
            result = read(report)
            assert result['evaluated_bytes'] == e['target_bytes_per_book']
            book_rows.append(dict(book=ident, role=role, **result))
    for i, prompt in enumerate(g['prompts']):
        output = out / f'sample-{i}.txt'
        native('sample', '--checkpoint', checkpoint, '--prompt', prompt, '--tokens', g['bytes'],
               '--seed', g['seed'], '--temperature', g['temperature'], '--top-k', g['top_k'],
               '--graph', '--output', output)
        raw = output.read_bytes()
        assert raw.startswith(prompt.encode()) and len(raw) == len(prompt.encode()) + g['bytes']
        generated.append(dict(prompt=prompt, output_utf8=raw.decode('utf-8', errors='replace'),
                              output_hex=raw.hex(), sha256=sha(output), generated_bytes=g['bytes']))
    assert file_hash(checkpoint) == before
    assert all(file_hash(name) == digest for name, digest in plan['authenticated_inputs'].items())
    validation = [r['loss_nats_per_byte'] for r in book_rows if r['role'] == 'validation']
    result = dict(complete=True, **identity, books=book_rows, samples=generated,
                  validation_mean_nats_per_byte=sum(validation) / len(validation),
                  checkpoint_unchanged=True, reserved_tests_scored=False,
                  limitation=spec['limits'])
    write(out / 'result.json', result)
    print(founder['profile'], 'stage', stage, 'validation mean', result['validation_mean_nats_per_byte'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest='command', required=True)
    declaration = commands.add_parser('declare')
    declaration.add_argument('--out', type=Path, required=True)
    assessment = commands.add_parser('assess')
    assessment.add_argument('--plan', type=Path, required=True)
    assessment.add_argument('--run', type=Path, required=True)
    assessment.add_argument('--stage', type=int, choices=(1, 2, 3, 4), required=True)
    assessment.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'declare':
        declare(args.out)
    else:
        assess(args.plan, args.run, args.stage, args.out)
