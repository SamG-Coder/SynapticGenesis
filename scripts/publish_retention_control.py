"""Publish the completed restart control while its paired rate trial continues."""
import argparse
import hashlib
from pathlib import Path
import struct

from experiment_checkpoint import policy_checkpoint
from native_experiment import read
from prose_founder import file_hash, write
from prose_size_comparison import counters


def publish(root, out):
    if out.exists():
        raise ValueError('Use a fresh publication path')
    plan = read(root / 'protocol.json')
    spec, evaluation = plan['specification'], plan['evaluation']
    partial = read(root / 'partial.json')
    identity = read(root / 'control-identity.json')
    assert identity['passed'] and identity['full_checkpoint_byte_identical']
    for name, expected in plan['authenticated_inputs'].items():
        assert file_hash(name) == expected, name
    labels = ['parent', *(f'resumed-control-{end}' for end in spec['endpoints'])]
    rows = [r for r in partial['assessments'] if r['label'] in labels]
    sessions = [s for s in partial['sessions'] if s['arm'] == 'resumed-control']
    assert [r['label'] for r in rows] == labels and len(sessions) == 2
    original = plan['original_completion']
    final = Path(rows[-1]['checkpoint'])
    assert file_hash(original['checkpoint']) == file_hash(final) == original['checkpoint_sha256']
    assert identity['original_sha256'] == identity['resumed_sha256'] == original['checkpoint_sha256']
    commands = read(root / 'commands/commands.json')
    expected_commands, raw_hashes = [], {}
    previous = Path(spec['checkpoint'])
    for index, row in enumerate(rows):
        checkpoint = Path(row['checkpoint'])
        assert file_hash(checkpoint) == row['checkpoint_sha256']
        assert counters(checkpoint) == row['state']['counters']
        meta, extra = policy_checkpoint(checkpoint)
        assert hashlib.sha256(struct.pack(f'<{len(extra)}Q', *extra)).hexdigest() == row['state']['replay_payload_sha256']
        if index:
            end = spec['endpoints'][index - 1]
            assert meta[24] == end and sessions[index - 1]['endpoint'] == end
            assert read(checkpoint.parent / 'session.json') == sessions[index - 1]['native_session']
            expected_commands.append(list(map(str, ['live', '--resume', previous, '--curriculum',
                evaluation['schedule'], '--out', checkpoint.parent, '--updates', end, '--lr', .0003,
                '--prompt', 'The bird ', '--log-every', 2048, '--save-every', 8192])))
            previous = checkpoint
        directory = root / 'assessments' / row['label']
        roles = [('validation', n) for n in evaluation['validation_books']]
        roles += [('training_retention', n) for n in evaluation['training_retention_books']]
        roles += [('new_stage_training', n) for n in plan['additional_books']]
        assert [(b['role'], b['book']) for b in row['books']] == roles
        e, g = evaluation['evaluation'], evaluation['generation']
        for book, (_, ident) in zip(row['books'], roles):
            output = directory / f'book-{ident}.json'
            assert {k: v for k, v in book.items() if k not in ('book', 'role')} == read(output)
            assert book['evaluated_bytes'] == 65536 and book['step'] == meta[7]
            expected_commands.append(list(map(str, ['evaluate', '--checkpoint', checkpoint, '--data',
                Path(evaluation['prepared']) / f'{ident}.txt', '--batch', e['batch'], '--context', e['context'],
                '--batches', e['batches'], '--output', output])))
            raw_hashes[output.as_posix()] = file_hash(output)
        assert len(row['samples']) == len(g['prompts']) == 4
        for n, (sample, prompt) in enumerate(zip(row['samples'], g['prompts'])):
            output = directory / f'sample-{n}.txt'
            raw = output.read_bytes()
            assert raw.startswith(prompt.encode()) and len(raw) == len(prompt.encode()) + g['bytes']
            assert raw.hex() == sample['output_hex'] and file_hash(output) == sample['sha256']
            assert raw.decode('utf-8', errors='replace') == sample['output_utf8']
            expected_commands.append(list(map(str, ['sample', '--checkpoint', checkpoint, '--prompt', prompt,
                '--tokens', g['bytes'], '--seed', g['seed'], '--temperature', g['temperature'],
                '--top-k', g['top_k'], '--graph', '--output', output])))
            raw_hashes[output.as_posix()] = file_hash(output)
        validation = [b['loss_nats_per_byte'] for b in row['books'] if b['role'] == 'validation']
        assert sum(validation) / 4 == row['validation_mean_nats_per_byte']
        assert row['checkpoint_unchanged'] and not row['reserved_tests_scored']
    assert len(expected_commands) == 38
    assert [c[1:] for c in commands[:38]] == expected_commands
    assert all(Path(c[0]).resolve() == Path('build/synapticgenesis.exe').resolve() for c in commands[:38])
    for n in range(1, 39):
        path = root / 'commands' / f'command-{n:03d}.log'
        assert path.exists()
        raw_hashes[path.as_posix()] = file_hash(path)
    for row, original_row in [(rows[0], plan['parent_assessment']),
                               (rows[-1], read('reports/prose-105m-complete.json')['assessments'][-1])]:
        assert [b for b in row['books'] if b['role'] != 'new_stage_training'] == original_row['books']
        assert row['samples'] == original_row['samples']
        assert row['validation_mean_nats_per_byte'] == original_row['validation_mean_nats_per_byte']
    write(out, dict(passed=True, scope='Completed original-rate restart control only',
        protocol_sha256=file_hash(root / 'protocol.json'), experiment_source_commit=plan['source_commit'],
        original_completion=original, control_identity=identity, assessments=rows, sessions=sessions,
        native_commands_verified=38, native_learning_commands=2, native_assessment_commands=36,
        completed_command_prefix=commands[:38], raw_artifact_sha256=raw_hashes,
        parent_and_final_original_assessments_exact=True, checkpoint_files_unchanged=True,
        publication_script_sha256=file_hash(Path(__file__)), publication_native_commands=0,
        quarter_rate_results_included=False, reproduction_admitted=False, reserved_tests_scored=False,
        limit='Exact native restart and recorded-output evidence on one 105M run. '
              'The paired lower-rate outcome and improved learning remain unproven in this snapshot.'))
    print('Published exact 105M restart control: 38 completed native commands verified.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('runs/prose-retention-lr'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    publish(args.root, args.out)
