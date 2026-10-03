"""Read-only verification shared by partial and complete retention publications."""
import hashlib
from pathlib import Path
import struct

from experiment_checkpoint import policy_checkpoint
from native_experiment import read
from prose_founder import file_hash
from prose_size_comparison import counters


def checkpoint_state(path):
    meta, extra = policy_checkpoint(path)
    with path.open('rb') as stream:
        stream.seek(256)
        hp = list(struct.unpack('<8f', stream.read(32)))
    assert meta[1] == 6
    return dict(counters=counters(path), hyperparameters=hp, cell=meta[1], seed=meta[10],
        batch=meta[5], chunk=meta[6], fast=meta[16],
        cursor_and_rng=list(meta[19:22]) + [meta[23], meta[25]],
        speech_policy=list(meta[26:30]), graph=extra[8], replay_every=extra[2],
        replay_capacity=extra[3],
        replay_payload_sha256=hashlib.sha256(struct.pack(f'<{len(extra)}Q', *extra)).hexdigest())


def verify_prefix(root, plan, rows, sessions):
    spec, evaluation = plan['specification'], plan['evaluation']
    labels = ['parent', *(f"{a['name']}-{end}" for a in spec['arms'] for end in spec['endpoints'])]
    assert 1 <= len(rows) <= len(labels) and [r['label'] for r in rows] == labels[:len(rows)]
    assert len(sessions) == len(rows) - 1
    expected_commands, raw_hashes = [], {}
    previous = {}
    arm_rates = {a['name']: a['learning_rate'] for a in spec['arms']}
    for index, row in enumerate(rows):
        checkpoint = Path(row['checkpoint'])
        assert file_hash(checkpoint) == row['checkpoint_sha256']
        actual = checkpoint_state(checkpoint)
        assert actual == row['state']
        end = actual['counters']['online_updates']
        assert all(actual['counters'][k] == v for k, v in plan['expected_exposure'][str(end)].items())
        if index:
            arm, session = row['arm'], sessions[index - 1]
            assert row['label'] == f'{arm}-{end}' and row['learning_rate'] == arm_rates[arm]
            assert session['arm'] == arm and session['endpoint'] == end
            assert read(checkpoint.parent / 'session.json') == session['native_session']
            prior = previous.get(arm, Path(spec['checkpoint']))
            expected_commands.append(list(map(str, ['live', '--resume', prior, '--curriculum',
                evaluation['schedule'], '--out', checkpoint.parent, '--updates', end, '--lr', arm_rates[arm],
                '--prompt', spec['prompt'], '--log-every', spec['log_every'], '--save-every', spec['save_every']])))
            previous[arm] = checkpoint
            rate = struct.unpack('<f', struct.pack('<f', arm_rates[arm]))[0]
            assert actual['hyperparameters'][0] == actual['hyperparameters'][7] == rate
            assert actual['hyperparameters'][1:7] == rows[0]['state']['hyperparameters'][1:7]
        directory = root / 'assessments' / row['label']
        assert all(row[k] == v for k, v in read(directory / 'result.json').items())
        roles = [('validation', n) for n in evaluation['validation_books']]
        roles += [('training_retention', n) for n in evaluation['training_retention_books']]
        roles += [('new_stage_training', n) for n in plan['additional_books']]
        assert [(b['role'], b['book']) for b in row['books']] == roles
        e, g = evaluation['evaluation'], evaluation['generation']
        for book, (_, ident) in zip(row['books'], roles):
            output = directory / f'book-{ident}.json'
            assert {k: v for k, v in book.items() if k not in ('book', 'role')} == read(output)
            assert book['evaluated_bytes'] == 65536 and book['step'] == actual['counters']['global_updates']
            expected_commands.append(list(map(str, ['evaluate', '--checkpoint', checkpoint, '--data',
                Path(evaluation['prepared']) / f'{ident}.txt', '--batch', e['batch'], '--context', e['context'],
                '--batches', e['batches'], '--output', output])))
            raw_hashes[output.as_posix()] = file_hash(output)
        assert len(row['samples']) == len(g['prompts']) == 4
        for n, (sample, prompt) in enumerate(zip(row['samples'], g['prompts'])):
            output = directory / f'sample-{n}.txt'
            raw = output.read_bytes()
            assert raw.startswith(prompt.encode()) and len(raw) == len(prompt.encode()) + g['bytes']
            assert sample['prompt'] == prompt and sample['generated_bytes'] == g['bytes']
            assert raw.hex() == sample['output_hex'] and file_hash(output) == sample['sha256']
            assert raw.decode('utf-8', errors='replace') == sample['output_utf8']
            expected_commands.append(list(map(str, ['sample', '--checkpoint', checkpoint, '--prompt', prompt,
                '--tokens', g['bytes'], '--seed', g['seed'], '--temperature', g['temperature'],
                '--top-k', g['top_k'], '--graph', '--output', output])))
            raw_hashes[output.as_posix()] = file_hash(output)
        values = [b['loss_nats_per_byte'] for b in row['books'] if b['role'] == 'validation']
        assert sum(values) / 4 == row['validation_mean_nats_per_byte']
        assert row['checkpoint_unchanged'] and not row['reserved_tests_scored']
    count = len(expected_commands)
    commands = read(root / 'commands/commands.json')[:count]
    assert [c[1:] for c in commands] == expected_commands
    assert all(Path(c[0]).resolve() == Path('build/synapticgenesis.exe').resolve() for c in commands)
    for n in range(1, count + 1):
        path = root / 'commands' / f'command-{n:03d}.log'
        assert path.exists()
        raw_hashes[path.as_posix()] = file_hash(path)
    return commands, raw_hashes
