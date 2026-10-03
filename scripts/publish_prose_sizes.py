"""Verify the completed native size comparison and publish its exact result bytes."""
import argparse
import hashlib
from pathlib import Path
import struct

from experiment_checkpoint import policy_checkpoint
from native_experiment import read
from prose_founder import file_hash, write
from prose_size_comparison import counters


def publish(root, report, audit_path):
    if report.exists() or audit_path.exists():
        raise ValueError('Publication outputs already exist')
    comparison_path = root / 'comparison.json'
    result = read(comparison_path)
    assert result['complete'] and result['matched_source_replay_and_speech_counters']
    assert not result['reserved_tests_scored'] and not result['reproduction_admitted']
    assert result['protocol'] == read(root / 'protocol.json')
    assert {(r['profile'], r['stage']) for r in result['rows']} == {
        (p, s) for p in ('2m', '27m', '105m') for s in (1, 2, 3, 4)}
    plan = read(result['protocol']['assessment_plan'])
    spec, e, g = plan['specification'], plan['specification']['evaluation'], plan['specification']['generation']
    for name, expected in plan['authenticated_inputs'].items():
        assert file_hash(name) == expected, name
    cases, total, replay_by_stage = [], 0, {}
    for row in result['rows']:
        directory = root / f'{row["profile"]}-stage-{row["stage"]}'
        local = read(directory / 'result.json')
        assert all(row[k] == v for k, v in local.items())
        checkpoint = Path(row['checkpoint'])
        assert file_hash(checkpoint) == row['checkpoint_sha256']
        assert counters(checkpoint) == row['exposure_counters']
        meta, extra = policy_checkpoint(checkpoint)
        assert extra[3] == 1024 and meta[24] == spec['stage_endpoints'][row['stage'] - 1]
        assert list(meta[2:5]) == spec['profiles'][row['profile']]
        digest = hashlib.sha256(struct.pack(f'<{len(extra)}Q', *extra)).hexdigest()
        replay_by_stage.setdefault(row['stage'], set()).add(digest)
        commands = read(directory / 'commands.json')
        wanted = []
        book_roles = [('validation', n) for n in spec['validation_books']]
        book_roles += [('training_retention', n) for n in spec['training_retention_books']]
        assert [(b['role'], b['book']) for b in row['books']] == book_roles
        for book, (role, ident) in zip(row['books'], book_roles):
            output = directory / f'book-{ident}.json'
            expected = ['evaluate', '--checkpoint', checkpoint,
                        '--data', Path(spec['prepared']) / f'{ident}.txt',
                        '--batch', e['batch'], '--context', e['context'], '--batches', e['batches'], '--output', output]
            wanted.append(list(map(str, expected)))
            assert {k: v for k, v in book.items() if k not in ('book', 'role')} == read(output)
            assert book['evaluated_bytes'] == 65536 and book['step'] == meta[7]
        for index, (sample, prompt) in enumerate(zip(row['samples'], g['prompts'])):
            output = directory / f'sample-{index}.txt'
            wanted.append(list(map(str, ['sample', '--checkpoint', checkpoint, '--prompt', prompt,
                '--tokens', g['bytes'], '--seed', g['seed'], '--temperature', g['temperature'],
                '--top-k', g['top_k'], '--graph', '--output', output])))
            raw = output.read_bytes()
            assert sample['prompt'] == prompt and raw.startswith(prompt.encode())
            assert len(raw) == len(prompt.encode()) + 256 and sample['generated_bytes'] == 256
            assert raw.hex() == sample['output_hex'] and file_hash(output) == sample['sha256']
            assert raw.decode('utf-8', errors='replace') == sample['output_utf8']
        assert len(row['samples']) == 4 and len(commands) == len(wanted) == 10
        assert [c[1:] for c in commands] == wanted
        assert all(Path(c[0]).resolve() == Path('build/synapticgenesis.exe').resolve() for c in commands)
        values = [b['loss_nats_per_byte'] for b in row['books'] if b['role'] == 'validation']
        assert sum(values) / 4 == row['validation_mean_nats_per_byte']
        cases.append(dict(profile=row['profile'], stage=row['stage'], checkpoint_sha256=row['checkpoint_sha256'],
            result_sha256=file_hash(directory / 'result.json'), command_journal_sha256=file_hash(directory / 'commands.json'),
            replay_policy_sha256=digest, native_assessment_commands=10,
            exact_declared_arguments=True, all_raw_results_match=True))
        total += len(commands)
    assert total == 120 and len(cases) == 12 and all(len(v) == 1 for v in replay_by_stage.values())
    for stage in range(1, 5):
        entries = [r['exposure_counters'] for r in result['rows'] if r['stage'] == stage]
        for key in ('online_updates', 'global_updates', 'observed_pairs', 'generated_bytes', 'replay_updates', 'replay_pairs'):
            assert len({r[key] for r in entries}) == 1
    report.write_bytes(comparison_path.read_bytes())
    assert file_hash(report) == file_hash(comparison_path)
    write(audit_path, dict(passed=True, native_size_comparison_complete=True, cases=cases,
        native_assessment_commands=total, book_evaluations=72, raw_generations=48,
        matched_source_replay_and_speech_counters=True, identical_replay_payloads_across_sizes=True,
        comparison_sha256=file_hash(comparison_path), published_report_sha256=file_hash(report),
        exact_result_copy=True, audit_script_sha256=file_hash(Path(__file__)),
        new_native_commands=0, model_updates=0, reserved_tests_scored=False,
        limit='Verification of recorded native execution and immutable artifacts; not a fresh numerical '
              'oracle, extra seed, new benchmark or proof of useful general language.'))
    print('Published all three sizes: 120 native assessment journals, 72 book results and 48 raw generations verified.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('runs/prose-size-panel'))
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    args = parser.parse_args()
    publish(args.root, args.report, args.audit)
