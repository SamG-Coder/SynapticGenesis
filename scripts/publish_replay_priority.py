"""Publish authenticated diagnostic evidence without checkpoint payloads."""
import argparse
import json
from pathlib import Path
import shutil

from native_experiment import read, sha


def publish(root, fixture, diagnosis):
    # Keep the original Windows captures before producing portable public JSON.
    for name in ('execution', 'summary'):
        capture = root / (name + '.json')
        if not capture.exists():
            shutil.copyfile(f'reports/replay-priority-{name}.json', capture)
    comparison = read(root / 'comparison.json')
    execution = read(root / 'execution.json')
    summary = read(root / 'summary.json')
    checked = read(fixture / 'result.json')
    diagnosed = read(diagnosis / 'result.json')
    assert comparison['complete'] and execution['execution_passed'] and checked['passed']
    assert execution['comparison_sha256'] == summary['comparison_sha256'] == sha(root / 'comparison.json')
    assert diagnosed['execution_sha256'] == summary['execution_sha256'] == sha(root / 'execution.json')
    assert not diagnosed['original_strict_cpu_check_passed'] and not execution['all_cpu_scores_passed']
    for section in ('authenticated_inputs', 'source_sha256'):
        assert all(sha(path) == identity for path, identity in comparison['protocol'][section].items())
    assert all(sha(path) == identity for path, identity in diagnosed['source_sha256'].items())
    log = fixture / 'memcheck.log'
    assert 'ERROR SUMMARY: 0 errors' in log.read_text()
    ctest = Path('build/Testing/Temporary/LastTest.log')
    assert ctest.read_text().count('Test Passed.') == 18
    validation = dict(native_ctest_suites_passed=18, native_ctest_log_sha256=sha(ctest),
        fixture=checked, cuda_memcheck_errors=0, cuda_memcheck_log_sha256=sha(log),
        cuda_memcheck_scope='Ten source updates with the measured observer, auditing, snapshots and a stage boundary.',
        learned_execution_passed=True, learned_cpu_scores_all_passed=False,
        limits='Isolation and execution are verified in the stated cases. No learning-quality claim; '
               'one of 192 learned CPU score checks exceeds the unchanged tolerance.')
    records = {}

    def publish_json(name, value, capture=None):
        path = Path(f'reports/replay-priority-{name}.json')
        path.write_bytes((json.dumps(value, indent=2) + '\n').encode('utf-8'))
        records[path.as_posix()] = dict(sha256=sha(path), bytes=path.stat().st_size)
        if capture is not None:
            records[path.as_posix()]['original_capture_sha256'] = sha(capture)
        return sha(path)

    comparison_hash = publish_json('comparison', comparison, root / 'comparison.json')
    execution['comparison_sha256'] = comparison_hash
    execution_hash = publish_json('execution', execution, root / 'execution.json')
    summary.update(comparison_sha256=comparison_hash, execution_sha256=execution_hash)
    publish_json('summary', summary, root / 'summary.json')
    diagnosed['execution_sha256'] = execution_hash
    publish_json('diagnosis', diagnosed, diagnosis / 'result.json')
    publish_json('validation', validation)
    for base in comparison['protocol']['bases']:
        seed = base['seed']
        capture = root / f'{seed}-0-measured/scores.jsonl'
        path = Path(f'reports/replay-priority-scores-{seed}.jsonl')
        path.write_bytes(capture.read_bytes().replace(b'\r\n', b'\n'))
        records[path.as_posix()] = dict(sha256=sha(path), bytes=path.stat().st_size,
                                      original_capture_sha256=sha(capture))
    document = Path('docs/replay-priority-diagnostic.md')
    records[document.as_posix()] = dict(sha256=sha(document), bytes=document.stat().st_size)
    publish_json('publication', dict(complete=True, files=dict(records),
        serialization='Published JSON and JSONL use LF. Cross-report hashes refer to the published bytes. '
                      'Original capture hashes preserve the identities of unmodified local Windows results; '
                      'measurements, source/executable identities and failed checks are unchanged.',
        source_sha256={'scripts/publish_replay_priority.py': sha(__file__)}, generated_text_targets=False,
        changed_replay_selection=False, checkpoint_payloads_published=False))
    print('Published diagnostic records, raw candidate journals and verification limits.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--diagnosis', type=Path, required=True)
    args = parser.parse_args()
    publish(args.root, args.fixture, args.diagnosis)
