"""Fresh-process CPU gate using the frozen candidate's own acceptance readers."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def verify(root, expected_protocol, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    require(not output.exists(), 'Gate verification output already exists')
    require(sha(root / 'protocol.json') == expected_protocol, 'Acceptance protocol changed')
    plan = read(root / 'protocol.json')
    require(plan['version'] == 'membrane-acceptance-v1' and
            plan['status'] == 'declared_before_gpu_execution', 'Unexpected acceptance declaration')
    require(not (root / 'failure.json').exists() and (root / 'result.json').is_file(),
            'Candidate acceptance has not completed successfully')
    result, execution = read(root / 'result.json'), read(root / 'execution.json')
    require(result['complete'] is True and result['passed'] is True and
            result['protocol_sha256'] == expected_protocol and execution['phase'] == 'complete' and
            execution['protocol_sha256'] == expected_protocol and execution['native_work_started'] is True,
            'Incomplete candidate acceptance')
    # A fresh interpreter avoids mixing candidate modules with main-workspace imports.
    sys.path.insert(0, str(Path(plan['root']) / 'scripts'))
    import membrane_acceptance as accepted
    require(accepted.ROOT == Path(plan['root']), 'Acceptance reader came from another checkout')
    accepted.authenticate(plan['authenticated_inputs'])
    exe = plan['executables']
    wanted = accepted.command_plan(root, exe['python'], exe['ctest'], exe['old'], exe['new'], exe['probe'])
    require(plan['commands'] == wanted == read(root / 'commands.json'), 'Acceptance command journal differs')
    completed = read(root / 'completed-commands.json')
    require(completed == result['commands'] and len(completed) == len(wanted) == 20 and
            execution['completed'] == 20, 'Acceptance command coverage differs')
    artifacts = {}
    for index, (actual, intended) in enumerate(zip(completed, wanted), 1):
        log = root / f'command-{index:03d}.log'
        require(actual['label'] == intended['label'] and actual['returncode'] == 0 and
                sha(log) == actual['log_sha256'], 'Acceptance command result differs')
        artifacts[str(log)] = sha(log)
    observed = accepted.collect_results(root, plan)
    require(all(result[k] == v for k, v in observed.items()), 'Acceptance artifacts differ from published completion')
    require(sha(root / 'predecessor-verified.json') == result['predecessor_verification_sha256'],
            'Acceptance predecessor verification changed')
    previous = accepted.verify_predecessor(plan['workspace'], plan['predecessor'], plan['predecessor_protocol_sha256'])
    require(previous == read(root / 'predecessor-verified.json'), 'Completed 411M evidence changed')
    for name in ('protocol.json', 'result.json', 'execution.json', 'commands.json',
                 'completed-commands.json', 'predecessor-verified.json'):
        artifacts[str(root / name)] = sha(root / name)
    verified = dict(passed=True, acceptance_protocol_sha256=expected_protocol,
        acceptance_result_sha256=sha(root / 'result.json'), candidate_runtime=exe['new'],
        candidate_runtime_sha256=sha(exe['new']), command_count=len(wanted),
        native_checks_and_legacy_compatibility_revalidated=True,
        whole_411m_predecessor_revalidated=True, artifact_sha256=artifacts,
        native_commands_launched=0, reproduction_admitted=False)
    output.write_text(json.dumps(verified, indent=2) + '\n', encoding='utf-8', newline='\n')
    print('Verified completed candidate CUDA acceptance and entire 411M predecessor; no model execution.')
    return verified


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--acceptance', type=Path, required=True)
    parser.add_argument('--protocol-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    verify(args.acceptance, args.protocol_sha256, args.output)
