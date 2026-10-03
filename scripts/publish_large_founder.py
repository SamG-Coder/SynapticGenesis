"""Publish a completed 411M comparison using its frozen, read-only verifier."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from prose_founder import file_hash, write


# A fresh interpreter keeps the frozen checkpoint reader separate from newer
# readers imported by other experiments in the main checkout.
WORKER = """
import json
from pathlib import Path
import sys
checkout, workspace, root, protocol_hash = sys.argv[1:]
sys.path.insert(0, str(Path(checkout) / 'scripts'))
from large_founder_verification import verify
from prose_founder import file_hash
evidence = verify(workspace, root, protocol_hash)
sources = {}
scripts = (Path(checkout) / 'scripts').resolve()
for module in list(sys.modules.values()):
    name = getattr(module, '__file__', None)
    if name and Path(name).resolve().is_relative_to(scripts):
        path = Path(name).resolve()
        sources[str(path)] = file_hash(path)
print(json.dumps(dict(evidence=evidence, verifier_sources_sha256=sources)))
"""


def publish(workspace, root, checkout, protocol_hash, report, audit):
    workspace, root, checkout, report, audit = map(
        lambda p: Path(p).resolve(), (workspace, root, checkout, report, audit))
    if report == audit or report.exists() or audit.exists():
        raise ValueError('Use two distinct fresh publication paths')
    if not (root / 'result.json').is_file() or (root / 'failure.json').exists():
        raise ValueError('The full study has not completed successfully')
    if file_hash(root / 'protocol.json') != protocol_hash:
        raise ValueError('Study protocol identity changed')
    command = [sys.executable, '-X', 'utf8', '-c', WORKER, str(checkout),
               str(workspace), str(root), protocol_hash]
    result = subprocess.run(command, cwd=workspace, capture_output=True, check=True)
    verified = json.loads(result.stdout)
    evidence = verified['evidence']
    if not evidence['passed'] or evidence['cuda_executed']:
        raise ValueError('Expected successful read-only completion verification')
    if (evidence['verified_learning_commands'], evidence['verified_assessment_commands']) != (1, 80):
        raise ValueError('Incomplete study command coverage')
    raw = (root / 'result.json').read_bytes()
    if file_hash(root / 'result.json') != evidence['result_sha256']:
        raise ValueError('Completed result changed after verification')
    for path in (report, audit):
        path.parent.mkdir(parents=True, exist_ok=True)
    report.write_bytes(raw)
    if file_hash(report) != evidence['result_sha256']:
        raise ValueError('Published result differs from the verified result')
    write(audit, dict(**verified, exact_result_copy=True,
        published_result_sha256=file_hash(report), protocol_sha256=protocol_hash,
        publisher_sha256=file_hash(__file__), new_native_commands=0,
        limits='Verification and exact publication of one completed exploratory seed. '
               'This does not add another seed, a speed benchmark, useful QA or reproductive admission.'))
    print('Published the complete comparison: one learning command and 80 saved assessments verified.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=Path, default=Path('.'))
    parser.add_argument('--root', type=Path, default=Path('runs/prose-411m-v1'))
    parser.add_argument('--verifier-checkout', type=Path,
                        default=Path('runs/membrane-regularization-worktree'))
    parser.add_argument('--protocol-sha256', required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    args = parser.parse_args()
    publish(args.workspace, args.root, args.verifier_checkout, args.protocol_sha256,
            args.report, args.audit)
