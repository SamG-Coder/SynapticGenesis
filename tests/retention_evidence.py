"""Check shared publication verification against completed native artifacts."""
import argparse
import copy
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from native_experiment import read
from prose_founder import file_hash, write
from publish_retention_control import publish
from retention_evidence import verify_prefix


def audit(out):
    if out.exists():
        raise ValueError('Use a fresh audit directory')
    out.mkdir(parents=True)
    root = Path('runs/prose-retention-lr')
    reproduced = out / 'control.json'
    publish(root, reproduced)
    old, current = read('reports/prose-retention-control.json'), read(reproduced)
    old.pop('publication_script_sha256')
    current.pop('publication_script_sha256')
    assert old == current
    plan, partial = read(root / 'protocol.json'), read(root / 'partial.json')
    rows, sessions = partial['assessments'][:4], partial['sessions'][:3]
    assert len(rows) == 4 and rows[-1]['label'] == 'quarter-rate-155748'
    commands, _ = verify_prefix(root, plan, rows, sessions)
    assert len(commands) == 51
    failures = []
    for name in ('wrong_order', 'changed_counter', 'changed_replay', 'wrong_rate', 'wrong_role', 'wrong_session'):
        changed, runs = copy.deepcopy(rows), copy.deepcopy(sessions)
        if name == 'wrong_order': changed[0]['label'] = 'quarter-rate-155748'
        elif name == 'changed_counter': changed[0]['state']['counters']['observed_pairs'] += 1
        elif name == 'changed_replay': changed[-1]['state']['replay_payload_sha256'] = '0' * 64
        elif name == 'wrong_rate': changed[-1]['learning_rate'] = .0003
        elif name == 'wrong_role': changed[0]['books'][0]['role'] = 'training_retention'
        elif name == 'wrong_session': runs[0]['endpoint'] += 1
        try:
            verify_prefix(root, plan, changed, runs)
        except AssertionError:
            failures.append(name)
        else:
            raise AssertionError('Changed publication evidence accepted: ' + name)
    result = dict(passed=True, control_publication_values_unchanged=True,
        unchanged_control_native_commands=38, completed_native_prefix_verified=51,
        rejected_cases=failures, new_native_commands=0, model_updates=0,
        protocol_sha256=file_hash(root / 'protocol.json'),
        implementation_sha256={p.as_posix(): file_hash(p) for p in (Path(__file__),
            Path('scripts/retention_evidence.py'), Path('scripts/publish_retention_control.py'))},
        scope='Archived control plus completed quarter-rate halfway artifacts; no final candidate claim.')
    write(out / 'result.json', result)
    print('Control publication unchanged; 51 completed commands and six altered-evidence rejections verified.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    audit(args.out)
