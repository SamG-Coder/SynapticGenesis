"""Explicit admission for new book preparation and learning experiments."""
import hashlib
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
SELECTION = REPO / 'data/training-selection.json'


def selection():
    return json.loads(SELECTION.read_text(encoding='utf-8'))


def require_active_edition(version):
    policy = selection()
    if version in policy['retired_editions']:
        raise ValueError(f'Edition {version} is retired for new preparation and learning. '
                         f"Use an active edition from {SELECTION.name}; historical reports remain available.")
    if version not in policy['active_editions']:
        raise ValueError(f'Edition {version} has not been admitted in {SELECTION.name}')
    return policy['active_editions'][version]


def require_training_spec(path):
    raw = Path(path).read_bytes()
    spec = json.loads(raw)
    record = require_active_edition(spec['version'])
    upstream = spec.get('upstream_repository', '').casefold().rstrip('/').removesuffix('.git')
    if upstream in selection()['excluded_source_repositories']:
        raise ValueError('Source repository is excluded from new training selections')
    if hashlib.sha256(raw).hexdigest() != record['sha256']:
        raise ValueError('Source selection differs from the admitted edition; review it as a new version')
    return spec
