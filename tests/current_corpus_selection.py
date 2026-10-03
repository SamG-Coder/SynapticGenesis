"""Authenticate the current foundation edition and pre-extension learning bases."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from corpus.selection import require_training_spec, selection
from experiment_checkpoint import state_record
from extend_curriculum import read_schedule
from native_experiment import sha


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def check(out):
    policy = selection()
    prepared = Path(policy['default_prepared'])
    manifest = read(prepared / 'manifest.json')
    original = read('data/prepared/foundations-v1/manifest.json')
    assert manifest['outputs'] == original['outputs']
    assert manifest['source_spec_sha256'] == sha(policy['default_sources'])
    assert require_training_spec(prepared / 'source-spec.json')['version'] == manifest['version']
    for role, record in manifest['outputs'].items():
        assert sha(prepared / (role + '.dat')) == record['sha256']
    for source in manifest['sources']:
        assert sha(prepared / (str(source['id']) + '.txt')) == source['clean_sha256']
    retired, retired_count = set(), 0
    for source in policy['retired_editions'].values():
        for record in read(source)['sources']:
            retired.add(record['body_sha256'])
            retired_count += 1
    schedule = Path(policy['continuation_schedule'])
    _, stages = read_schedule(schedule)
    assert sha(schedule) == policy['continuation_schedule_sha256']
    history = read('reports/teacher-retention-language.json')
    admission = history['protocol']['source_admission']
    assert len(stages) == len(admission['editions']) == 4
    for stage, edition in zip(stages, admission['editions']):
        assert sha(stage['source']) == edition['sha256']
        assert not {hashlib.sha256(doc).hexdigest() for doc in stage['content'].split(b'\x1e')} & retired
    assert not {record['clean_sha256'] for record in manifest['sources']} & retired
    def authenticate(records):
        bases = []
        for base in records:
            assert sha(base['checkpoint']) == base['sha256']
            endpoint = base.get('online_updates', 190000)
            previous = next(r for r in history['runs'] if r['seed'] == base['seed']
                            and r['arm'] == 'associative-control' and r['online_updates'] == endpoint)
            assert previous['checkpoint_sha256'] == base['sha256']
            state = state_record(Path(base['checkpoint']))
            assert state['online_updates'] == endpoint and state['parameters'] == 1951624
            bases.append(dict(**base, recorded_historical_checkpoint_identical=True,
                              source_observations=state['online_updates'], parameters=state['parameters']))
        return bases
    bases = authenticate(policy['continuation_bases'])
    diagnostic_bases = authenticate(policy.get('diagnostic_bases', []))
    result = dict(passed=True, selection_sha256=sha('data/training-selection.json'),
        active_foundation_directory=prepared.as_posix(),
        foundation_source_spec_sha256=manifest['source_spec_sha256'],
        foundation_manifest_sha256=sha(prepared / 'manifest.json'), foundation_outputs=manifest['outputs'],
        foundation_outputs_identical_to_original=True,
        training_sources=[dict(id=r['id'], title=r['title'], sha256=r['clean_sha256'])
                          for r in manifest['sources'] if r['split'] == 'train'],
        retired_editions=list(policy['retired_editions']), retired_source_documents_checked=retired_count,
        continuation_schedule_sha256=sha(schedule), continuation_stages_authenticated=4,
        continuation_documents=len(stages[-1]['content'].split(b'\x1e')),
        exact_retired_documents_absent_from_selected_foundations_and_continuation_sources=True,
        continuation_bases=bases, diagnostic_bases=diagnostic_bases,
        no_new_training_performed=True, reserved_tests_not_scored=True,
        provenance='The recorded 160000- and 190000-observation checkpoints precede admission of both retired story editions. '
                   'Checks authenticate the published study, source schedule and checkpoint bytes; later descendants are not selected. '
                   'Earlier experimental artifacts remain historical records.')
    out.write_bytes((json.dumps(result, indent=2) + '\n').encode('utf-8'))
    print(f'Foundation outputs reproduce exactly; {len(bases)} continuation and {len(diagnostic_bases)} diagnostic checkpoints, {len(stages)} source stages authenticated.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    check(parser.parse_args().out)
