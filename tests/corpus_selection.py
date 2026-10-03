"""Reject retired book inputs before preparation or native learning can begin."""
import argparse
import importlib
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'scripts'))
from corpus.selection import require_training_spec, selection


def check(out):
    policy = selection()
    accepted, rejected = [], []

    def reject(name, call, reason):
        try:
            call()
        except ValueError as error:
            assert reason in str(error), (name, error)
            rejected.append(dict(case=name, error=str(error)))
        else:
            raise AssertionError(f'Unexpected acceptance: {name}')

    for version, record in policy['active_editions'].items():
        assert require_training_spec(REPO / record['sources'])['version'] == version
        accepted.append(version)
    for version, source in policy['retired_editions'].items():
        reject(version, lambda source=source: require_training_spec(REPO / source), 'retired')

    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        changed = root / 'renamed.json'
        original = json.loads((REPO / policy['default_sources']).read_text())
        for name, updates, reason in [
            ('unreviewed version', {'version': 'unreviewed-edition'}, 'has not been admitted'),
            ('changed sources under admitted version', {'sources': []}, 'differs from the admitted edition'),
        ]:
            changed.write_text(json.dumps({**original, **updates}))
            reject(name, lambda: require_training_spec(changed), reason)
        retired = json.loads((REPO / policy['retired_editions']['selected-early-readers-v1']).read_text())
        changed.write_text(json.dumps({**retired, 'version': original['version']}))
        reject('retired texts relabelled as foundation', lambda: require_training_spec(changed),
               'Source repository is excluded')

        for module_name, edition in [
            ('prepare_early_readers', 'selected-early-readers-v1'),
            ('prepare_reading_breadth', 'selected-reading-breadth-v1'),
        ]:
            module = importlib.import_module(module_name)
            source = REPO / policy['retired_editions'][edition]
            output, cache = root / module_name, root / (module_name + '-cache')
            reject(module_name, lambda: module.prepare(source, cache, output, root / 'unused-protected'), 'retired')
            assert not output.exists() and not cache.exists()

        for module_name in ('adaptation_experiment', 'live_reading_experiment',
                            'reading_breadth_experiment', 'native_demonstration'):
            module = importlib.import_module(module_name)
            # An empty argument object makes any access beyond admission fail.
            # No executable, output directory or model is supplied to the driver.
            reject(module_name, lambda: module.run(SimpleNamespace()), 'retired')

        output, cache = root / 'generic-output', root / 'generic-cache'
        result = subprocess.run([sys.executable, str(REPO / 'scripts/prepare_corpus.py'),
            '--sources', str(REPO / policy['retired_editions']['selected-early-readers-v1']),
            '--raw', str(cache), '--out', str(output)], capture_output=True, text=True)
        assert result.returncode != 0 and 'is retired' in result.stderr
        assert not output.exists() and not cache.exists()
        rejected.append(dict(case='generic corpus CLI', error='Retired edition rejected before output/cache creation'))

    report = dict(passed=True, active_editions=accepted, rejected_cases=rejected,
        rejection_count=len(rejected), no_model_loaded_or_trained=True,
        no_retired_downloads_or_prepared_outputs=True,
        limits='Book admission and four historical experiment entrypoints. The native executable remains a generic file consumer; provenance rules also apply to direct native commands.')
    out.write_bytes((json.dumps(report, indent=2) + '\n').encode('utf-8'))
    print(f'{len(accepted)} admitted editions and {len(rejected)} rejection cases passed; no native model work.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    check(parser.parse_args().out)
