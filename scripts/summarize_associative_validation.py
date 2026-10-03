"""Collect numerical and lifecycle evidence for architecture 6."""
import argparse
from pathlib import Path
import subprocess

from native_experiment import read, sha, write


def summarize(root, build_log, output):
    root = Path(root)
    assert '100% tests passed out of 15' in Path(build_log).read_text(encoding='utf-8-sig')
    files = {
        'native': 'build/associative-test-results/native.json',
        'causal_memory': 'build/associative-test-results/association.json',
        'inheritance_and_resources': 'build/evolution-test-results/native.json',
        'curriculum': 'build/curriculum-test-results/native.json',
        'stage_replay': 'build/stage-replay-test-results/native.json',
        'repeatability': 'build/reduction-test-results/native.json',
    }
    for name in ('curriculum_cli', 'curriculum_extension_cli', 'probes_cli', 'binding_cli',
                 'burn_policy', 'population_live_cli'):
        files[name] = (root/name/'result.json').as_posix()
    files['population_scarcity'] = (root/'population_cli-v2/result.json').as_posix()
    evidence = {name: read(path) for name, path in files.items()}
    assert all(value['passed'] for value in evidence.values())
    fixtures = ['test-results', 'live-test-results', 'adaptive-test-results', 'trace-test-results',
                'gated-test-results', 'selective-test-results', 'associative-test-results']
    fixtures += ['feedback-test-results/'+cell for cell in
                 ('lif', 'alif', 'trace', 'gated', 'selective', 'associative')]
    oracles = {name: read(Path('build')/name/'oracle.json') for name in fixtures}
    assert all(value['passed'] for value in oracles.values())
    legacy = Path('runs/associative-legacy-selective')
    legacy_fixtures = list(legacy.glob('*.f32'))
    assert legacy_fixtures and all(p.read_bytes() == (Path('build/selective-test-results')/p.name).read_bytes()
                                   for p in legacy_fixtures)
    source_files = [*Path('src').glob('*.cu'), *Path('src').glob('*.cuh'),
                    Path('tests/oracle.py'), Path('tests/probes_cli.py'), Path('tests/associative_reference.py')]
    result = dict(passed=True, native_test_suites=15, independent_gradient_fixtures=len(oracles),
                  executable_sha256=sha('build/synapticgenesis.exe'),
                  prior_selective_fixtures_byte_identical=True,
                  archived_executable_sha256=sha('runs/legacy-b0/synapticgenesis.exe'),
                  source_parent_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  source_sha256={p.as_posix(): sha(p) for p in sorted(source_files)},
                  evidence=evidence, oracles=oracles,
                  source_reports={name: dict(path=path, sha256=sha(path)) for name, path in files.items()},
                  test_fixture_correction='The old fixed 1 MiB scarcity test correctly allowed zero births for '
                      'the larger cell. The test now derives a small budget from resident costs, admits some '
                      'offspring, then verifies exhaustion. The zero-budget and old-age controls remain.',
                  limits='Correctness and lifecycle evidence only. No claim of language improvement or biological equivalence.')
    write(output, result)
    print(f'{output}: 15 native suites, {len(oracles)} CPU gradient fixtures and all CLI checks passed')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path('runs/associative-validation-v1'))
    p.add_argument('--build-log', type=Path, default=Path('build-associative.log'))
    p.add_argument('--output', type=Path, default=Path('reports/associative-validation.json'))
    a = p.parse_args()
    summarize(a.root, a.build_log, a.output)
