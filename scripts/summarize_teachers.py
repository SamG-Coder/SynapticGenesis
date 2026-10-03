"""Publish compact integration evidence without hiding the strict oracle failure."""
import argparse
import re
from pathlib import Path
import subprocess

from native_experiment import read, write, sha


def summarize(args):
    cli = read(args.cli/'result.json')
    population = read(args.population/'result.json')
    compatibility = read(args.compatibility/'result.json')
    large = read(args.large/'result.json')
    native = read(args.build/'teacher-replay-test-results/result.json')
    oracle = read(args.build/'distillation-test-results/independent-oracle.json')
    executable = sha(args.build/'synapticgenesis.exe')
    assert all(r['passed'] for r in (cli, population, compatibility, native, large))
    assert all(r['executable_sha256'] == executable for r in (cli, population, compatibility, large))
    log = args.build/'Testing/Temporary/LastTest.log'
    tests = re.findall(r'^\d+/\d+ Test: (.+)$', log.read_text(), re.MULTILINE)
    assert len(tests) == 18 and log.read_text().count('Test Passed.') == 18 and 'Test Failed.' not in log.read_text()
    race = args.racecheck.read_text()
    assert 'RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)' in race
    failures = [dict(fixture=r['fixture'], **r['failed_adam_tolerance'])
                for r in oracle['records'] if not r['passed'] and 'failed_adam_tolerance' in r]
    assert not oracle['passed'] and len(failures) == 1 and failures[0]['fixture'] == 'cell-6'
    assert failures[0]['max_abs_error'] == 9.20519232749939e-6
    # Capture the evaluated source files, independent of later documentation
    # edits or the report's own Git commit. Never include local checkpoints.
    files = sorted(Path('src').glob('*.cu')) + sorted(Path('src').glob('*.cuh'))
    files += [Path(p) for p in ('CMakeLists.txt', 'scripts/experiment_checkpoint.py',
                               'scripts/inspect_dynamics.py', 'scripts/summarize_teachers.py',
                               'tests/teacher_replay_cli.py', 'tests/population_teachers_cli.py',
                               'tests/teacher_compatibility.py', 'tests/teacher_large_smoke.py',
                               'tests/oracle.py', 'tests/distillation_oracle.py')]
    numeric = [{k: v for k, v in r.items() if k not in ('tensors',)} for r in oracle['records']]
    result = dict(stage='Immutable selected-source teaching in the native shared live learner',
                  base_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  executable_sha256=executable, source_sha256={p.as_posix(): sha(p) for p in files},
                  runtime_integration_passed=True, all_independent_numerical_checks_passed=False,
                  performance_or_language_quality_claim=False,
                  native_suites=tests, ctest_log_sha256=sha(log),
                  native_teacher_replay=native, standalone_cli=cli, population_cli=population,
                  compatibility=compatibility,
                  larger_model_teaching_smoke=large,
                  cuda_racecheck=dict(passed=True, hazards=0, log_sha256=sha(args.racecheck),
                                      kernel_filter='kns=distillation',
                                      scope='Teacher probability and penalty kernels throughout the native replay suite',
                                      unfiltered_run='Stopped early for instrumentation cost; no complete result claimed'),
                  independent_objective_records=numeric, retained_strict_failures=failures,
                  evidence_paths=dict(cli=args.cli.as_posix(), population=args.population.as_posix(),
                                      compatibility=args.compatibility.as_posix(), large=args.large.as_posix(),
                                      racecheck=args.racecheck.as_posix()),
                  limitations=['Synthetic runtime/lifecycle checks; no teacher-assisted language study yet.',
                               'Cold first-Adam-step independent tolerance failure is retained.',
                               'Teachers are frozen resident models; no new efficiency or biological claim.',
                               'The immutable policy has no teacher-replacement or standalone-to-population migration.',
                               'FNV identities are consistency checks, not signatures or complete data provenance.'])
    write(args.output, result)
    print(args.output, 'runtime checks pass; one unchanged independent numerical failure retained')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--build', type=Path, default=Path('build'))
    p.add_argument('--cli', type=Path, required=True)
    p.add_argument('--population', type=Path, required=True)
    p.add_argument('--compatibility', type=Path, required=True)
    p.add_argument('--large', type=Path, required=True)
    p.add_argument('--racecheck', type=Path, required=True)
    p.add_argument('--output', type=Path, default=Path('reports/live-teachers-validation.json'))
    summarize(p.parse_args())
