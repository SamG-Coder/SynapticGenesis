"""Report the measured allocation change and preserved learned continuations."""
import argparse
from pathlib import Path
import re
import subprocess

from native_experiment import read, sha, write


def summarize(args):
    executable = sha(args.build/'synapticgenesis.exe')
    learned = read(args.learned/'result.json')
    ordinary = read(args.ordinary/'result.json')
    large = read(args.large/'result.json')
    cli = read(args.cli/'result.json')
    population = read(args.population/'result.json')
    frozen = read(args.build/'teacher-replay-test-results/frozen-model.json')
    native = read(args.build/'teacher-replay-test-results/result.json')
    for result in (learned, ordinary, large, cli, population, frozen, native):
        assert result['passed']
    for result in (ordinary, large, cli, population):
        assert result['executable_sha256'] == executable
    assert learned['protocol']['new_executable_sha256'] == executable
    assert len(learned['records']) == 9 and learned['native_commands'] == 18
    assert len(frozen['cases']) == 36 and frozen['graph_cases'] == 12
    assert ordinary['numerical_fixture_files_identical'] == 67
    assert len(ordinary['continuations']) == 3
    earlier = read('reports/live-teachers-validation.json')
    assert learned['protocol']['old_executable_sha256'] == earlier['executable_sha256']
    old_large = earlier['larger_model_teaching_smoke']
    large_pairs = []
    for record in large['records']:
        original = next(r for r in old_large['records'] if r['model'] == record['model'])
        assert record['checkpoint_sha256'] == original['checkpoint_sha256']
        assert record['teacher_extra_gpu_bytes'] < original['teacher_extra_gpu_bytes']
        large_pairs.append(dict(model=record['model'], checkpoint_sha256=record['checkpoint_sha256'],
                                complete_checkpoint_identical=True,
                                original_teacher_gpu_bytes=original['teacher_extra_gpu_bytes'],
                                frozen_teacher_gpu_bytes=record['teacher_extra_gpu_bytes']))
    log = args.build/'Testing/Temporary/LastTest.log'
    log_text = log.read_text()
    tests = re.findall(r'^\d+/\d+ Test: (.+)$', log_text, re.MULTILINE)
    assert len(tests) == 18 and log_text.count('Test Passed.') == 18 and 'Test Failed.' not in log_text
    memory_log = args.memcheck.read_text()
    assert 'ERROR SUMMARY: 0 errors' in memory_log and 'PASS frozen model: 36' in memory_log
    oracle = read(args.build/'distillation-test-results/independent-oracle.json')
    failures = [dict(fixture=r['fixture'], **r['failed_adam_tolerance'])
                for r in oracle['records'] if not r['passed'] and 'failed_adam_tolerance' in r]
    assert not oracle['passed'] and failures == earlier['retained_strict_failures']
    # The extraction must keep the forward computation literally unchanged.
    original_main = subprocess.check_output(
        ['git', 'show', args.source_base+':src/spike_lm.cu'], text=True, encoding='utf-8')
    current_main = Path('src/spike_lm.cu').read_text(encoding='utf-8')
    model_source = Path('src/model.cuh').read_text(encoding='utf-8')
    assert original_main.split('struct Cache {')[0] == current_main.split('#include "model.cuh"')[0]
    assert original_main.split('#include "model_memory.cuh"', 1)[1] == current_main.split('#include "model_memory.cuh"', 1)[1]
    def forward(text):
        return text.split('    void forward_device(', 1)[1].split('    float forward(', 1)[0]
    assert forward(original_main) == forward(model_source)
    files = sorted(Path('src').glob('*.cu'))+sorted(Path('src').glob('*.cuh'))
    files += [Path(p) for p in ('CMakeLists.txt', 'tests/frozen_teacher_compatibility.py',
                               'scripts/summarize_frozen_teachers.py')]
    evidence = dict(learned=args.learned, ordinary=args.ordinary, large=args.large,
                    cli=args.cli, population=args.population)
    result = dict(stage='Frozen teacher buffers and shared model ownership module',
                  source_base=args.source_base, executable_sha256=executable,
                  source_sha256={p.as_posix(): sha(p) for p in files},
                  native_suites=tests, ctest_log_sha256=sha(log),
                  production_forward_source_unchanged=True, native_frozen_model=frozen,
                  native_teacher_replay=native, learned_checkpoint_compatibility=learned,
                  ordinary_compatibility=ordinary, two_teacher_large_compatibility=large_pairs,
                  standalone_cli=cli, population_cli=population,
                  cuda_memcheck=dict(passed=True, errors=0, log_sha256=sha(args.memcheck),
                                     scope='Complete native teacher-replay-test including frozen allocation cases'),
                  retained_strict_failures=failures, all_independent_numerical_checks_passed=False,
                  evidence={key: dict(path=value.as_posix(), sha256=sha(value/'result.json'))
                            for key, value in evidence.items()},
                  limits=['Explicit buffer bytes exclude CUDA and cuBLAS overhead; admission retains its reserve.',
                          'Teachers retain all caches used by the existing forward kernels.',
                          'Live speech views can learn short observed tails and retain their learning buffers.',
                          'Allocation savings do not establish a speedup, energy benefit or improved language quality.',
                          'The existing strict independent cold Adam failure is unchanged.'])
    write(args.output, result)
    print(args.output, 'allocation reduction verified; learned checkpoints unchanged; numerical failure retained')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--build', type=Path, default=Path('build'))
    for name in ('learned', 'ordinary', 'large', 'cli', 'population', 'memcheck'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--source-base', default='d437063bf515a4d4d5ded6fd76ce85efd231d902')
    p.add_argument('--output', type=Path, default=Path('reports/frozen-teachers-validation.json'))
    summarize(p.parse_args())
