"""Publish the measured CUDA layout change with numerical and timing boundaries."""
import argparse
from pathlib import Path

from native_experiment import read, sha, write


def summarize(root, kernels, output):
    measured = read(root / 'comparison.json')
    protocol = measured['protocol']
    assert measured['passed'] and measured['native_commands'] == 23
    assert len(measured['live']) == 14 and len(measured['decode']) == 6
    assert len(measured['full_reproductions']) == 3
    assert measured['complete_live_checkpoints_identical']
    assert measured['generated_samples_identical']
    assert sha('build/synapticgenesis.exe') == protocol['optimized_executable_sha256']
    assert all(sha(path) == identity for path, identity in protocol['changed_source_sha256'].items())
    for variant, count in [('original', 10), ('optimized', 13)]:
        commands = read(root / variant / 'commands.json')
        assert len(commands) == count
    reference = {row['checkpoint_sha256'] for row in measured['live']}
    assert len(reference) == 1
    for row in measured['live']:
        path = root / row['variant'] / f'live-{row["repeat"]}' / 'latest.ckpt'
        assert sha(path) == row['checkpoint_sha256']
    for row in measured['full_reproductions']:
        assert row['complete_checkpoint_identical']
        assert sha(root / 'optimized' / f'full-{row["seed"]}' / 'latest.ckpt') == row['checkpoint_sha256']

    kernel = read(kernels / 'benchmark.json')
    assert kernel['passed'] and kernel['case_count'] == 10
    assert kernel['rounds'] == 7 and kernel['iterations'] == 100
    assert all(row['gradients_bitwise_identical'] and row['forward_state_identical']
               for row in kernel['cases'])
    build = Path('build-association-final.log')
    assert '100% tests passed out of 16' in build.read_text()
    fixtures = []
    original = Path('runs/association-layout-legacy-fixture')
    current = Path('build/associative-test-results')
    for file in sorted(original.glob('*.f32')):
        assert file.read_bytes() == (current / file.name).read_bytes(), file.name
        fixtures.append(dict(file=file.name, sha256=sha(file), byte_identical=True))
    assert len(fixtures) == 7
    oracles = {name: read(path / 'oracle.json') for name, path in [
        ('unweighted', current), ('weighted', Path('build/feedback-test-results/associative'))]}
    assert all(oracle['passed'] and oracle['cell'] == 6 for oracle in oracles.values())
    profile_log = Path('runs/association-layout-profile32.log')
    assert 'ERR_NVGPUCTRPERM' in profile_log.read_text()
    focused = next(row for row in kernel['cases'] if row['batch'] == 1 and row['context'] == 128)
    result = dict(
        passed=True, runtime=measured, recurrence_kernels=kernel,
        validation=dict(native_suites=16, build_log_sha256=sha(build),
                        associative_float_fixtures=fixtures, independent_cpu_oracles=oracles),
        timing_reduction_percent={key: 100 * (1 - 1 / value)
                                  for key, value in measured['speedups'].items()},
        context_128_kernel_reduction_percent=dict(
            padding_only=100 * (1 - focused['padded_us'] / focused['packed_us']),
            cached_padded=100 * (1 - focused['cached_padded_us'] / focused['packed_us']),
            forward_register=100 * (1 - focused['forward_register_us'] / focused['forward_shared_us'])),
        hardware='NVIDIA GeForce RTX 5080, 16 GB; CUDA 13.3; MSVC 19.51; Windows',
        profiler=dict(counters_available=False, reason='ERR_NVGPUCTRPERM',
                      log_sha256=sha(profile_log), profiling_timings_used=False),
        boundaries=[
            'No new learning-quality claim: repeated timings resume identical learned history.',
            'Three complete 24000-observation trajectories reproduce the published original files exactly.',
            'Forward shared control also includes hoisted inputs and loop unrolling; it is not the exact old kernel.',
            'Archived-executable comparisons are the before/after runtime measurements.',
            'Kernel timing excludes projections, feature normalization, optimizer and host transfer.',
            'Kernel speedup does not equal whole-model speedup or energy improvement.',
            'Hardware counters were unavailable, so memory-layout causation is not established by profiling.',
            'No additional persistent/global model bytes and no checkpoint-format change.',
            'Bitwise agreement is measured on this GPU and toolchain, not guaranteed across devices.'
        ])
    write(output, result)
    print(result['timing_reduction_percent'])
    print(result['context_128_kernel_reduction_percent'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('runs/association-runtime-panel'))
    parser.add_argument('--kernels', type=Path, default=Path('runs/association-final-kernels'))
    parser.add_argument('--output', type=Path, default=Path('reports/association-runtime.json'))
    args = parser.parse_args()
    summarize(args.root, args.kernels, args.output)
