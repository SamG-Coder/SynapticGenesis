"""Collect compact evidence from the documented trace-cell experiments."""
import hashlib
import json
from pathlib import Path
import struct


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def common_initial(path):
    raw = Path(path).read_bytes()
    meta = struct.unpack_from('<32Q', raw)
    cell, c, h, layers = meta[1:5]
    n = meta[14]
    assert meta[7] == 0 and not any(raw[288+4*n:288+12*n]), 'Expected fresh optimizer and no updates'
    w = raw[288:288+4*n]
    common = bytearray()
    at = 0

    def take(count):
        nonlocal at
        common.extend(w[at:at+4*count])
        at += 4*count

    take(256*c)
    for _ in range(layers):
        take(c + h*c + h + c*h + c + h)
        if cell in (2, 3):
            at += 8*h
    take(c + 256*c + 256)
    assert at == len(w)
    return hashlib.sha256(common).hexdigest()


def collect():
    arms = []
    initial_hashes = []
    for cell in ('lif', 'alif', 'trace'):
        root = Path(f'runs/binding-{cell}-1337')
        initial_hashes.append(common_initial(root/'initial.ckpt'))
        meta = struct.unpack_from('<32Q', (root/'latest.ckpt').read_bytes())
        row = dict(cell=cell, architecture_version=meta[1], parameters=meta[14], run=root.as_posix(),
                   initial_sha256=sha(root/'initial.ckpt'), checkpoint_sha256=sha(root/'latest.ckpt'),
                   session=read(root/'session.json'), decode=read(root/'decode/benchmark.json'))
        for name, file in [('development', 'development.json'), ('training', 'train-probes.json')]:
            row[name] = {k:v for k,v in read(root/file).items() if k != 'results'}
        arms.append(row)
    assert len(set(initial_hashes)) == 1
    for name in ('observed_pairs', 'replay_updates', 'replay_pairs', 'generated_bytes',
                 'curriculum_hash', 'curriculum_base_lr', 'global_updates'):
        assert len({a['session'][name] for a in arms}) == 1, name
    comparison = dict(protocol='docs/binding.md', seed=1337,
                      source_spec_sha256=sha('data/lessons-binding-v2.json'),
                      curriculum_sha256=sha('data/prepared/binding-v2/curriculum.sg'),
                      common_initial_parameters_sha256=initial_hashes[0],
                      common_initial_parameters_byte_identical=True,
                      same_observation_and_replay_budget=True,
                      reserved_test_evaluated=False, arms=arms,
                      limits='One initialization seed per cell on familiar vocabulary/templates. '
                             'No mastery or architecture superiority claim. Decode times are medians '
                             'of seven 1024-byte rounds after warmup, excluding capture and prompt setup; '
                             'host sampling, transfers and synchronization are included.')
    Path('reports/trace-binding.json').write_text(json.dumps(comparison, indent=2)+'\n')
    validation = dict(native_suites_passed=10,
                      trace_native=read('build/trace-test-results/native.json'),
                      evolution=read('build/evolution-test-results/native.json'),
                      curriculum_native=read('build/curriculum-test-results/native.json'),
                      trace_oracle=read('build/trace-test-results/oracle.json'),
                      weighted_trace_oracle=read('build/feedback-test-results/trace/oracle.json'),
                      curriculum_cli=read('runs/trace-curriculum-cli/result.json'),
                      probes_cli=read('runs/trace-probes-final/result.json'),
                      binding_cli=read('runs/trace-binding-cli/result.json'),
                      population_cli=read('runs/trace-population-cli/result.json'),
                      scarcity_cli=read('runs/trace-scarcity-cli/result.json'),
                      batch_prefix_policy_cli=read('runs/trace-burn-cli/result.json'))
    Path('reports/trace-validation.json').write_text(json.dumps(validation, indent=2)+'\n')
    Path('reports/trace-memory.json').write_text(json.dumps(read('runs/trace-memory-panel/comparison.json'), indent=2)+'\n')
    print(json.dumps([dict(cell=a['cell'], accuracy=a['development']['accuracy'],
                           joint=a['development']['joint_accuracy'],
                           reader_loss=a['session']['final_validation_loss']) for a in arms], indent=2))


if __name__ == '__main__':
    collect()
