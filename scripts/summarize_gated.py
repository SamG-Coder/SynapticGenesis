"""Collect read-gate evidence, including the ordinary-trace capacity control."""
import hashlib
import json
from pathlib import Path
import struct

from audit_binding import audit


def read(path):
    return json.loads(Path(path).read_text())


def common_weights(path):
    raw = Path(path).read_bytes()
    meta = struct.unpack_from('<32Q',raw)
    cell,c,h,layers = meta[1:5]
    assert cell in (3,4) and meta[7] == 0
    weights = raw[288:288+4*meta[14]]
    chunks = [weights[:256*c*4]]
    at = 256*c*4
    for _ in range(layers):
        size = (2*c*h + 2*c + 4*h)*4
        chunks.append(weights[at:at+size])
        at += size
        if cell == 4:
            gate_size = (c*h+h)*4
            assert not any(weights[at:at+gate_size]), 'Initial gate must be zero'
            at += gate_size
    chunks.append(weights[at:])
    assert len(weights)-at == (c+256*c+256)*4
    return hashlib.sha256(b''.join(chunks)).hexdigest()


def collect():
    base = Path('runs/gated-binding-panel')
    rows = []
    for cell, hidden in [('trace',512), ('gated',344), ('gated',512), ('trace',768)]:
        root = Path('runs/binding-trace-1337') if (cell,hidden)==('trace',512) else base/f'{cell}-{hidden}'
        meta = struct.unpack_from('<32Q',(root/'latest.ckpt').read_bytes())
        row = dict(cell=cell, hidden=hidden, parameters=meta[14], run=root.as_posix(),
                   checkpoint_sha256=hashlib.sha256((root/'latest.ckpt').read_bytes()).hexdigest(),
                   initial_sha256=hashlib.sha256((root/'initial.ckpt').read_bytes()).hexdigest(),
                   session=read(root/'session.json'), decode=read(root/'decode/benchmark.json'))
        for split in ('train','development'):
            file = root/('train-probes.json' if (cell,hidden,split)==('trace',512,'train') else f'{split}.json')
            row[split] = {k:v for k,v in read(file).items() if k!='results'}
            row[split+'_audit'] = audit(file,split)
        rows.append(row)
    a = common_weights(Path('runs/binding-trace-1337/initial.ckpt'))
    b = common_weights(base/'gated-512/initial.ckpt')
    assert a == b
    for key in ('observed_pairs','replay_pairs','replay_updates','generated_bytes','global_updates','curriculum_hash'):
        assert len({r['session'][key] for r in rows}) == 1, key
    comparison = dict(protocol='docs/gated-readout.md', seed=1337, runs=rows,
                      common_initial_parameters_sha256=a,
                      matched_width_common_initial_parameters_identical=True,
                      near_matched_parameter_pairs=[['trace-512','gated-344'],['gated-512','trace-768']],
                      same_curriculum_and_exposure=True, reserved_test_evaluated=False,
                      limits='One seed per setting. Different widths change initialization and recurrent-state capacity. '
                             'No mastery or general architecture ranking established.')
    Path('reports/gated-binding.json').write_text(json.dumps(comparison,indent=2)+'\n')
    validation = dict(native_suites_passed=11,
                      gated_native=read('build/gated-test-results/native.json'),
                      oracle=read('build/gated-test-results/oracle.json'),
                      weighted_oracle=read('build/feedback-test-results/gated/oracle.json'),
                      evolution=read('build/evolution-test-results/native.json'),
                      curriculum_native=read('build/curriculum-test-results/native.json'),
                      curriculum_cli=read('runs/gated-curriculum-cli/result.json'),
                      probes_cli=read('runs/gated-probes-cli/result.json'),
                      binding_cli=read('runs/gated-binding-cli/result.json'),
                      population_cli=read('runs/gated-population-cli/result.json'),
                      scarcity_cli=read('runs/gated-scarcity-cli/result.json'),
                      batch_prefix_policy_cli=read('runs/gated-burn-cli/result.json'))
    Path('reports/gated-validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    cues = [read(f'runs/gated-cue-{seed}-{delay}/result.json')
            for seed in (1337,2026,31415) for delay in (128,256)]
    assert all(r['parameters']==68016 and r['all_state_erased_accuracy']==.5 for r in cues)
    Path('reports/gated-cue.json').write_text(json.dumps(dict(channels=64,hidden=88,layers=2,
        fast_math=True, steps=2000, learning_rate=.001, synthetic_only=True, runs=cues,
        limits='Disposable one-bit recall controls. Different width from the earlier ungated trace; '
               '68,016 versus 67,136 parameters. No language or general intelligence claim.'),indent=2)+'\n')
    for row in rows:
        print(row['cell'],row['hidden'],row['parameters'],row['development']['joint_accuracy'],
              row['session']['final_validation_loss'],row['decode']['graph_us_per_byte'])


if __name__ == '__main__':
    collect()
