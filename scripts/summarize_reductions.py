"""Collect the ordered-reduction checks without relabeling earlier experiments."""
import hashlib
import json
from pathlib import Path
import statistics

from replay_experiment import checkpoint


def read(path):
    return json.loads(Path(path).read_text())


def main():
    root = Path('runs/ordered-reductions/final')
    data = read(root/'comparison.json')
    executable = hashlib.sha256(Path('build/synapticgenesis.exe').read_bytes()).hexdigest()
    assert data['protocol']['executable_sha256']['ordered'] == executable
    assert data['native_commands'] == 29 and data['protocol']['long_endpoint'] == 130000
    assert data['restart_comparison']['complete_checkpoint_identical']
    assert all(r['complete_checkpoint_identical'] for r in data['fresh_repeat_comparisons']['ordered'])
    assert '100% tests passed out of 14' in Path('build-ordered-reductions-final.log').read_text(encoding='utf-8-sig')
    fixtures = ['test-results','live-test-results','adaptive-test-results','trace-test-results',
                'gated-test-results','selective-test-results']
    fixtures += ['feedback-test-results/'+cell for cell in ('lif','alif','trace','gated','selective')]
    oracles = [dict(fixture=path, **{k:v for k,v in read(Path('build')/path/'oracle.json').items() if k!='tensors'})
               for path in fixtures]
    assert all(r['passed'] for r in oracles)
    checks = dict(native_reductions=read('build/reduction-test-results/native.json'),
                  policy_cli=read('runs/ordered-reductions/integration-stage/result.json'),
                  curriculum_extension=read('runs/ordered-reductions/integration-extension/result.json'),
                  population=read('runs/ordered-reductions/integration-population/result.json'))
    assert all(r['passed'] for r in checks.values())
    checks.update(native_suites_passed=14, executable_sha256=executable,
                  independent_cpu_oracles=oracles, environment=read('runs/ordered-reductions/environment.json'))
    controls = {}
    for which in ('legacy','ordered'):
        ma, ea, _, _ = checkpoint(root/f'{which}-fresh-0/latest.ckpt')
        mb, eb, _, _ = checkpoint(root/f'{which}-one-stage/latest.ckpt')
        assert ma[17] == 4 and mb[17] == 5 and eb[16] == 1
        assert ea[16:] == eb[22:] and ea[4] == eb[4]
        assert all(ma[i] == mb[i] for i in (7,22,24,30))
        controls[which] = dict(replay_descriptors_equal=True, replay_rng_equal=True,
                               exposure_counters_equal=True, **data['one_stage_policy_controls'][which])
    summary = dict(protocol=data['protocol'], one_stage_controls=controls,
                   long_restart=data['restart_comparison'], live_timing=data['timing'])
    for name in ('weights','logits'):
        assert (root/'legacy-gradient-0'/f'{name}.f32').read_bytes() == (root/'ordered-gradient-0'/f'{name}.f32').read_bytes()
    summary['frozen_fixture_old_new_weights_and_forward_outputs_bitwise_identical'] = True
    summary['live_median_change_percent'] = 100*(data['timing']['ordered']['median_seconds'] /
                                                data['timing']['legacy']['median_seconds'] - 1)
    summary['batched_timing'] = []
    for batch, context in data['protocol']['batched_shapes']:
        record = dict(batch=batch, context=context, steps=400)
        for which in ('legacy','ordered'):
            selected = [r for r in data['batched_runs'] if (r['batch'],r['context'],r['binary'])==(batch,context,which)]
            assert len(selected) == 3
            times = [r['final_metrics']['elapsed_seconds'] for r in selected]
            record[which] = dict(seconds=times, median_seconds=statistics.median(times),
                                  repeated_checkpoints_identical=len({r['checkpoint_sha256'] for r in selected})==1)
        record['median_change_percent'] = 100*(record['ordered']['median_seconds']/record['legacy']['median_seconds']-1)
        summary['batched_timing'].append(record)
    summary['decode_from_identical_legacy_checkpoint'] = data['decode_from_identical_legacy_checkpoint']
    per_sequence = 4 * data['protocol']['hidden'] * 3 # Selective cell: three parameter gradients.
    summary['additional_explicit_scratch_bytes'] = dict(single_sequence_learner=0,
                                                        optional_live_batch16_evaluator=16*per_sequence,
                                                        batch8_learner=8*per_sequence,
                                                        batch16_learner=16*per_sequence,
                                                        checkpoint=0)
    summary['interpretation'] = ('Fixed-order reductions remove observed same-run arithmetic variation in these checks. '
                                 'This is not evidence of a language-quality gain or cross-platform bitwise reproducibility.')
    # Write only after every input and executable binding has been checked.
    for name, value in [('ordered-reductions-validation.json',checks),
                        ('ordered-reductions-comparison.json',data),
                        ('ordered-reductions-summary.json',summary)]:
        Path('reports',name).write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps({k:summary[k] for k in ('live_timing','live_median_change_percent','batched_timing')},indent=2))


if __name__ == '__main__':
    main()
