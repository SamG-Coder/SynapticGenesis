"""Collect the two native continuation arms without re-running training."""
import json
import hashlib
from pathlib import Path


def collect():
    directories = [Path('runs/live-extension-demonstration'), Path('runs/live-extension-more-replay')]
    arms = [json.loads((p / 'result.json').read_text()) for p in directories]
    assert arms[0]['stages'][0]['checkpoint'] == arms[1]['stages'][0]['checkpoint']
    binary_hash = hashlib.sha256(Path('build/synapticgenesis.exe').read_bytes()).hexdigest()
    assert all(a['executable_sha256'] == binary_hash for a in arms)
    table = []
    for directory, arm, cadence in zip(directories, arms, [4, 1]):
        assert all(s['session']['replay_every'] == cadence for s in arm['stages'][1:])
        before, final = arm['stages'][0]['checkpoint'], arm['stages'][-1]['checkpoint']
        added = final['online_updates'] - before['online_updates']
        replays = final['replay_updates'] - before['replay_updates']
        assert added == 28000 and replays == added // cadence
        assert final['optimizer_updates'] - before['optimizer_updates'] == added + replays
        for stage in arm['stages'][1:]:
            table.append(dict(run=directory.as_posix(), replay_every=cadence, stage=stage['label'],
                              single_fact_pair_accuracy=stage['single_fact_training_fit']['joint_accuracy'],
                              single_fact_greedy_pair_accuracy=stage['single_fact_training_fit']['greedy_exact_joint_accuracy'],
                              binding_joint_accuracy=stage['binding_development']['joint_accuracy'],
                              reader_loss=stage['session']['final_validation_loss'],
                              live_seconds=stage['session']['elapsed_seconds'],
                              ordinary_tick_p95_ms=stage['session']['update_tick_p95_ms'],
                              speech_tick_p95_ms=stage['session']['update_and_speech_tick_p95_ms']))
        arm['training_replay_every'] = cadence
        arm['live_seconds'] = sum(s['session']['elapsed_seconds'] for s in arm['stages'][1:])
        arm['added_online_updates'], arm['added_replay_updates'] = added, replays
    for a, b in zip(arms[0]['stages'], arms[1]['stages']):
        for field in ('online_updates', 'observed_pairs', 'generated_bytes', 'curriculum_hash'):
            assert a['checkpoint'][field] == b['checkpoint'][field]
    result = dict(protocol='docs/live-extension-experiment.md',
                  same_starting_checkpoint=True, same_online_content_and_exposure=True,
                  same_learning_rate=True, same_generated_byte_count=True,
                  replay_cadence_changed=True, replay_and_optimizer_budgets_differ=True,
                  one_seed=True, default_policy_changed=False, reserved_test_evaluated=False,
                  table=table, arms=arms)
    Path('reports/live-extension-experiment.json').write_text(json.dumps(result, indent=2) + '\n')
    assert '100% tests passed out of 11' in Path('build-extension.log').read_text(encoding='utf-8-sig')
    validation = dict(native_suites_passed=11, executable_sha256=binary_hash,
                      native_curriculum=json.loads(Path('build/curriculum-test-results/native.json').read_text()),
                      ordinary_curriculum_cli=json.loads(Path('runs/extension-ordinary-curriculum-cli/result.json').read_text()),
                      extension_cli=json.loads(Path('runs/curriculum-extension-cli/result.json').read_text()),
                      preparation=json.loads(Path('runs/extension-preparation-cli/result.json').read_text()))
    assert all(validation[k]['passed'] for k in ('native_curriculum', 'ordinary_curriculum_cli', 'extension_cli', 'preparation'))
    Path('reports/live-extension-validation.json').write_text(json.dumps(validation, indent=2) + '\n')
    print(json.dumps(table, indent=2))


if __name__ == '__main__':
    collect()
