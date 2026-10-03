"""Read-only CPU interventions on learned fast memory; no native runtime change."""
import argparse
from collections import defaultdict
import math
from pathlib import Path
import sys
import torch

from associative_reference import forward
from binding_learned_oracle import development_rows
from probes_cli import Reference
sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
from native_experiment import read, sha, write


def literal_controls():
    emission = torch.zeros(1, 4, 98)
    for t in range(4):
        emission[0, t, 1 if t == 3 else 0] = 1000
        emission[0, t, 32 + t % 2] = 1000
        emission[0, t, 64] = torch.atanh(torch.tensor([.25, .5, -.75, -.75][t]))
        emission[0, t, 96] = 0 if t == 3 else 40
        emission[0, t, 97] = -40 if t == 3 else 40
    parameters = (torch.eye(98), torch.zeros(98), torch.eye(32), torch.full((32,), .125))
    expected = {'normal': [.375, .375, -.625, .375],
                'discard_history': [.375, .125, -.625, .125], 'zero_read': [.125] * 4}
    for mode, values in expected.items():
        output, _ = forward(emission, *parameters, mode=mode)
        assert torch.allclose(output[0, :, 0], torch.tensor(values), atol=1e-7, rtol=0), mode
        assert torch.equal(output[0, :, 1:], torch.full((4, 31), .125)), mode
    default, state = forward(emission, *parameters)
    explicit, explicit_state = forward(emission, *parameters, mode='normal')
    assert torch.equal(default, explicit) and torch.equal(state, explicit_state)
    try:
        forward(emission, *parameters, mode='unknown')
    except ValueError:
        pass
    else:
        raise AssertionError('Unknown diagnostic mode accepted')
    return dict(passed=True, literal_modes=list(expected), output_bias_preserved=True,
                default_mode_unchanged=True, invalid_mode_rejected=True)


def assess(reference, fixtures):
    records, groups = [], defaultdict(list)
    erased_cache = {}
    for index, fixture in enumerate(fixtures):
        prompt = fixture['context'] + fixture['query']
        choices = [fixture['choice0'], fixture['choice1']]
        scores = [reference.score(prompt, choice) for choice in choices]
        assert all(math.isfinite(value) for value in scores), fixture['id']
        predicted = 0 if scores[0] <= scores[1] else 1
        key = (fixture['query'], *choices)
        if key not in erased_cache:
            erased_cache[key] = [reference.score(key[0], choice) for choice in choices]
        erased = erased_cache[key]
        assert all(math.isfinite(value) for value in erased), fixture['id']
        erased_choice = 0 if erased[0] <= erased[1] else 1
        greedy = reference.greedy(prompt, 4)
        record = dict(id=fixture['id'], pair=fixture['pair'], gold=fixture['correct'],
                      candidate_nll=scores, context_erased_nll=erased, choice=predicted,
                      correct=predicted == fixture['correct'], greedy=greedy,
                      greedy_correct=greedy == choices[fixture['correct']],
                      erased_correct=erased_choice == fixture['correct'])
        records.append(record)
        groups[fixture['pair']].append(record)
        if (index + 1) % 144 == 0:
            print(f'  {index + 1}/{len(fixtures)} questions assessed', flush=True)
    assert len(groups) == 144 and all(len(rows) == 4 for rows in groups.values())
    return dict(items=len(records), groups=len(groups),
                accuracy=sum(row['correct'] for row in records) / len(records),
                context_erased_accuracy=sum(row['erased_correct'] for row in records) / len(records),
                joint_accuracy=sum(all(row['correct'] for row in rows) for rows in groups.values()) / len(groups),
                greedy_joint_accuracy=sum(all(row['greedy_correct'] for row in rows) for rows in groups.values()) / len(groups),
                results=records)


def run(root, out):
    data = read(root / 'comparison.json')
    source = data['protocol']
    assert source['seeds'] == [1337, 2026, 31415] and source['online_endpoints'] == [34000, 67000, 130000]
    fixtures = development_rows(source)
    assert len(fixtures) == 576
    out.mkdir(parents=True, exist_ok=False)
    files = ['tests/associative_reference.py', 'tests/probes_cli.py', 'tests/associative_history.py']
    protocol = dict(status='declared_before_intervention', architecture='associative',
        seeds=source['seeds'], online_endpoint=130000, questions=576, groups=144,
        modes=['normal', 'discard_history', 'zero_read'], full_development=True,
        reference_source_sha256={path: sha(path) for path in files},
        probe_sha256=source['probe_sha256']['development.sgprobe'],
        learning_comparison_sha256=sha(root / 'comparison.json'),
        numerical_tolerance=3e-5, reserved_test_evaluated=False, model_parameters_changed=False,
        motivation='Exploratory follow-up planned after observing seed 1337 binding improvement and forgetting. '
                   'Tests reliance on fast matrix history in the trained model, not superiority during learning.',
        modes_definition=dict(normal='Unmodified CPU equations, also compared with all native development results.',
            discard_history='Clear only matrix history before every byte; keep current-byte write/read, projection weights and bias.',
            zero_read='Zero the matrix read vector before output projection; retain its bias and all other model equations.'),
        limits='Interventions change downstream activation distributions. No retraining, hyperparameter selection '
               'or biological-memory claim. CPU threshold disagreements must be reported.')
    write(out / 'protocol.json', protocol)
    controls = literal_controls()
    rows, numerical = [], []
    for seed in source['seeds']:
        directory = root / f'{seed}-associative'
        path = directory / 'checkpoint-130000.ckpt'
        identity = sha(path)
        declared = next(row for row in data['runs'] if row['seed'] == seed and
                        row['arm'] == 'associative' and row['online_updates'] == 130000)
        assert identity == declared['checkpoint_sha256']
        native = read(directory / 'development-130000.json')
        assert len(native['results']) == len(fixtures)
        for mode in protocol['modes']:
            print(f'Seed {seed}, mode {mode}', flush=True)
            with torch.inference_mode():
                result = assess(Reference(path, association_mode=mode), fixtures)
            write(out / f'{seed}-{mode}.json', result)
            if mode == 'normal':
                error, choices, greedies, failures = 0., 0, 0, []
                for cpu, gpu in zip(result['results'], native['results']):
                    assert cpu['id'] == gpu['id'] and cpu['gold'] == gpu['gold']
                    assert all(math.isfinite(value) for field in ('candidate_nll', 'context_erased_nll')
                               for value in gpu[field]), gpu['id']
                    item_error = max(abs(a - b) for field in ('candidate_nll', 'context_erased_nll')
                                     for a, b in zip(cpu[field], gpu[field]))
                    different_choice = cpu['choice'] != (0 if gpu['candidate_nll'][0] <= gpu['candidate_nll'][1] else 1)
                    different_greedy = cpu['greedy'] != gpu['greedy']
                    error = max(error, item_error)
                    choices += different_choice
                    greedies += different_greedy
                    if item_error >= protocol['numerical_tolerance'] or different_choice or different_greedy:
                        failures.append(dict(id=cpu['id'], max_score_error=item_error,
                                             choice_disagreement=different_choice, greedy_disagreement=different_greedy))
                numerical.append(dict(seed=seed, items=576, maximum_score_error=error,
                    candidate_choice_disagreements=choices, greedy_disagreements=greedies,
                    strict_tolerance_passed=not failures, failures=failures))
            assert sha(path) == identity
            rows.append(dict(seed=seed, mode=mode, checkpoint_sha256=identity,
                             **{k: v for k, v in result.items() if k != 'results'}))
            write(out / 'partial.json', dict(rows=rows, numerical=numerical))
            print(f'  complete groups={result["joint_accuracy"]:.4f}, greedy={result["greedy_joint_accuracy"]:.4f}', flush=True)
    assert all(sha(path) == identity for path, identity in protocol['reference_source_sha256'].items())
    write(out / 'comparison.json', dict(protocol=protocol, controls=controls, rows=rows, numerical=numerical,
          completed=True, all_normal_scores_within_tolerance=all(row['strict_tolerance_passed'] for row in numerical)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('runs/associative-long-panel'))
    parser.add_argument('--out', type=Path)
    parser.add_argument('--fixtures-only', action='store_true')
    args = parser.parse_args()
    if args.fixtures_only:
        print(literal_controls())
    elif args.out is None:
        parser.error('--out is required for learned interventions')
    else:
        run(args.root, args.out)
