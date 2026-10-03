"""Authenticate a completed native activity census and write compact results.

This audits saved artifacts without executing a model or repeating the census.
Run from the repository root while its recorded sources are still available.
"""
import argparse
import math
from pathlib import Path
import statistics

from native_experiment import read, sha, write


def require(condition, message):
    if not condition:
        raise ValueError(message)


def identities(records):
    for path, expected in records.items():
        require(sha(path) == expected, 'Changed evidence: ' + str(path))


def summarize(panel, study, compilation):
    data = read(panel / 'result.json')
    protocol, records = data['protocol'], data['records']
    require(protocol == read(panel / 'protocol.json'), 'Census protocol changed')
    require(data['all_checkpoints_and_sources_unchanged']
            and data['all_native_windows_inspected']
            and not data['measured_learning_capacity'], 'Incomplete census')
    require(len(records) == protocol['checkpoints'] == 27
            and len({r['label'] for r in records}) == 27, 'Incomplete model panel')
    require([{k: r[k] for k in p} for p, r in zip(protocol['models'], records)]
            == protocol['models'], 'Model record identities changed')
    require(protocol['native_commands'] == data['native_commands'] == 162,
            'Wrong declared native command count')
    require(len(protocol['windows']) == 6 and protocol['state_reset_each_window']
            and not protocol['reserved_tests_scored'], 'Wrong observation panel')
    identities(protocol['source_sha256'])
    identities({study / 'comparison.json': protocol['study_comparison_sha256'],
                study / 'execution-check.json': protocol['study_execution_sha256']})
    require(read(study / 'execution-check.json')['passed'], 'Study audit failed')
    build = read(compilation)
    require(build['compilation_passed']
            and build['trace_executable_sha256'] == protocol['diagnostic_sha256'],
            'Native compilation identity mismatch')
    identities(build['source_sha256'])
    observations = sum(w['observed_steps'] for w in protocol['windows'])
    require(observations == 1172, 'Unexpected observation count')
    native_commands = trace_files = 0
    loss_errors, window_identities = [], []
    for record in records:
        identities({record['checkpoint']: record['checkpoint_sha256']})
        require(len(record['pooled_layers']) == 4 and len(record['windows']) == 6,
                'Incomplete model observations')
        for descriptor, row in zip(protocol['windows'], record['windows']):
            require(row['window'] == descriptor['name'], 'Window order changed')
            directory = panel / record['label'] / row['window']
            identities({directory / 'result.json': row['result_sha256']})
            detail = read(directory / 'result.json')
            p = detail['protocol']
            require(p == read(directory / 'protocol.json'), 'Window protocol changed')
            require(p['checkpoint_sha256'] == record['checkpoint_sha256']
                    and p['source_sha256'] == descriptor['sha256']
                    and p['diagnostic_sha256'] == protocol['diagnostic_sha256'],
                    'Window input identity mismatch')
            identities({p['checkpoint']: p['checkpoint_sha256'],
                        p['source']: p['source_sha256'],
                        p['diagnostic']: p['diagnostic_sha256']})
            identities(p['script_sha256'])
            require(p['native_source_sha256'] == build['source_sha256'],
                    'Window source differs from compiled native source')
            require(p['observed_steps'] == descriptor['observed_steps']
                    and p['input_bytes'] == descriptor['input_bytes']
                    and detail['input_checkpoint_and_diagnostic_unchanged'],
                    'Invalid window completion')
            commands = read(directory / 'commands.json')
            require(len(commands) == detail['native_commands'] == 1,
                    'Unexpected native command count')
            require([Path(x).resolve() for x in commands[0]]
                    == [Path(x).resolve() for x in
                        (p['diagnostic'], p['checkpoint'], p['source'], directory / 'trace')],
                    'Native command arguments disagree with protocol')
            actual = {f.name for f in (directory / 'trace').iterdir()}
            require(actual == set(detail['trace_sha256']), 'Trace file set changed')
            identities({directory / 'trace' / f: digest
                        for f, digest in detail['trace_sha256'].items()})
            error = detail['independent_loss_consistency_max_error']
            require(math.isfinite(error) and 0 <= error < 3e-5
                    and error == row['independent_loss_consistency_max_error']
                    and detail['mean_loss_nats_per_byte'] == row['mean_loss_nats_per_byte'],
                    'Window loss check failed')
            loss_errors.append(error)
            native_commands += len(commands)
            trace_files += len(actual)
            window_identities.append(dict(model=record['label'], window=row['window'],
                                          result_sha256=row['result_sha256']))
        for layer in record['pooled_layers']:
            require(layer['steps'] == observations and layer['hidden'] in (512, 588),
                    'Unexpected pooled layer shape')
    require(native_commands == 162, 'Native commands are incomplete')
    metrics = ('spike_event_fraction', 'silent_spike_emission_squared_fraction',
               'centered_emission_participation_ratio')
    aggregates = []
    for architecture in ('selective', 'wide-selective', 'associative'):
        for role in ('parent', 'control', 'teacher'):
            selected = [r for r in records if r['architecture'] == architecture and r['role'] == role]
            require(sorted(r['seed'] for r in selected) == [1337, 2026, 31415],
                    'Wrong aggregate seed panel')
            layers = [layer for r in selected for layer in r['pooled_layers']]
            require(all(math.isfinite(layer[key]) for layer in layers for key in metrics),
                    'Nonfinite aggregate input')
            aggregates.append(dict(architecture=architecture, role=role, models=3, layers=12,
                **{key: dict(mean=statistics.mean(layer[key] for layer in layers),
                             minimum=min(layer[key] for layer in layers),
                             maximum=max(layer[key] for layer in layers)) for key in metrics},
                never_spiked_neurons=sum(layer['never_spiked_neurons'] for layer in layers),
                never_emitted_above_epsilon_neurons=sum(
                    layer['never_emitted_above_epsilon_neurons'] for layer in layers)))
    all_layers = [layer for r in records for layer in r['pooled_layers']]
    return dict(kind='Authenticated finite-window native activity summary',
        panel_result_sha256=sha(panel / 'result.json'),
        panel_protocol_sha256=sha(panel / 'protocol.json'),
        compilation_record_sha256=sha(compilation),
        summarizer_sha256=sha(__file__),
        native_source_sha256=build['source_sha256'],
        passed=True, models=27, layers=108, windows_per_model=6,
        observed_positions_per_model=observations, native_commands=162,
        authenticated_trace_files=trace_files, authenticated_window_results=window_identities,
        independent_loss_consistency_max_error=max(loss_errors),
        all_checkpoints_sources_diagnostic_and_traces_unchanged=True,
        spike_event_fraction_range=[min(l['spike_event_fraction'] for l in all_layers),
                                    max(l['spike_event_fraction'] for l in all_layers)],
        never_spiked_neurons=sum(l['never_spiked_neurons'] for l in all_layers),
        never_emitted_above_epsilon_neurons=sum(
            l['never_emitted_above_epsilon_neurons'] for l in all_layers),
        aggregates=aggregates,
        aggregation='Equal mean over three seeds and four pooled layers per architecture/role; '
                    'each layer pools six windows with one weight per observed byte. '
                    'Layers and related models are not independent experimental replicates.',
        measured_learning_capacity=False, changed_models=False,
        limits=['Activity is from six selected reset-state training windows, not general language.',
                'Every unit firing in these windows does not establish useful task contribution.',
                'Squared emissions are mathematical magnitudes, not measured electrical energy.',
                'Participation ratio is not a count of unused units or a learning-capacity measure.',
                'Native-logit/loss consistency is not independent CPU model parity.',
                'This audit launches no model commands and makes no structural change.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--panel', type=Path, default=Path('runs/native-activity-panel'))
    parser.add_argument('--study', type=Path, default=Path('runs/teacher-retention-panel'))
    parser.add_argument('--compilation', type=Path, default=Path('runs/frozen-teacher-compilation.json'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.panel, args.study, args.compilation)
    write(args.out, result)
    print('Authenticated', result['models'], 'models,', result['native_commands'],
          'native windows and', result['authenticated_trace_files'], 'trace files.')
