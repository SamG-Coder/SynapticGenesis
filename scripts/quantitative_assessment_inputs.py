"""Bind read-only quantitative assessment to complete, selected-prose checkpoints."""
from pathlib import Path

from corpus.selection import SELECTION, require_training_spec
from membrane_study_state import read_state, require
from native_experiment import read
from prose_founder import file_hash
from quantitative_probes import build, protect


ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / 'data/quantitative-assessment-v1.json'


def authenticate(pins):
    for path, value in pins.items():
        require(file_hash(path) == value, 'Assessment input changed: ' + str(path))


def sample_rows(rows):
    return [row for row in rows if row['id'] == f"{row['skill']}-1-0-A"]


def arguments(runtime, case, suite, rows, generation, destination):
    checkpoint = case['checkpoint']
    commands = [[str(runtime), 'language-probes', '--checkpoint', checkpoint, '--probes',
                 str(suite / 'development.sgprobe'), '--output', str(destination / 'probes.json')]]
    for row in sample_rows(rows):
        commands.append(list(map(str, [runtime, 'sample', '--checkpoint', checkpoint,
            '--prompt', row['context'] + row['query'], '--tokens', generation['bytes'],
            '--seed', generation['seed'], '--temperature', generation['temperature'],
            '--top-k', generation['top_k'], '--graph', '--output', destination / f"sample-{row['skill']}.txt"])))
    return commands


def admit(spec_path=SPEC):
    spec_path = Path(spec_path)
    spec = read(spec_path)
    require(spec['version'] == 'quantitative-prose-baselines-v1' and
            spec['learning_commands'] == 0 and spec['native_commands_per_model'] == 9 and
            spec['native_commands_total'] == 18, 'Unexpected quantitative assessment contract')
    g = spec['generation']
    require((g['bytes'], g['seed'], g['temperature'], g['top_k']) == (128, 42, .8, 40),
            'Raw sample policy changed')
    refs = {key: (ROOT / spec[key] if key in ('suite_audit', 'suite_source') else Path(spec[key]))
            for key in ('suite_audit', 'suite_source', 'baseline_result', 'baseline_audit', 'source_spec', 'runtime')}
    pins = {p.as_posix(): spec[k + '_sha256'] for k, p in refs.items()}
    suite, predecessor = Path(spec['suite']), Path(spec['predecessor'])
    pins[(suite / 'manifest.json').as_posix()] = spec['suite_manifest_sha256']
    pins[(predecessor / 'protocol.json').as_posix()] = spec['predecessor_protocol_sha256']
    authenticate(pins)
    source = require_training_spec(refs['source_spec'])
    require(source['version'] == spec['source_edition'] == 'selected-prose-scale-v1',
            'Unexpected quantitative checkpoint source edition')
    manifest, suite_audit, suite_spec = read(suite / 'manifest.json'), read(refs['suite_audit']), read(refs['suite_source'])
    require(manifest['version'] == suite_spec['version'] == suite_audit['version'] == 'quantitative-development-v2'
            and suite_audit['passed'] and suite_audit['manifest_sha256'] == spec['suite_manifest_sha256']
            and manifest['source_spec_sha256'] == suite_audit['source_spec_sha256'] == spec['suite_source_sha256']
            and suite_audit['prepared_files'] == manifest['prepared_files'], 'Quantitative suite audit differs')
    require(not manifest['training_admitted'] and not manifest['model_quality_verified'] and
            suite_audit['native_commands'] == 0, 'Unexpected suite preparation history')
    pins.update(manifest['authenticated_inputs'])
    pins.update({(suite / name).as_posix(): value for name, value in manifest['prepared_files'].items()})
    authenticate(pins)
    rows, statements, _ = build(suite_spec)
    require(rows == read(suite / 'questions.json') and len(sample_rows(rows)) == 8,
            'Prepared questions or predetermined samples differ')
    pins.update(protect(rows, statements, suite_spec['source_files']))
    published, publication = read(refs['baseline_result']), read(refs['baseline_audit'])
    require(published['complete'] and published['random_initialization'] and not published['imported_weights']
            and not published['reserved_tests_scored'] and publication['exact_result_copy']
            and publication['evidence']['passed'] and publication['published_result_sha256'] ==
            publication['evidence']['result_sha256'] == spec['baseline_result_sha256'],
            'Prose baseline publication differs')
    require(published['protocol']['evaluation']['source_spec'] == spec['source_spec'] and
            published['protocol']['authenticated_inputs'][spec['source_spec']] == spec['source_spec_sha256'],
            'Checkpoint publication refers to another source edition')
    require([(c['profile'], c['stage']) for c in spec['cases']] == [('105m', 4), ('411m', 4)],
            'Unexpected baseline checkpoint selection')
    cases = []
    for expected, group in zip(spec['cases'], ('controls', 'rows')):
        selected = [r for r in published[group] if r['stage'] == 4]
        require(len(selected) == 1 and all(selected[0][k] == v for k, v in expected.items()),
                'Declared checkpoint differs from the completed prose comparison')
        old = selected[0]
        require(old['checkpoint_unchanged'] and not old['reserved_tests_scored'], 'Unexpected baseline assessment state')
        path = Path(expected['checkpoint'])
        require(path.name == 'stage-4.ckpt' and file_hash(path) == expected['checkpoint_sha256'],
                'Immutable prose endpoint changed')
        state = read_state(path)
        require(all(old['state']['counters'][k] == value for k, value in state['counters'].items())
                and state['seed'] == old['state']['seed'] == 1337
                and state['parameters'] == old['state']['counters']['parameters']
                and state['replay_payload_sha256'] == old['state']['replay_payload_sha256']
                and state['cursor_and_rng'] == old['state']['cursor_and_rng']
                and state['hyperparameters'] == old['state']['hyperparameters']
                and state['membrane_cost'] == 0 and state['hyperparameters'][4] == 0,
                'Published checkpoint state differs')
        require(state['counters']['online_updates'] == 216289 and state['counters']['observed_pairs'] == 27680129,
                'Expected the complete prose curriculum')
        cases.append(dict(expected, state=state))
        pins[path.as_posix()] = expected['checkpoint_sha256']
    prior = read(predecessor / 'protocol.json')
    accepted = read(predecessor / 'acceptance-verified.json')
    require(prior['version'] == 'membrane-learning-plan-v1' and prior['runtime'] == spec['runtime'] and
            prior['authenticated_inputs'][spec['runtime']] == spec['runtime_sha256'] and accepted['passed']
            and accepted['candidate_runtime_sha256'] == spec['runtime_sha256'], 'Candidate runtime admission differs')
    pins[(predecessor / 'acceptance-verified.json').as_posix()] = file_hash(predecessor / 'acceptance-verified.json')
    paths = [spec_path, SELECTION, *(ROOT / 'scripts' / name for name in (
        'quantitative_assessment_inputs.py', 'quantitative_assessment.py', 'quantitative_report.py',
        'quantitative_probes.py', 'process_gate.py', 'native_experiment.py', 'prose_founder.py',
        'membrane_study_state.py', 'corpus/selection.py')), ROOT / 'tests/quantitative_assessment.py']
    # The frozen runtime source defines raw-byte JSON and printed-score precision.
    native_root = Path(spec['runtime']).parent.parent
    paths.extend(native_root / 'src' / name for name in ('language_probes.cuh', 'spike_lm.cu'))
    pins.update({p.as_posix(): file_hash(p) for p in paths})
    return dict(specification=spec, cases=cases, questions=rows, authenticated_inputs=pins,
                suite_preparation_audit=suite_audit, predecessor_protocol=prior)


def completed_predecessor(directory, expected_hash):
    """Completion consistency only; process liveness is a separate held-handle gate."""
    directory = Path(directory)
    require(file_hash(directory / 'protocol.json') == expected_hash, 'GPU predecessor declaration changed')
    require(not (directory / 'failure.json').exists(), 'GPU predecessor failed')
    result, execution, protocol = [read(directory / name) for name in ('result.json', 'execution.json', 'protocol.json')]
    require(result['complete'] is True and result['protocol_sha256'] == expected_hash and
            result['protocol'] == protocol and execution['phase'] == 'complete' and
            execution['protocol_sha256'] == expected_hash and execution['completed_models'] == 15,
            'GPU predecessor has not completed')
    require(len(result['rows']) == result['native_learning_commands'] == 15 and
            result['native_assessment_commands'] == 360, 'GPU predecessor command coverage differs')
    expected = [(c['seed'], c['arm']) for c in protocol['cases']]
    require([(r['seed'], r['arm']) for r in result['rows']] == expected,
            'GPU predecessor cases differ')
    for case, row in zip(protocol['cases'], result['rows']):
        require(read(Path(case['directory']) / 'result.json') == dict(complete=True, **row) and
                len(row['assessments']) == 2 and all(a['complete'] for a in row['assessments']),
                'GPU predecessor case completion differs')
    return dict(complete=True, result_sha256=file_hash(directory / 'result.json'), cases=15,
        meaning='Normal process exit and completed case records authorize serial GPU use; this is not a new full numerical audit of the study.')
