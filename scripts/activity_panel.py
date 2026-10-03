"""Fixed native activity windows across all final models and their parents.

This is an exploratory read-only census, not a plasticity or quality experiment.
Run only after the retention study and its queued native checks have finished.
"""
import argparse
import hashlib
from pathlib import Path

import numpy as np

from activity_metrics import activity
from extend_curriculum import read_schedule
from inspect_native_activity import inspect, Reference
from native_experiment import read, sha, write


def selected_windows(study):
    """Choose two deterministic document windows per selected training group."""
    p = read(study / 'protocol.json')
    schedule = study / 'edition/curriculum.sg'
    if sha(schedule) != p['source_admission']['schedule_sha256']:
        raise ValueError('Activity curriculum identity changed')
    _, stages = read_schedule(schedule)
    editions = p['source_admission']['editions']
    if len(stages) != 4 or len(editions) != 4:
        raise ValueError('Expected the declared four-stage retention curriculum')
    documents, prior = [], []
    for stage, edition in zip(stages, editions):
        if (sha(stage['source']) != edition['sha256']
                or len(stage['content']) != edition['bytes']):
            raise ValueError('Activity source edition changed')
        current = stage['content'].split(b'\x1e')
        if len(current) != edition['documents'] or current[:len(prior)] != prior:
            raise ValueError('Activity source is not the authenticated cumulative edition')
        documents.append(current[len(prior):])
        prior = current
    windows = []
    for group, stage_index in [('reading', 0), ('binding', 2), ('narrative', 3)]:
        docs = documents[stage_index]
        for slot, numerator in enumerate((1, 3)):
            index = len(docs) * numerator // 4
            document = docs[index]
            length = min(len(document), 257)
            start = (len(document) - length) // 2
            content = document[start:start + length]
            if len(content) < 2:
                raise ValueError('Selected activity document is too short')
            windows.append(dict(name=f'{group}-{slot}', group=group, stage=stage_index+1,
                document_index_within_new_stage=index, document_bytes=len(document),
                document_sha256=hashlib.sha256(document).hexdigest(), byte_offset=start,
                input_bytes=len(content), observed_steps=len(content)-1,
                sha256=hashlib.sha256(content).hexdigest(), content=content,
                source_edition_sha256=editions[stage_index]['sha256']))
    return windows


def run(study, out, diagnostic):
    study, diagnostic = study.resolve(), diagnostic.resolve()
    data, execution = read(study / 'comparison.json'), read(study / 'execution-check.json')
    p = data['protocol']
    if (not execution['passed'] or len(data['runs']) != 54
            or p['status'] != 'declared_before_training' or read(study / 'protocol.json') != p
            or p['seeds'] != [1337, 2026, 31415]
            or p['baseline_online_updates'] != 130000 or p['online_endpoints'][-1] != 190000):
        raise ValueError('Expected the completed, execution-audited full retention study')
    windows = selected_windows(study)
    models = []
    for row in data['runs']:
        endpoint = row['online_updates']
        if endpoint not in (130000, 190000) or (endpoint == 130000 and row['variant'] != 'control'):
            continue
        parent = endpoint == 130000
        label = f"{row['seed']}-{row['architecture']}-" + ('parent' if parent else row['variant'])
        checkpoint = (Path(p['parent_directory']) / f"{row['seed']}-{row['architecture']}" if parent
                      else study / f"{row['seed']}-{row['arm']}") / f'checkpoint-{endpoint}.ckpt'
        if sha(checkpoint) != row['checkpoint_sha256']:
            raise ValueError('Activity checkpoint identity changed')
        models.append(dict(label=label, seed=row['seed'], architecture=row['architecture'],
                           role='parent' if parent else row['variant'], online_updates=endpoint,
                           checkpoint=checkpoint.as_posix(), checkpoint_sha256=row['checkpoint_sha256']))
    if len(models) != 27 or len({m['label'] for m in models}) != 27:
        raise ValueError('Incomplete activity model panel')
    out.mkdir(parents=True, exist_ok=False)
    inputs = out / 'inputs'
    inputs.mkdir()
    descriptors = []
    for window in windows:
        (inputs / (window['name'] + '.txt')).write_bytes(window['content'])
        descriptors.append({k: v for k, v in window.items() if k != 'content'})
    sources = ['scripts/activity_panel.py', 'scripts/activity_metrics.py',
               'scripts/inspect_native_activity.py', 'scripts/extend_curriculum.py',
               'scripts/native_experiment.py', 'tests/probes_cli.py', 'tests/learned_trace_dump.cu']
    protocol = dict(kind='Exploratory fixed-window native activity census; no model learning',
                    selection='Quarter and three-quarter document indexes of new reading, binding and narrative '
                              'training groups; centered window up to 257 bytes per document. Same six windows for every model.',
                    study_comparison_sha256=sha(study / 'comparison.json'),
                    study_execution_sha256=sha(study / 'execution-check.json'),
                    diagnostic_sha256=sha(diagnostic),
                    source_sha256={s: sha(s) for s in sources}, windows=descriptors, models=models,
                    checkpoints=27, native_commands=162, state_reset_each_window=True,
                    reserved_tests_scored=False, generated_text_targets=False,
                    pooled_weighting='Each observed byte contributes once; window lengths differ and are recorded.',
                    limits='No fresh-model learning curve, no causal plasticity diagnosis, no pruning or renewal. '
                           'Tiny reset-state samples cannot establish general activity or task importance.')
    write(out / 'protocol.json', protocol)
    results = []
    for model in models:
        checkpoint = Path(model['checkpoint'])
        ref = Reference(checkpoint)
        directory = out / model['label']
        directory.mkdir()
        window_results = []
        for window in windows:
            result = inspect(checkpoint, inputs / (window['name'] + '.txt'),
                             directory / window['name'], diagnostic)
            window_results.append(dict(window=window['name'],
                                       result_sha256=sha(directory / window['name'] / 'result.json'),
                                       mean_loss_nats_per_byte=result['mean_loss_nats_per_byte'],
                                       independent_loss_consistency_max_error=result['independent_loss_consistency_max_error']))
        layers = []
        for layer, block in enumerate(ref.blocks):
            arrays = {}
            for name in ('spikes', 'emission', 'u'):
                arrays[name] = np.concatenate([
                    np.fromfile(directory / window['name'] / 'trace' / f'layer-{layer}-{name}.f32',
                                dtype='<f4').reshape(window['observed_steps'], ref.h) for window in windows])
            layers.append(dict(layer_zero_based=layer,
                               **activity(arrays['spikes'], arrays['emission'], arrays['u'], block[3].numpy())))
        record = dict(**model, windows=window_results, pooled_layers=layers)
        results.append(record)
        write(out / 'partial.json', results)
        print(model['label'], 'six native windows complete', flush=True)
    if any(sha(s) != expected for s, expected in protocol['source_sha256'].items()):
        raise ValueError('Activity inspection sources changed')
    if sha(diagnostic) != protocol['diagnostic_sha256']:
        raise ValueError('Activity diagnostic changed')
    if (sha(study / 'comparison.json') != protocol['study_comparison_sha256']
            or sha(study / 'execution-check.json') != protocol['study_execution_sha256']):
        raise ValueError('Underlying study evidence changed during inspection')
    if any(sha(m['checkpoint']) != m['checkpoint_sha256'] for m in models):
        raise ValueError('Activity inspection changed a checkpoint')
    # Reauthenticate every source edition and reproduce all selected windows.
    if [{k: v for k, v in w.items() if k != 'content'} for w in selected_windows(study)] != descriptors:
        raise ValueError('Activity source selection changed')
    write(out / 'result.json', dict(protocol=protocol, records=results,
        all_checkpoints_and_sources_unchanged=True, all_native_windows_inspected=True,
        native_commands=162, measured_learning_capacity=False))
    print(out / 'result.json', 'all 27 models inspected; no plasticity conclusion')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--study', type=Path, default=Path('runs/teacher-retention-panel'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--diagnostic', type=Path,
                        default=Path('build/trace-diagnostic/synaptic-trace-diagnostic.exe'))
    args = parser.parse_args()
    run(args.study, args.out, args.diagnostic)
