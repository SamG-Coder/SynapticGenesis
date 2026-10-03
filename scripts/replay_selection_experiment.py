"""Declared retention comparison with equal exposure and equal live-time arms."""
import argparse
import json
from pathlib import Path
import subprocess

from experiment_checkpoint import state_record
from narrative_experiment import assess_books, samples
from native_experiment import NativeCommands, binding_scores, read, sha, verified_book_manifest, verified_manifest
from replay_priority_experiment import admitted_inputs


def write(path, value):
    Path(path).write_bytes((json.dumps(value, indent=2) + '\n').encode('utf-8'))


def run(args):
    schedule, bases, inputs = admitted_inputs()
    prior = read('reports/teacher-retention-language.json')['protocol']
    binding, books = Path(prior['binding_directory']), Path(prior['prepared_directory'])
    assert sha(binding / 'manifest.json') == prior['binding_manifest_sha256']
    verified_manifest(binding)
    verified_book_manifest(books, Path('data/sources-stories-v1.json'))
    assert sha(books / 'manifest.json') == prior['prepared_manifest_sha256']
    for path in [binding / 'manifest.json', books / 'manifest.json', binding / 'development.sgprobe',
                 *(books / f'{book}.txt' for book in (13853, 12228, 11757)), args.native, args.probe]:
        inputs[str(path)] = sha(path)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    protocol = dict(status='declared_before_execution', bases=bases, authenticated_inputs=inputs,
        schedule=str(schedule), start=160000, update_end=164096, time_budget_seconds=6,
        budgets=['updates', 'seconds'], modes=['none', 'uniform', 'interference'], candidate_seed=42,
        candidate_policy='Preview the real uniform stage/window choice; draw one different stored window '
                         'of the same length from that stage using a separate observation-indexed RNG. '
                         'Replace uniform only when the alternative has larger positive weighted loss increase.',
        checkpoint_policy='Ordinary native state plus a checksummed .sgpriority policy sidecar; no promotion.',
        book_batches=32, sample_bytes=192, probe_sha256=prior['probe_sha256'],
        source_sha256={p: sha(p) for p in ['src/live.cuh', 'experiments/replay_candidate_scoring.cuh',
            'experiments/replay_score_graph.cuh', 'experiments/replay_two_choice.cuh',
            'experiments/replay_selection_identity.cuh', 'experiments/replay_selection_probe.cu',
            'scripts/replay_selection_experiment.py', 'tests/replay_selection_probe.py',
            'tests/replay_selection_experiment.py']},
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        hardware=subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version,memory.total',
                                         '--format=csv,noheader'], text=True).strip(),
        primary='Complete development binding-group accuracy, priority minus both uniform controls, for each seed and budget.',
        gate='Priority must improve binding accuracy over both controls in all three seeds under each budget, '
             'lose no more than 0.02 nats/byte on each of three development books against either control, '
             'and not reduce binding accuracy from its ancestor. A pilot engineering gate, not general reliability.',
        timing='Native loop only, including scoring/capture and scheduled speech. Excludes model/source load, '
               'final save and all assessments. Time arms stop after the first completed tick reaching six seconds.',
        limitations='One run per seed/mode/budget. Repeated development material; no reserved tests or quality tuning. '
                    'Immediate interference may not predict durable retention. Native threshold sensitivity persists.',
        generated_text_targets=False, reserved_tests_scored=False, other_gpu_contexts_active=True)
    write(out / 'protocol.json', protocol)
    native_dir, probe_dir = out / 'native', out / 'probe'
    native_dir.mkdir(); probe_dir.mkdir()
    native, probe = NativeCommands(args.native, native_dir), NativeCommands(args.probe, probe_dir)
    rows, ancestors = [], []

    def assess(model, directory, endpoint):
        initial = sha(model)
        result = dict(state=state_record(model), books=assess_books(native, model, books, directory, endpoint, 32),
                      samples=samples(native, model, directory, endpoint, 192))
        result.update(binding_scores(native, model, binding, directory, endpoint, splits=('development',)))
        assert sha(model) == initial
        return result

    for index, base in enumerate(bases):
        seed, ancestor = base['seed'], Path(base['checkpoint'])
        directory = out / f'{seed}-ancestor'
        directory.mkdir()
        ancestors.append(dict(seed=seed, **assess(ancestor, directory, 160000)))
        write(out / 'ancestors.json', ancestors)
        for budget_index, budget in enumerate(protocol['budgets']):
            modes = protocol['modes']
            rotation = (index + budget_index) % 3
            for mode in modes[rotation:] + modes[:rotation]:
                directory = out / f'{seed}-{budget}-{mode}'
                probe('run', '--checkpoint', ancestor, '--curriculum', schedule, '--out', directory,
                      '--updates', 164096, '--seconds', 6 if budget == 'seconds' else 0,
                      '--mode', mode, '--seed', 42)
                session = read(directory / 'result.json')
                assert session['end'] == 164096 if budget == 'updates' else session['end'] < 164096
                if budget == 'seconds':
                    assert 6 <= session['live_seconds'] < 6 + session['max_tick_ms'] / 1000 + .003
                model = directory / 'latest.ckpt'
                row = dict(seed=seed, budget=budget, mode=mode, directory=directory.name,
                           session=session, **assess(model, directory, session['end']))
                rows.append(row)
                write(out / 'runs.json', rows)
                print(seed, budget, mode, 'end', session['end'], 'binding', row['development']['joint_accuracy'],
                      'overrides', session['overrides'], 'seconds', session['live_seconds'], flush=True)
            if budget == 'updates':
                assert sha(out / f'{seed}-updates-none/latest.ckpt') == sha(out / f'{seed}-updates-uniform/latest.ckpt')
    assert all(sha(p) == h for p, h in inputs.items())
    assert all(sha(p) == h for p, h in protocol['source_sha256'].items())
    write(out / 'comparison.json', dict(complete=True, protocol=protocol, ancestors=ancestors, runs=rows,
        native_commands=len(native.commands), learning_commands=len(probe.commands), inputs_unchanged=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--native', type=Path, default=Path('build/synapticgenesis.exe'))
    parser.add_argument('--probe', type=Path, default=Path('build/replay-selection-probe/synaptic-replay-selection-probe.exe'))
    parser.add_argument('--out', type=Path, required=True)
    run(parser.parse_args())
