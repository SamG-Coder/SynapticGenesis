"""Compare captured and ordinary scoring on identical admitted live trajectories."""
import argparse
import json
from pathlib import Path
import statistics
import subprocess

from native_experiment import NativeCommands, read, sha
from replay_priority_experiment import admitted_inputs


def write(path, value):
    Path(path).write_bytes((json.dumps(value, indent=2) + '\n').encode('utf-8'))


def run(args):
    schedule, bases, inputs = admitted_inputs()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    native = NativeCommands(args.probe, out)
    inputs[str(args.probe)] = sha(args.probe)
    previous = read('reports/replay-priority-comparison.json')
    protocol = dict(status='declared_before_execution', bases=bases, inputs=inputs,
        previous_comparison_sha256=sha('reports/replay-priority-comparison.json'),
        source_sha256={p: sha(p) for p in ['experiments/replay_priority_probe.cu',
            'experiments/replay_priority_scoring.cuh', 'experiments/replay_score_graph.cuh',
            'scripts/replay_score_graph_experiment.py', 'tests/replay_priority_probe.py']},
        start=160000, end=160512, every=32, per_group=8, seed=42, rounds=2,
        timing='Native live loop includes scoring, graph construction, learning, replay and speech; '
               'excludes initial load and final save. Each condition starts a fresh process.',
        hardware=subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version,memory.total',
                                         '--format=csv,noheader'], text=True).strip(),
        other_gpu_contexts_active=True, generated_text_targets=False, reserved_tests_scored=False,
        criterion='Entire score journals, snapshots from fixtures, final checkpoints and speech must match ordinary scoring.',
        original_cpu_score_check_remains_failed=True)
    write(out / 'protocol.json', protocol)
    rows = []
    for index, base in enumerate(bases):
        seed = base['seed']
        old = next(r for r in previous['runs'] if r['seed'] == seed and r['repeat'] == 0 and r['mode'] == 'measured')
        old_journal = [json.loads(line) for line in Path(f'reports/replay-priority-scores-{seed}.jsonl').read_text().splitlines()]
        for repeat in range(2):
            modes = ['disabled', 'ordinary', 'graph']
            order = modes[(index + repeat) % 3:] + modes[:(index + repeat) % 3]
            for mode in order:
                directory = out / f'{seed}-{repeat}-{mode}'
                native('run', '--checkpoint', base['checkpoint'], '--curriculum', schedule, '--out', directory,
                       '--updates', 160512, '--every', 0 if mode == 'disabled' else 32,
                       '--per-group', 8, '--seed', 42, '--graph-scoring', int(mode == 'graph'))
                assert sha(directory / 'latest.ckpt') == old['final']['checkpoint_sha256']
                journal = [json.loads(line) for line in (directory / 'scores.jsonl').read_text().splitlines()]
                assert journal == ([] if mode == 'disabled' else old_journal)
                old_speech = Path(f'runs/replay-priority-panel/{seed}-0-measured/speech.txt')
                assert (directory / 'speech.txt').read_bytes() == old_speech.read_bytes()
                row = dict(seed=seed, repeat=repeat, mode=mode, result=read(directory / 'result.json'),
                           checkpoint_sha256=sha(directory / 'latest.ckpt'), scores_sha256=sha(directory / 'scores.jsonl'))
                rows.append(row)
                write(out / 'runs.json', rows)
                print(seed, repeat, mode, row['result']['live_seconds'], flush=True)
    summary = []
    for base in bases:
        times = {mode: statistics.median(r['result']['live_seconds'] for r in rows
                                        if r['seed'] == base['seed'] and r['mode'] == mode)
                 for mode in ('disabled', 'ordinary', 'graph')}
        summary.append(dict(seed=base['seed'], median_seconds=times,
            graph_overhead_percent=100 * (times['graph'] / times['disabled'] - 1),
            total_loop_speedup=times['ordinary'] / times['graph']))
    assert all(sha(p) == h for p, h in inputs.items())
    assert all(sha(p) == h for p, h in protocol['source_sha256'].items())
    result = dict(complete=True, protocol=protocol, runs=rows, summary=summary,
                  all_old_and_new_scores_checkpoints_speech_exact=True,
                  original_cpu_score_check_passed=False, native_commands=len(native.commands))
    write(out / 'result.json', result)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--probe', type=Path, default=Path('build/replay-priority-probe/synaptic-replay-priority-probe.exe'))
    parser.add_argument('--out', type=Path, required=True)
    run(parser.parse_args())
