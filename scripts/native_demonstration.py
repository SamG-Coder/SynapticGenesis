"""Record actual generation speed, fixed prompts and one bounded live continuation."""
import argparse
from pathlib import Path
import shutil
import subprocess
import time

from adaptation_sources import verified_edition
from experiment_checkpoint import checkpoint, state_record
from extend_curriculum import prepare
from native_experiment import NativeCommands, read, sha, write


PROMPTS = [
    ('location', 'The toy is in the bag. The tag is in the box.\nWhere is the toy?\nAnswer: '),
    ('location_reversed', 'The toy is in the box. The tag is in the bag.\nWhere is the toy?\nAnswer: '),
    ('arithmetic', 'Question: What is 2 + 2?\nAnswer: '),
    ('everyday_fact', 'Question: What color is the sky?\nAnswer: '),
    ('explanation', 'Question: Why does ice melt?\nAnswer: '),
    ('story', 'Once upon a time '),
]


def run(args):
    model = Path('runs/teacher-retention-panel/1337-associative-control/checkpoint-190000.ckpt').resolve()
    original_schedule = Path('runs/teacher-retention-panel/edition/curriculum.sg').resolve()
    previous = read('runs/teacher-retention-panel/comparison.json')
    recorded = next(row for row in previous['runs'] if row['seed'] == 1337
                    and row['arm'] == 'associative-control' and row['online_updates'] == 190000)
    assert sha(model) == recorded['checkpoint_sha256']
    assert sha(original_schedule) == previous['protocol']['source_admission']['schedule_sha256']
    edition = Path('data/prepared/early-readers-v1').resolve()
    manifest = verified_edition(edition, Path('data/sources-early-readers-v1.json'))
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    end = 194096
    admission = prepare(original_schedule, edition / 'train.dat', out / 'curriculum', end,
                        holdouts=[edition / 'validation.dat', edition / 'test.dat'])
    exe_sha, ancestor_sha = sha(args.exe), sha(model)
    protocol = dict(kind='performance_and_output_demonstration_not_learning_quality_experiment',
        ancestor=str(model), ancestor_sha256=ancestor_sha, ancestor_state=state_record(model),
        executable_sha256=exe_sha,
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        script_sha256=sha(__file__),
        hardware=subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version,memory.total',
                                         '--format=csv,noheader'], text=True).strip(),
        benchmark_bytes=1024, benchmark_rounds=7, benchmark_warmup_rounds=1,
        benchmark_prompt='The bird ', benchmark_math='strict FP32',
        benchmark_scope='Resident decode includes sampling, transfers and synchronization; excludes load, graph capture, prompt prefill and output I/O.',
        sample_prompts=[dict(name=name, text=text) for name, text in PROMPTS],
        sample_bytes=192, sample_seed=42, sample_temperature=.8, sample_top_k=40,
        samples_are_continuations_not_instruction_tuned_answers=True,
        sample_wall_scope='Whole native process: launch, CUDA/model load, prompt, generation, output and exit; not disk-cold.',
        live_start=190000, live_end=end, live_replay_every=4, live_chunk=128,
        live_math='Inherited TF32 training and ordered gradient reductions',
        live_scope='Native loop includes source learning, replay, scheduled speech, logging and final save. Process time also includes startup and source admission.',
        live_source=manifest, admission=admission, live_prompt='The bird ',
        other_gpu_contexts_active=True, reserved_tests_scored=False,
        limits='One learned seed on one desktop GPU. Timing variability is retained. No learning-quality or general-language claim.')
    write(out / 'protocol.json', protocol)
    commands = out / 'commands'
    commands.mkdir()
    native = NativeCommands(args.exe, commands)
    native('decode-bench', '--checkpoint', model, '--out', out / 'decode',
           '--tokens', 1024, '--rounds', 7, '--prompt', 'The bird ')
    decode = read(out / 'decode/benchmark.json')
    decode['graph_bytes_per_second'] = 1e6 / decode['graph_us_per_byte']
    decode['regular_bytes_per_second'] = 1e6 / decode['regular_us_per_byte']
    decode['graph_round_bytes_per_second'] = [1e6 / t for t in decode['graph_rounds_us']]
    print('decode', decode['graph_bytes_per_second'], 'byte tokens / second', flush=True)
    outputs = []
    for name, prompt in PROMPTS:
        path = out / f'{name}.txt'
        started = time.perf_counter()
        native('sample', '--checkpoint', model, '--prompt', prompt, '--tokens', 192,
               '--temperature', .8, '--top-k', 40, '--seed', 42, '--graph', '--output', path)
        wall = time.perf_counter() - started
        raw = path.read_bytes()
        assert raw.startswith(prompt.encode('ascii')) and len(raw) == len(prompt) + 192
        outputs.append(dict(name=name, prompt=prompt, generated_bytes=192,
            continuation=raw[len(prompt):].decode('utf-8', errors='backslashreplace'),
            raw_sha256=sha(path), process_seconds=wall, process_bytes_per_second=192 / wall))
        write(out / 'outputs.json', outputs)
        print(name, repr(outputs[-1]['continuation']), flush=True)
    live = out / 'live'
    started = time.perf_counter()
    native('live', '--resume', model, '--curriculum', original_schedule,
           '--extend-curriculum', out / 'curriculum/curriculum.sg',
           '--out', live, '--updates', end, '--prompt', 'The bird ',
           '--log-every', 1024, '--save-every', 4096)
    live_wall = time.perf_counter() - started
    session = read(live / 'session.json')
    final = state_record(live / 'latest.ckpt')
    initial = protocol['ancestor_state']
    assert final['online_updates'] - initial['online_updates'] == 4096
    assert final['replay_updates'] - initial['replay_updates'] == 1024
    assert final['global_updates'] - initial['global_updates'] == 5120
    assert session['shared_weights'] and session['persistent_membranes'] and not session['trains_on_generated_text']
    assert sha(args.exe) == exe_sha and sha(model) == ancestor_sha
    verified_edition(edition, Path('data/sources-early-readers-v1.json'))
    all_pairs = final['observed_pairs'] - initial['observed_pairs'] + final['replay_pairs'] - initial['replay_pairs']
    report = dict(protocol=protocol, decode=decode, samples=outputs, live_session=session,
        live_final_state=final, live_process_seconds=live_wall,
        live_new_source_pairs_per_second=session['session_observed_pairs'] / session['elapsed_seconds'],
        live_source_and_replay_pairs_per_second=all_pairs / session['elapsed_seconds'],
        live_observation_chunks_per_second=4096 / session['elapsed_seconds'],
        live_optimizer_updates_per_second=5120 / session['elapsed_seconds'],
        command_count=len(native.commands), ancestor_unchanged=True, executable_unchanged=True,
        live_transcript=(live / 'transcript.txt').read_bytes().decode('utf-8', errors='backslashreplace'))
    write(out / 'report.json', report)
    print('live', report['live_new_source_pairs_per_second'], 'new source byte targets / second', flush=True)
    print(out / 'report.json', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe', type=Path, default=Path('build/synapticgenesis.exe'))
    p.add_argument('--out', type=Path, required=True)
    run(p.parse_args())
