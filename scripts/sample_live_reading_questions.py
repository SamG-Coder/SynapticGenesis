"""Display the earlier six demonstration prompts on every final live condition.

These are illustrative continuations, not an additional scored quality gate.
Every declared final model is included; no samples are selected or retried.
"""
import argparse
from pathlib import Path

from native_demonstration import PROMPTS
from native_experiment import NativeCommands, read, sha, write


def run(root, exe):
    data = read(root / 'comparison.json')
    p = data['protocol']
    assert data['complete'] and sha(exe) == p['executable_sha256']
    previous = read('reports/native-demonstration-20261003.json')
    assert [dict(name=n, text=t) for n, t in PROMPTS] == previous['protocol']['sample_prompts']
    output = root / 'questions'
    output.mkdir(parents=True, exist_ok=False)
    native = NativeCommands(exe, output)
    protocol = dict(comparison_sha256=sha(root / 'comparison.json'), script_sha256=sha(__file__),
        prompt_script_sha256=sha('scripts/native_demonstration.py'),
        prompts=previous['protocol']['sample_prompts'], seeds=p['seeds'], arms=p['arms'],
        end=p['online_endpoints'][-1], generated_bytes=192, temperature=.8, top_k=40, seed=42,
        sampling_math='strict FP32; CUDA graph decode; fresh context',
        limits='Post-training illustrative continuations using all six prior prompts on all final conditions. '
               'No training, retries, sample filtering, instruction-tuning claim or additional scored gate.')
    write(output / 'protocol.json', protocol)
    results = []
    for seed in p['seeds']:
        for arm in p['arms']:
            model = root / f'{seed}-{arm}' / f"checkpoint-{protocol['end']}.ckpt"
            before = sha(model)
            row = next(r for r in data['runs'] if (r['seed'], r['arm'], r['online_updates']) == (seed, arm, protocol['end']))
            assert row['checkpoint_sha256'] == before
            samples = []
            for name, prompt in PROMPTS:
                path = output / f'{seed}-{arm}-{name}.txt'
                native('sample', '--checkpoint', model, '--prompt', prompt, '--tokens', 192,
                       '--temperature', .8, '--top-k', 40, '--seed', 42, '--graph', '--output', path)
                raw = path.read_bytes()
                assert raw.startswith(prompt.encode()) and len(raw) == len(prompt) + 192
                samples.append(dict(name=name, prompt=prompt, file_sha256=sha(path),
                    continuation=raw[len(prompt):].decode('utf-8', errors='backslashreplace')))
            assert sha(model) == before
            results.append(dict(seed=seed, arm=arm, checkpoint_sha256=before, samples=samples))
            write(output / 'partial.json', results)
            print(seed, arm, 'all six fixed prompts saved', flush=True)
    result = dict(protocol=protocol, complete=True, commands=len(native.commands),
                  models_unchanged=True, models=results)
    write(root / 'questions.json', result)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--exe', type=Path, default=Path('build/synapticgenesis.exe'))
    a = p.parse_args()
    run(a.root.resolve(), a.exe.resolve())
