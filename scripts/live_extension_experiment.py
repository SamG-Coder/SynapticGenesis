"""Teach an existing reading/science founder with two later curriculum admissions.

Uses the already selected binding-v2 lessons. Model computation is native CUDA;
Python prepares sources, invokes the executable and records bounded evidence.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import struct
import subprocess

from extend_curriculum import prepare
from prepare_lessons import write_probes


def checkpoint_state(path):
    raw = path.read_bytes()
    meta = struct.unpack_from('<32Q', raw)
    extra = struct.unpack_from(f'<{meta[31]}Q', raw, 288 + 12 * meta[14] + 4 * meta[18])
    return dict(sha256=hashlib.sha256(raw).hexdigest(), parameters=meta[14], cell=meta[1],
                online_updates=meta[24], optimizer_updates=meta[7], observed_pairs=meta[22],
                generated_bytes=meta[30], curriculum_hash=str(extra[14]),
                curriculum_stage=extra[15]+1, replay_updates=extra[6], replay_pairs=extra[7],
                replay_seen=extra[5], speech_every=meta[26], speech_length=meta[27])


def run(exe, out, source, replay_every=4):
    exe = exe.resolve()
    before = checkpoint_state(source)
    assert before['online_updates'] == 12000 and before['curriculum_stage'] == 2
    out.mkdir(parents=True, exist_ok=False)
    learner = out / 'learner'
    selected = Path('data/prepared/binding-v2')
    # Every single-fact question is a selected training example. The paired
    # reversals test context use on that training set, not novel generalization.
    spec_path = Path('data/lessons-binding-v2.json')
    spec = json.loads(spec_path.read_text())
    training = set((selected / 'prerequisites.dat').read_bytes().split(b'\x1e'))
    probes = []
    for obj, locs, style, query in itertools.product(spec['objects'], itertools.combinations(spec['locations'], 2),
                                                    range(len(spec['facts'])), range(len(spec['queries']))):
        group = f'{obj}-{locs[0]}-{locs[1]}-{style}-{query}'
        for reverse in (0, 1):
            context = spec['facts'][style].format(object=obj, location=locs[reverse]) + '\n'
            question = spec['queries'][query].format(object=obj)
            answers = [spec['answer'].format(location=loc) for loc in locs]
            assert (context + question + answers[reverse] + '\n').encode() in training
            probes.append(dict(id=f'{group}-{reverse}', pair=group, skill='single-fact-training-fit',
                               context=context, query=question, choice0=answers[0], choice1=answers[1], correct=reverse))
    probe_path = out / 'prerequisites.sgprobe'
    write_probes(probe_path, probes)
    calls = 0

    def native(*args):
        nonlocal calls
        calls += 1
        p = subprocess.run([str(exe), *map(str, args)], capture_output=True)
        (out / f'command-{calls:02d}.log').write_bytes(p.stdout + p.stderr)
        if p.returncode:
            raise RuntimeError(f'{args}: {p.stdout}\n{p.stderr}')

    def assess(path, label):
        record = dict(label=label, checkpoint=checkpoint_state(path))
        for name, suite in [('single_fact_training_fit', probe_path),
                            ('binding_development', selected / 'development.sgprobe')]:
            target = out / f'{label}-{name}.json'
            native('language-probes', '--checkpoint', path, '--probes', suite, '--output', target)
            record[name] = {k: v for k, v in json.loads(target.read_text()).items() if k != 'results'}
        print(f'{label}: single-fact pairs={record["single_fact_training_fit"]["joint_accuracy"]:.4f}, '
              f'binding groups={record["binding_development"]["joint_accuracy"]:.4f}', flush=True)
        return record

    rows = [assess(source, 'before')]
    holdouts = [Path('data/prepared/development-v2-final/13853.txt'),
                Path('data/prepared/development-v2-final/12228.txt')]
    previous = Path('data/curricula/reading-to-science.sg')
    current = source
    manifests = []
    for label, filename, end in [('prerequisites', 'prerequisites.dat', 16000), ('binding', 'binding.dat', 40000)]:
        prepared = out / f'data-{label}'
        manifests.append(prepare(previous, selected / filename, prepared, end,
                                 rate_scale=.25, scope='new', answer_scale=64, holdouts=holdouts))
        following = prepared / 'curriculum.sg'
        native('live', '--resume', current, '--curriculum', previous, '--extend-curriculum', following,
               '--out', learner, '--prompt', 'The bird ', '--validation', holdouts[0],
               '--eval-batches', 32, '--log-every', 4000, '--replay-every', replay_every)
        current = learner / 'latest.ckpt'
        record = assess(current, label)
        record['session'] = json.loads((learner / 'session.json').read_text())
        assert record['checkpoint']['online_updates'] == end
        new_replays = end // replay_every - before['online_updates'] // replay_every
        assert record['checkpoint']['optimizer_updates'] == before['optimizer_updates'] + end - 12000 + new_replays
        assert record['checkpoint']['replay_updates'] == before['replay_updates'] + new_replays
        rows.append(record)
        (out / 'partial.json').write_text(json.dumps(rows, indent=2) + '\n')
        previous = following
    native('decode-bench', '--checkpoint', current, '--tokens', 1024, '--rounds', 7, '--out', out / 'decode')
    events = [json.loads(row) for row in (learner / 'metrics.jsonl').read_text().splitlines()]
    extensions = [r for r in events if r.get('event') == 'curriculum_extension']
    assert len(extensions) == 2 and [r['online_update'] for r in extensions] == [12000, 16000]
    assert [r['replay_updates'] for r in extensions] == [3000, 3000 + 4000 // replay_every]
    assert all(r['replay_windows_preserved'] == 1024 for r in extensions)
    assert before == checkpoint_state(source), 'Original founder checkpoint was changed'
    result = dict(protocol='docs/live-extension-experiment.md', seed=1337, native_commands=calls,
                  training_replay_every=replay_every,
                  source_checkpoint=str(source), source_unchanged=True, imported_model_weights=False,
                  executable_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),
                  lesson_spec_sha256=hashlib.sha256(spec_path.read_bytes()).hexdigest(),
                  existing_selected_material_only=True, reserved_test_evaluated=False,
                  admissions=manifests, extension_events=extensions, stages=rows,
                  decode=json.loads((out / 'decode/benchmark.json').read_text()),
                  limits='One existing LIF founder. Single-fact probes measure fit to the selected training '
                         'lessons; stronger binding probes use development pairs. No causal curriculum '
                         'advantage, general dialogue or biological learning claim.')
    (out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(result=str(out / 'result.json'), commands=calls,
                          reader_loss=rows[-1]['session']['final_validation_loss']), indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--exe', type=Path, default=Path('build/synapticgenesis.exe'))
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, default=Path('runs/development-replay-v2-seed-1337/latest.ckpt'))
    p.add_argument('--replay-every', type=int, choices=[1, 4], default=4)
    a = p.parse_args()
    run(a.exe, a.out, a.checkpoint, a.replay_every)
