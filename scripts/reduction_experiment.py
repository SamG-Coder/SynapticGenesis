"""Compare native ordered reductions with a preserved pre-change executable.

Repeated identical inputs test reproducibility, not seed robustness or learning
quality. Native CUDA performs all training and generation.
"""
import argparse
from array import array
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import subprocess

from extend_curriculum import read_schedule
from prepare_lessons import quoted
from experiment_checkpoint import checkpoint


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def changed_tensors(config, first, second):
    c, h, layers = config['channels'], config['hidden'], config['layers']
    assert config['cell'] == 5
    spans = [('embedding', 256*c)]
    for layer in range(layers):
        spans += [(f'{layer}.{name}', count) for name, count in
                  [('gain',c),('input.weight',h*c),('input.bias',h),('output.weight',c*h),
                   ('output.bias',c),('leak',h),('trace_logit',h),('trace_scale',h),
                   ('gate.weight',h*c),('gate.bias',h)]]
    spans += [('final.gain',c),('head',256*c),('bias',256)]
    offset, changed = 0, []
    for name, count in spans:
        errors = [abs(x-y) for x,y in zip(first[offset:offset+count], second[offset:offset+count])]
        if max(errors):
            changed.append(dict(tensor=name, differing_parameters=sum(e != 0 for e in errors),
                                max_abs_error=max(errors)))
        offset += count
    assert offset == len(first) == len(second)
    return changed


def compare(a, b):
    ma, _, ca, ha = checkpoint(a)
    mb, _, cb, hb = checkpoint(b)
    assert ma[14] == mb[14] and len(ca) == len(cb)
    va, vb = array('f', ca), array('f', cb)
    p = ma[14]
    parts = {}
    for name, lo, hi in [('weights', 0, p), ('first_moment', p, 2*p),
                         ('second_moment', 2*p, 3*p), ('recurrence', 3*p, len(va))]:
        parts[name] = max((abs(x-y) for x, y in zip(va[lo:hi], vb[lo:hi])), default=0)
    return dict(checkpoint_sha256=[ha, hb], complete_checkpoint_identical=ha == hb,
                core_arrays_bitwise_identical=ca == cb, max_abs_errors=parts)


def run(exe, legacy, out, long_end):
    exe, legacy, out = exe.resolve(), legacy.resolve(), out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    binaries = {'legacy': legacy, 'ordered': exe}
    identities = {name: sha(path) for name, path in binaries.items()}
    selected = Path('data/prepared/binding-v2')
    manifest = json.loads((selected/'manifest.json').read_text())
    for name, record in manifest['files'].items():
        assert sha(selected/name) == record['sha256'], name
    _, stages = read_schedule(selected/'curriculum.sg')
    stages[-1]['end_update'] = long_end
    schedule = out / 'curriculum.sg'
    schedule.write_text('SGCURRICULUM3\n' + '\n'.join(
        f'{s["end_update"]} {quoted(s["source"].as_posix())} {s["rate"]} {s["scope"]} {s["answer"]}'
        for s in stages) + '\n')
    protocol = dict(executable_sha256=identities, cell='selective', channels=256,
                    hidden=512, layers=4, seed=1337, repeated_fresh_runs=3,
                    first_stage_observations=6000, long_endpoint=long_end,
                    restart_at=17403, replay_slots=1024, replay_every=4, chunk=128,
                    batched_shapes=[[8,128],[16,256]], batched_steps=400,
                    tf32=True, graph_speech=True, source_hashes=[sha(s['source']) for s in stages],
                    selected_spec_sha256=manifest['spec_sha256'],
                    reserved_test_evaluated=False,
                    limits='Same device/toolchain only. Repeats measure arithmetic reproducibility, '
                           'not robustness across seeds or better learning. Legacy and ordered '
                           'sums differ in rounding and can follow different learned trajectories.',
                    timing='Native elapsed_seconds includes live updates, speech, replay, logging and '
                           'checkpoint I/O; excludes setup and held-out evaluation. Rotated binary order.')
    (out/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    calls = 0

    def native(which, *args):
        nonlocal calls
        calls += 1
        result = subprocess.run([str(binaries[which]), *map(str, args)], capture_output=True)
        (out/f'command-{calls:03d}.log').write_bytes(result.stdout+result.stderr)
        if result.returncode:
            raise RuntimeError(f'{which} {args}: {result.stderr!r}')

    rows, repeat_comparisons, policy_controls, gradient_repeats = [], {}, {}, {}
    common = ['--curriculum', schedule, '--cell', 'selective', '--seed', 1337,
              '--lr', .0003, '--chunk', 128, '--replay-capacity', 1024, '--replay-every', 4,
              '--graph', '--speak-every', 500, '--tokens', 96, '--fast']
    evaluation = ['--prompt', 'The bird ', '--validation', 'data/prepared/development-v2-final/13853.txt',
                  '--eval-batches', 32, '--log-every', 6000, '--save-every', 5000]
    # Same frozen fixture and cached forward pass: exposes tiny gradient changes
    # separately from amplified differences after thousands of learned updates.
    for which in binaries:
        folders = [out/f'{which}-gradient-{r}' for r in range(2)]
        for folder in folders:
            native(which, 'selective-test', '--out', folder)
        differences = {}
        config = json.loads((folders[0]/'fixture.json').read_text())
        for name in ('weights', 'logits', 'gradients', 'gradients_regularized', 'updated'):
            raw = [(folder/f'{name}.f32').read_bytes() for folder in folders]
            values = [array('f', item) for item in raw]
            differences[name] = dict(bitwise_identical=raw[0] == raw[1],
                                      max_abs_error=max(abs(a-b) for a,b in zip(*values)))
            if name in ('gradients','gradients_regularized'):
                differences[name]['changed_tensors'] = changed_tensors(config, *values)
        gradient_repeats[which] = differences

    for repetition in range(3):
        order = ('legacy', 'ordered') if repetition % 2 == 0 else ('ordered', 'legacy')
        for which in order:
            dest = out/f'{which}-fresh-{repetition}'
            native(which, 'live', '--out', dest, '--updates', 6000, '--replay', 'reservoir',
                   *common, *evaluation)
            session = json.loads((dest/'session.json').read_text())
            assert session['online_updates'] == 6000
            row = dict(binary=which, repetition=repetition, session=session,
                       checkpoint_sha256=sha(dest/'latest.ckpt'))
            rows.append(row)
            (out/'partial.json').write_text(json.dumps(rows, indent=2)+'\n')
            print(f'{which} repeat {repetition}: loss={session["final_validation_loss"]:.6f} '
                  f'loop={session["elapsed_seconds"]:.3f}s', flush=True)
    for which in binaries:
        reference = out/f'{which}-fresh-0/latest.ckpt'
        repeat_comparisons[which] = [compare(reference, out/f'{which}-fresh-{r}/latest.ckpt') for r in (1,2)]
        dest = out/f'{which}-one-stage'
        native(which, 'live', '--out', dest, '--updates', 6000, '--replay', 'stage', *common, *evaluation)
        policy_controls[which] = compare(reference, dest/'latest.ckpt')
    assert all(row['complete_checkpoint_identical'] for row in repeat_comparisons['ordered'])
    assert policy_controls['ordered']['core_arrays_bitwise_identical']
    assert all(v['bitwise_identical'] for v in gradient_repeats['ordered'].values())

    # An actual old executable checkpoint is admitted without resetting learned
    # weights/moments/history. Both new continuations start with that same file.
    ancestor = out/'legacy-fresh-0/latest.ckpt'
    ancestor_sha = sha(ancestor)
    continuation = []
    for arm, stops in [('continuous', [long_end]), ('resumed', [17403, long_end])]:
        dest = out/arm
        source = ancestor
        for index, endpoint in enumerate(stops):
            convert = ['--replay', 'stage'] if index == 0 else []
            native('ordered', 'live', '--resume', source, '--out', dest, '--curriculum', schedule,
                   '--updates', endpoint, *convert, *evaluation)
            session = json.loads((dest/'session.json').read_text())
            assert session['online_updates'] == endpoint
            snapshot = dest/f'checkpoint-{endpoint}.ckpt'
            shutil.copyfile(dest/'latest.ckpt', snapshot)
            continuation.append(dict(arm=arm, endpoint=endpoint, session=session, checkpoint_sha256=sha(snapshot)))
            source = snapshot
            print(f'{arm} endpoint {endpoint}: loss={session["final_validation_loss"]:.6f}', flush=True)
    restart = compare(out/'continuous/latest.ckpt', out/'resumed/latest.ckpt')
    assert restart['complete_checkpoint_identical'], restart
    assert sha(ancestor) == ancestor_sha
    batched = []
    for batch, context in protocol['batched_shapes']:
        for repetition in range(3):
            order = ('legacy', 'ordered') if repetition % 2 == 0 else ('ordered', 'legacy')
            for which in order:
                dest = out/f'batch-{batch}-{context}-{which}-{repetition}'
                native(which, 'train', '--out', dest, '--data', stages[0]['source'],
                       '--validation', 'data/prepared/development-v2-final/13853.txt', '--cell', 'selective',
                       '--batch', batch, '--context', context, '--steps', 400, '--warmup', 50,
                       '--lr', .0003, '--seed', 1337, '--fast', '--eval-every', 400,
                       '--eval-batches', 4, '--save-every', 400)
                metrics = [json.loads(line) for line in (dest/'metrics.jsonl').read_text().splitlines()]
                final = metrics[-1]
                assert final['step'] == 400
                batched.append(dict(binary=which, batch=batch, context=context, repetition=repetition,
                                    final_metrics=final, checkpoint_sha256=sha(dest/'latest.ckpt')))
        repeated = [r['checkpoint_sha256'] for r in batched if r['batch']==batch and r['binary']=='ordered']
        assert len(set(repeated)) == 1, 'Batched repeats diverged'
        print(f'Batch {batch} context {context}: repeated ordered checkpoints identical', flush=True)
    decode = {}
    for which in binaries:
        dest = out/f'decode-{which}'
        native(which, 'decode-bench', '--checkpoint', ancestor, '--tokens', 1024, '--rounds', 7, '--out', dest)
        decode[which] = json.loads((dest/'benchmark.json').read_text())
    costs = {}
    for which in binaries:
        selected = [r['session']['elapsed_seconds'] for r in rows if r['binary'] == which]
        costs[which] = dict(seconds=selected, median_seconds=statistics.median(selected))
        assert sha(binaries[which]) == identities[which]
    result = dict(protocol=protocol, native_commands=calls, frozen_gradient_repeats=gradient_repeats,
                  fresh_runs=rows, fresh_repeat_comparisons=repeat_comparisons,
                  one_stage_policy_controls=policy_controls, timing=costs,
                  long_continuations=continuation, restart_comparison=restart,
                  batched_runs=batched, decode_from_identical_legacy_checkpoint=decode,
                  legacy_ancestor_checkpoint_sha256=ancestor_sha, legacy_ancestor_unchanged=True)
    (out/'comparison.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exe', type=Path, default=Path('build/synapticgenesis.exe'))
    parser.add_argument('--legacy', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--long-end', type=int, default=130000)
    args = parser.parse_args()
    if args.long_end < 34000:
        parser.error('Long endpoint must include the complete three-stage curriculum')
    run(args.exe, args.legacy, args.out, args.long_end)
