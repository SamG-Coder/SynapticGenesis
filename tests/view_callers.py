"""Compare shared-view CLI callers with a preserved native runtime.

All learning uses disposable synthetic fixtures. host-test only prepares and
checks the protocol/comparators; run executes native CUDA commands sequentially.
"""
import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from native_experiment import NativeCommands, sha
from prepare_lessons import write_probes

CELLS = ('lif', 'alif', 'trace', 'gated', 'selective', 'associative')
TIMINGS = {
    'probes': ('elapsed_seconds',),
    'decode': ('graph_setup_ms', 'regular_us_per_byte', 'graph_us_per_byte',
               'speedup', 'regular_rounds_us', 'graph_rounds_us'),
    'memory': ('training_seconds',),
    'context': (),
}


def write(path, value):
    Path(path).write_bytes((json.dumps(value, indent=2) + '\n').encode())


def comparable(value, kind):
    result = deepcopy(value)
    for key in TIMINGS[kind]:
        if key not in result:
            raise ValueError(f'Missing {kind} timing field: {key}')
        del result[key]
    return result


def fixtures(out):
    train, val = out / 'train.dat', out / 'validation.dat'
    train.write_bytes(b'The child counts seeds. A bird sits by the red gate. ' * 32)
    val.write_bytes(b'A teacher has a letter. The tree is beside a blue wall. ' * 24)
    suites = []
    for version in (1, 2):
        rows = []
        for group in range(8):
            # Distinct lengths force repeated eviction from a four-shape cache.
            padding = 'It is a calm day. ' * group
            for swap in (0, 1):
                context = padding + ('The key is in the cup. The hat is in the box. '
                                     if swap else 'The key is in the box. The hat is in the cup. ')
                for query in range(version):
                    rows.append(dict(id=f'{group}-{swap}-{query}', pair=f'g{group}', skill='location',
                                     correct=swap ^ query, context=context,
                                     query=f'Where is the {"hat" if query else "key"}? Answer: ',
                                     choice0='box.', choice1='cup.'))
        lengths = {len(r['context'] + r['query'] + r['choice0']) - 1 for r in rows}
        assert len(lengths) > 4
        for index, ordered in enumerate((rows, list(reversed(rows)))):
            path = out / f'probes-v{version}-order{index}.sgprobe'
            write_probes(path, ordered, f'SGPROBE{version}')
            suites.append(path)
    return train, val, suites


def protocol(out, old_exe, new_exe):
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    train, val, suites = fixtures(out)
    calls, checks = [], []

    def command(runtime, label, *args):
        calls.append(dict(runtime=runtime, label=label, args=list(map(str, args))))

    def check(kind, left, right):
        checks.append(dict(kind=kind, left=str(left), right=str(right)))

    for cell in CELLS:
        for fast in (False, True):
            label = cell + ('-tf32' if fast else '-fp32')
            fast_flag = ['--fast'] if fast else []
            common = ['--data', train, '--validation', val, '--batch', 2, '--context', 16,
                      '--channels', 32, '--hidden', 64, '--layers', 2, '--cell', cell,
                      '--burn-in', 32, '--steps', 8, '--warmup', 0, '--seed', 17,
                      '--eval-every', 4, '--eval-batches', 2, '--save-every', 4, *fast_flag]
            for policy in ('warm', 'reset'):
                destinations = [out / f'{label}-{policy}-{side}' for side in ('old', 'new')]
                for side, dest in zip(('old', 'new'), destinations):
                    command(side, label, 'train', *common, '--burn-policy', policy, '--out', dest)
                check('bytes', destinations[0] / 'latest.ckpt', destinations[1] / 'latest.ckpt')
            # Both versions resume exactly the same learned checkpoint, inheriting
            # prefix policy and optimizer state. The schedule is matched per run.
            checkpoint = out / f'{label}-warm-old/latest.ckpt'
            for side in ('old', 'new'):
                command(side, label, 'train', '--resume', checkpoint, '--data', train, '--validation', val,
                        '--steps', 12, '--eval-every', 4, '--eval-batches', 2, '--save-every', 4,
                        '--out', out / f'{label}-resume-{side}')
            check('bytes', out / f'{label}-resume-old/latest.ckpt', out / f'{label}-resume-new/latest.ckpt')
            for side in ('old', 'new'):
                command(side, label, 'context-bench', '--checkpoint', checkpoint, '--data', val,
                        '--prefix', 32, '--context', 16, '--batch', 2, '--batches', 3,
                        '--output', out / f'{label}-context-{side}.json')
            check('context', out / f'{label}-context-old.json', out / f'{label}-context-new.json')
            for suite in suites:
                for side in ('old', 'new'):
                    command(side, label, 'language-probes', '--checkpoint', checkpoint, '--probes', suite,
                            '--output', out / f'{label}-{suite.stem}-{side}.json')
                check('probes', out / f'{label}-{suite.stem}-old.json', out / f'{label}-{suite.stem}-new.json')
            for version in (1, 2):
                for side in ('old', 'new'):
                    check('probe_order', out / f'{label}-probes-v{version}-order0-{side}.json',
                          out / f'{label}-probes-v{version}-order1-{side}.json')
            for side in ('old', 'new'):
                command(side, label, 'decode-bench', '--checkpoint', checkpoint, '--tokens', 64,
                        '--rounds', 2, '--prompt', 'The bird ', *fast_flag, '--out', out / f'{label}-decode-{side}')
            for name, kind in [('sample.txt', 'bytes'), ('benchmark.json', 'decode')]:
                check(kind, out / f'{label}-decode-old/{name}', out / f'{label}-decode-new/{name}')
            for side in ('old', 'new'):
                command(side, label, 'memory-bench', '--cell', cell, '--delay', 6, '--steps', 4,
                        '--batch', 2, '--channels', 32, '--hidden', 64, '--layers', 2, '--seed', 17,
                        *fast_flag, '--out', out / f'{label}-memory-{side}')
            for name, kind in [('result.json', 'memory'), ('synthetic-only.ckpt', 'bytes'), ('metrics.jsonl', 'bytes')]:
                check(kind, out / f'{label}-memory-old/{name}', out / f'{label}-memory-new/{name}')
    result = dict(status='declared_before_cuda_checks',
                  scope='Disposable native CLI ownership and compatibility controls, not language quality or speed.',
                  executables={side: dict(path=str(Path(exe).resolve()), sha256=sha(exe))
                               for side, exe in [('old', old_exe), ('new', new_exe)]},
                  fixtures={str(p): sha(p) for p in [train, val, *suites]},
                  cells=list(CELLS), precisions=['fp32', 'tf32'], commands=calls, comparisons=checks,
                  timing_fields_excluded=TIMINGS, new_learning_sources_admitted=False)
    write(out / 'protocol.json', result)
    return result


def verify(pair):
    left, right = Path(pair['left']), Path(pair['right'])
    if pair['kind'] == 'bytes':
        assert left.read_bytes() == right.read_bytes(), pair
    else:
        a = json.loads(left.read_text(encoding='utf-8'))
        b = json.loads(right.read_text(encoding='utf-8'))
        if pair['kind'] == 'probe_order':
            assert a['results'] == list(reversed(b['results'])), pair
            assert a['format'] == b['format'], pair
            for key in ('parameters_and_optimizer_unchanged', 'strict_fp32'):
                assert a[key] is True and b[key] is True, pair
        else:
            assert comparable(a, pair['kind']) == comparable(b, pair['kind']), pair


def host_test(out, old_exe, new_exe):
    plan = protocol(out, old_exe, new_exe)
    counts = Counter(c['args'][0] for c in plan['commands'])
    assert counts == {'train': 72, 'context-bench': 24, 'language-probes': 96,
                      'decode-bench': 24, 'memory-bench': 24}
    assert Counter(c['runtime'] for c in plan['commands']) == {'old': 120, 'new': 120}
    assert len(plan['comparisons']) == 204
    assert len({c['label'] for c in plan['commands']}) == 12
    # Comparator checks are CPU-only. A timing difference is admissible; a score,
    # sample, checkpoint or extra output field must not silently disappear.
    fixture = out / 'comparator-fixtures'
    fixture.mkdir()
    rejected = 0
    for kind, fields in TIMINGS.items():
        a = dict(result=[1.25, 2.5], generated='abc', **{k: 1 for k in fields})
        b = dict(a, **{k: 999 for k in fields})
        left, right = fixture / f'{kind}-left.json', fixture / f'{kind}-right.json'
        pair = dict(kind=kind, left=str(left), right=str(right))
        write(left, a); write(right, b)
        verify(pair)
        for mutation in ('score', 'sample', 'extra'):
            changed = deepcopy(b)
            if mutation == 'score': changed['result'][0] += .125
            elif mutation == 'sample': changed['generated'] = 'abd'
            else: changed['unexpected'] = True
            write(right, changed)
            try:
                verify(pair)
            except AssertionError:
                rejected += 1
            else:
                raise AssertionError('Comparator accepted changed substantive output')
        if fields:
            changed = dict(b)
            del changed[fields[0]]
            try:
                comparable(changed, kind)
            except ValueError:
                rejected += 1
            else:
                raise AssertionError('Comparator accepted missing timing field')
    left, right = fixture / 'checkpoint-a.bin', fixture / 'checkpoint-b.bin'
    left.write_bytes(b'abc\x00\xff'); right.write_bytes(left.read_bytes())
    pair = dict(kind='bytes', left=str(left), right=str(right))
    verify(pair)
    right.write_bytes(b'abc\x00\xfe')
    try:
        verify(pair)
    except AssertionError:
        rejected += 1
    else:
        raise AssertionError('Comparator accepted changed checkpoint byte')
    a = dict(format='SGPROBE1', results=[dict(id='a', score=1.), dict(id='b', score=2.)],
             parameters_and_optimizer_unchanged=True, strict_fp32=True)
    b = dict(a, results=list(reversed(a['results'])))
    left, right = fixture / 'order-a.json', fixture / 'order-b.json'
    pair = dict(kind='probe_order', left=str(left), right=str(right))
    write(left, a); write(right, b)
    verify(pair)
    for mutation in ('score', 'order', 'unchanged_flag'):
        changed = deepcopy(b)
        if mutation == 'score': changed['results'][0]['score'] = 3.
        elif mutation == 'order': changed['results'].reverse()
        else: changed['parameters_and_optimizer_unchanged'] = False
        write(right, changed)
        try:
            verify(pair)
        except AssertionError:
            rejected += 1
        else:
            raise AssertionError('Comparator accepted broken probe order independence')
    result = dict(passed=True, cuda_executed=False, native_commands_planned=len(plan['commands']),
                  comparisons_planned=len(plan['comparisons']), altered_comparisons_rejected=rejected,
                  protocol_sha256=sha(out / 'protocol.json'))
    write(out / 'host-result.json', result)
    print(json.dumps(result, indent=2))


def run(out, old_exe, new_exe):
    plan = protocol(out, old_exe, new_exe)
    runners = {}
    for side, exe in [('old', old_exe), ('new', new_exe)]:
        journal = out / f'{side}-commands'
        journal.mkdir()
        runners[side] = NativeCommands(exe, journal)
    checked = set()
    for index, command in enumerate(plan['commands']):
        for side, identity in plan['executables'].items():
            assert sha(identity['path']) == identity['sha256'], side
        runners[command['runtime']](*command['args'])
        for number, pair in enumerate(plan['comparisons']):
            if number not in checked and Path(pair['left']).exists() and Path(pair['right']).exists():
                verify(pair)
                checked.add(number)
        print(f'{index + 1}/{len(plan["commands"])} {command["label"]} {command["runtime"]} '
              f'{command["args"][0]}; {len(checked)} comparisons passed', flush=True)
    assert len(checked) == len(plan['comparisons'])
    for pair in plan['comparisons']:
        verify(pair)
    for path, expected in plan['fixtures'].items():
        assert sha(path) == expected, path
    # Preserve every checked artifact identity for later publication review.
    artifacts = {path: sha(path) for pair in plan['comparisons'] for path in (pair['left'], pair['right'])}
    write(out / 'result.json', dict(passed=True, native_commands=len(plan['commands']),
                                   exact_comparisons=len(checked), protocol_sha256=sha(out / 'protocol.json'),
                                   artifacts_sha256=artifacts, controlled_speed_comparison=False,
                                   synthetic_models_only=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('host-test', 'run'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--old-exe', type=Path, required=True)
    parser.add_argument('--new-exe', type=Path, default=ROOT / 'build/synapticgenesis.exe')
    args = parser.parse_args()
    (host_test if args.command == 'host-test' else run)(args.out.resolve(), args.old_exe, args.new_exe)
