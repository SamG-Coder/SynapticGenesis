"""Localize the largest score error in each failed fixed-group CPU check.

Selection is post-hoc. The original strict failure remains a failure; a forced
spike diagnoses a numerical difference rather than independently validating it.
"""
import argparse
from pathlib import Path
import torch

from binding_learned_oracle import development_rows
from probes_cli import Reference
from trace_comparison import compare
from native_experiment import NativeCommands, read, sha, write
from prepare_lessons import write_probes


def diagnose(study, out, exe, diagnostic):
    data, oracle = read(study / 'comparison.json'), read(study / 'learned-oracle.json')
    p = data['protocol']
    assert sha(exe) == p['executable_sha256']
    failures = oracle.get('failed_score_tolerance', [])
    assert failures and not oracle['passed']
    fixtures = development_rows(p)[:4]
    endpoint = p['online_endpoints'][-1]
    sources = ['tests/score_diagnosis.py', 'tests/trace_comparison.py', 'tests/learned_trace_dump.cu',
               'tests/probes_cli.py', 'tests/associative_reference.py']
    protocol = dict(selection='Largest absolute score error within the four fixed development questions, '
                    'separately for each model that failed the existing score tolerance. Post-hoc diagnosis.',
                    source_sha256={name: sha(name) for name in sources},
                    study_executable_sha256=sha(exe), diagnostic_executable_sha256=sha(diagnostic),
                    study_comparison_sha256=sha(study / 'comparison.json'),
                    failed_oracle_sha256=sha(study / 'learned-oracle.json'),
                    limits='Only selected failing scores are traced. No full development CPU audit. '
                           'A single-spike override is a diagnostic, not an independent oracle pass.')
    out.mkdir(parents=True, exist_ok=False)
    write(out / 'protocol.json', protocol)
    write_probes(out / 'group.sgprobe', fixtures, 'SGPROBE2')
    native, records = NativeCommands(exe, out), []
    for failure in failures:
        seed, arm = failure['seed'], failure['arm']
        name = f'{seed}-{arm}'
        directory = study / name
        checkpoint = directory / f'checkpoint-{endpoint}.ckpt'
        identity = sha(checkpoint)
        declared = next(row for row in data['runs'] if row['seed'] == seed and row['arm'] == arm
                        and row['online_updates'] == endpoint)
        assert identity == declared['checkpoint_sha256']
        original = read(directory / f'development-{endpoint}.json')['results'][:4]
        ref, differences = Reference(checkpoint), []
        for index, (fixture, actual) in enumerate(zip(fixtures, original)):
            assert fixture['id'] == actual['id']
            for field in ('candidate_nll', 'context_erased_nll'):
                prompt = (fixture['context'] if field == 'candidate_nll' else '') + fixture['query']
                for choice in range(2):
                    score = ref.score(prompt, fixture[f'choice{choice}'])
                    differences.append((abs(score - actual[field][choice]), index, field, choice, score))
        error, index, field, choice, score = max(differences)
        assert error == failure['oracle_max_score_error'] and error >= failure['tolerance']
        native.exe = exe.resolve()
        native('language-probes', '--checkpoint', checkpoint, '--probes', out / 'group.sgprobe',
               '--output', out / f'{name}-repeat.json')
        assert read(out / f'{name}-repeat.json')['results'] == original
        selected = fixtures[index]
        prompt = (selected['context'] if field == 'candidate_nll' else '') + selected['query']
        raw = (prompt + selected[f'choice{choice}']).encode('ascii')
        source, trace = out / f'{name}.txt', out / f'{name}-trace'
        source.write_bytes(raw)
        native.exe = diagnostic.resolve()
        native(checkpoint, source, trace)
        with torch.inference_mode():
            record = compare(ref, raw, trace)
        assert abs(record['native_nll'] - original[index][field][choice]) < 1e-9
        assert abs(record['cpu_nll'] - score) < 1e-9
        assert sha(checkpoint) == identity
        records.append(dict(seed=seed, arm=arm, id=selected['id'], field=field, choice=choice,
                            original_score_error=error, checkpoint_sha256=identity, **record))
    assert all(sha(name) == expected for name, expected in protocol['source_sha256'].items())
    assert sha(exe) == protocol['study_executable_sha256']
    assert sha(diagnostic) == protocol['diagnostic_executable_sha256']
    assert sha(study / 'learned-oracle.json') == protocol['failed_oracle_sha256']
    result = dict(protocol=protocol, records=records, native_commands=len(native.commands),
                  native_repeat_results_identical=True, checkpoint_files_unchanged=True,
                  original_strict_cpu_score_check_passed=False)
    write(out / 'result.json', result)
    print(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--exe', type=Path, default=Path('build/synapticgenesis.exe'))
    parser.add_argument('--diagnostic', type=Path,
                        default=Path('build/trace-diagnostic/synaptic-trace-diagnostic.exe'))
    args = parser.parse_args()
    diagnose(args.study, args.out, args.exe, args.diagnostic)
