"""Trace the first seed's largest full-audit score discrepancy, without learning."""
import argparse
from pathlib import Path
import torch

from binding_learned_oracle import development_rows
from probes_cli import Reference
from trace_comparison import compare
from native_experiment import NativeCommands, read, sha, write
from prepare_lessons import write_probes


def check(study, history, out):
    data = read(study / 'comparison.json')
    fixtures = development_rows(data['protocol'])
    cpu = read(history / '1337-normal.json')['results']
    original = read(study / '1337-associative/development-130000.json')['results']
    checkpoint = study / '1337-associative/checkpoint-130000.ckpt'
    identity = sha(checkpoint)
    assert identity == next(row['checkpoint_sha256'] for row in data['runs'] if
                            row['seed'] == 1337 and row['arm'] == 'associative' and row['online_updates'] == 130000)
    assert len(cpu) == len(original) == len(fixtures) == 576
    differences = []
    for index, (reference, native, fixture) in enumerate(zip(cpu, original, fixtures)):
        assert reference['id'] == native['id'] == fixture['id']
        for field in ('candidate_nll', 'context_erased_nll'):
            for choice in range(2):
                differences.append((abs(reference[field][choice] - native[field][choice]), index, field, choice))
    error, index, field, choice = max(differences)
    assert error > 3e-5
    selected = fixtures[index]
    group_indices = [i for i, row in enumerate(fixtures) if row['pair'] == selected['pair']]
    assert len(group_indices) == 4
    exe, diagnostic = Path('build/synapticgenesis.exe'), Path('build/trace-diagnostic/synaptic-trace-diagnostic.exe')
    assert sha(exe) == data['protocol']['executable_sha256']
    sources = ['tests/associative_threshold.py', 'tests/trace_comparison.py', 'tests/learned_trace_dump.cu',
               'tests/probes_cli.py', 'tests/associative_reference.py']
    protocol = dict(selection='Largest absolute native/CPU score difference among all first-seed normal '
                    'development scores, selected after observing the failed strict audit.',
                    seed=1337, id=selected['id'], field=field, largest_error_choice=choice,
                    largest_error=error, checkpoint_sha256=identity,
                    source_sha256={name: sha(name) for name in sources},
                    study_executable_sha256=sha(exe), diagnostic_executable_sha256=sha(diagnostic),
                    normal_result_sha256=sha(history / '1337-normal.json'),
                    limits='Post-hoc local diagnosis. Original strict failures remain failures; '
                           'forcing a spike is not an independent passing oracle.')
    out.mkdir(parents=True, exist_ok=False)
    write(out / 'protocol.json', protocol)
    write_probes(out / 'group.sgprobe', [fixtures[i] for i in group_indices], 'SGPROBE2')
    native = NativeCommands(exe, out)
    native('language-probes', '--checkpoint', checkpoint, '--probes', out / 'group.sgprobe',
           '--output', out / 'native-repeat.json')
    repeated = read(out / 'native-repeat.json')['results']
    assert repeated == [original[i] for i in group_indices]
    native.exe = diagnostic.resolve()
    ref, records = Reference(checkpoint), []
    for candidate in range(2):
        prompt = (selected['context'] if field == 'candidate_nll' else '') + selected['query']
        raw = (prompt + selected[f'choice{candidate}']).encode('ascii')
        source, directory = out / f'choice-{candidate}.txt', out / f'choice-{candidate}'
        source.write_bytes(raw)
        native(checkpoint, source, directory)
        with torch.inference_mode():
            record = compare(ref, raw, directory)
        assert abs(record['native_nll'] - original[index][field][candidate]) < 1e-9
        assert abs(record['cpu_nll'] - cpu[index][field][candidate]) < 1e-9
        records.append(dict(choice=candidate, **record))
    assert sha(checkpoint) == identity and sha(exe) == protocol['study_executable_sha256']
    assert all(sha(name) == expected for name, expected in protocol['source_sha256'].items())
    localized = all(r.get('first_difference_within_1e_minus_5_of_threshold') and
                    r.get('single_spike_override_matches_all_native_spikes') and
                    r.get('single_spike_override_nll_error', 1) < 3e-5 for r in records)
    result = dict(protocol=protocol, records=records, native_repeat_results_identical=True,
                  checkpoint_file_unchanged=True, study_executable_unchanged=True,
                  single_near_threshold_spike_localizes_both_scores=localized,
                  original_strict_cpu_score_check_passed=False)
    write(out / 'result.json', result)
    print(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--study', type=Path, default=Path('runs/associative-long-panel'))
    parser.add_argument('--history', type=Path, default=Path('runs/associative-history-panel'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    check(args.study, args.history, args.out)
