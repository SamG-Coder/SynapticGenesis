"""Freeze the arithmetic/science development suite before any native assessment."""
import argparse
from pathlib import Path

from corpus.reviewed_edition import authenticate, require
from corpus.selection import SELECTION
from native_experiment import read, sha, verified_book_manifest
from prepare_lessons import write_probes
from prose_founder import write
from quantitative_probes import build, protect


ROOT = Path(__file__).resolve().parents[1]


def prepare(spec_path, out):
    require(not out.exists(), 'Use a fresh quantitative suite directory')
    spec = read(spec_path)
    rows, statements, groups = build(spec)
    arithmetic = authenticate('data/sources-prealgebra-v1.json',
        'reports/prealgebra-worked-selection.json', 'data/prepared/prealgebra-worked-v1-pinned', paired=True)
    physics = authenticate(ROOT / 'data/sources-physics-v2.json',
        ROOT / 'reports/physics-revision-checks.json', 'data/prepared/physics-v2-reviewed')
    prose = Path('data/prepared/prose-scale-v1-pinned')
    verified_book_manifest(prose, Path('data/sources-prose-scale-v1.json'))
    require(spec['source_files'] == [dict(label=label, path=(prepared / 'train.dat').as_posix(),
                                        sha256=sha(prepared / 'train.dat')) for label, prepared in
        (('prose', prose), ('arithmetic', arithmetic.prepared), ('physics', physics.prepared))],
        'Unexpected training streams in quantitative suite')
    identities = {p.as_posix(): sha(p) for p in [spec_path, SELECTION, *arithmetic.inputs, *physics.inputs,
        prose / 'manifest.json', prose / 'source-spec.json', Path('data/sources-prose-scale-v1.json'),
        *(ROOT / p for p in ('scripts/prepare_quantitative_probes.py', 'scripts/quantitative_probes.py',
            'scripts/prepare_lessons.py', 'scripts/corpus/reviewed_edition.py',
            'scripts/corpus/physics_revision.py', 'scripts/corpus/paired_selection.py',
            'scripts/corpus/selection.py', 'scripts/native_experiment.py', 'scripts/prose_founder.py',
            'src/language_probes.cuh'))]}
    for module in ('m54104', 'm54142', 'm54271', 'm54290'):
        selected = next(r for r in physics.spec['sources'] if r['id'] == module)
        require(selected['split'] == 'train', 'Quantitative concept is not in selected training modules')
    identities.update(protect(rows, statements, spec['source_files']))
    out.mkdir(parents=True)
    (out / 'source-spec.json').write_bytes(spec_path.read_bytes())
    write_probes(out / 'development.sgprobe', rows, version='SGPROBE2')
    write(out / 'questions.json', rows)
    write(out / 'arithmetic-witnesses.json', groups)
    write(out / 'protected-statements.json', statements)
    require(all(sha(p) == h for p, h in identities.items()), 'Input changed during probe preparation')
    files = {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()}
    result = dict(version=spec['version'], status='prepared_before_native_scoring',
        source_spec_sha256=sha(spec_path), native_format='SGPROBE2', groups=len(groups), items=len(rows),
        skills={skill['id']: 16 for skill in spec['skills']}, equal_length_choices=True,
        context_query_crossing=True, candidate_labels_balanced=True, training_admitted=False,
        authored_contexts_checked=len({r['context'] for r in rows}),
        authored_trial_statements_checked=len(statements),
        maximum_prompt_bytes=max(len((r['context'] + r['query']).encode('ascii')) for r in rows),
        protected_input=dict(path=(out / 'development.sgprobe').as_posix(),
                             sha256=files['development.sgprobe'], kind='probe'),
        authenticated_inputs=identities, prepared_files=files, native_commands=0,
        model_quality_verified=False, test_set_created=False, reproduction_admitted=False,
        limits=spec['policy']['limits'])
    write(out / 'manifest.json', result)
    print('Prepared', len(rows), 'development items in', len(groups), 'four-item groups; no model execution.')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', type=Path, default=ROOT / 'data/quantitative-probes-v1.json')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.spec, args.out)
