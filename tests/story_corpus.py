"""Reproduce selected books and audit boundaries without running any model."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
from native_experiment import read, sha, write
from prepare_binding import examples
from prepare_corpus import clean


def paragraphs(text):
    return {normal for p in re.split(r'\n\s*\n', text)
            if len(normal := re.sub(r'\s+', ' ', p).strip()) >= 120}


def audit(out, prepared):
    out.mkdir(parents=True, exist_ok=False)
    for name, spec in [('stories', 'data/sources-stories-v1.json'),
                       ('legacy', 'data/sources-development-v2.json')]:
        command = [sys.executable, 'scripts/prepare_corpus.py', '--sources', spec, '--out', str(out / name)]
        result = subprocess.run(command, capture_output=True)
        (out / f'{name}.log').write_bytes(result.stdout + result.stderr)
        assert result.returncode == 0, name
    original = Path('data/prepared/development-v2-final')
    compared = {}
    for name, expected in [('stories', prepared), ('legacy', original)]:
        files = sorted(p for p in expected.iterdir() if p.suffix in ('.txt', '.dat', '.sg'))
        assert files
        for file in files:
            assert file.read_bytes() == (out / name / file.name).read_bytes(), (name, file.name)
        compared[name] = len(files)
    manifest = read(prepared / 'manifest.json')
    assert sha('data/sources-stories-v1.json') == manifest['source_spec_sha256']
    assert (prepared / 'source-spec.json').read_bytes() == Path('data/sources-stories-v1.json').read_bytes()
    assert len(manifest['sources']) == 12
    groups = {split: [r for r in manifest['sources'] if r['split'] == split]
              for split in ('train', 'validation', 'test')}
    assert {r['id'] for r in groups['train']} == {572, 5312, 43936, 11, 236, 17314, 17396}
    assert {r['id'] for r in groups['validation']} == {13853, 12228, 11757}
    assert {r['id'] for r in groups['test']} == {15659, 902}
    sets = {split: set() for split in groups}
    for split, records in groups.items():
        docs = []
        for source in records:
            file = prepared / f'{source["id"]}.txt'
            assert sha(file) == source['clean_sha256']
            assert sha(f'data/raw/{source["id"]}.txt') == source['raw_sha256']
            assert sha(f'data/raw/{source["id"]}.html') == source['catalog_sha256']
            text = file.read_text(encoding='utf-8')
            assert not any(marker in text for marker in
                           ('\ufffd', '\x1e', 'Project Gutenberg', '[Picture:', '[Illustration', "Transcriber's"))
            sets[split].update(paragraphs(text))
            docs.append(file.read_bytes())
        payload = b'\x1e'.join(docs)
        assert payload == (prepared / f'{split}.dat').read_bytes()
        assert sha(prepared / f'{split}.dat') == manifest['outputs'][split]['sha256']
    assert not sets['train'] & (sets['validation'] | sets['test'])
    for book in (13853, 12228, 15659):
        assert (prepared / f'{book}.txt').read_bytes() == (original / f'{book}.txt').read_bytes()
    cumulative = []
    for stage in manifest['stages']:
        docs = [(prepared / f'{r["id"]}.txt').read_bytes() for r in groups['train'] if r['stage'] == stage['id']]
        assert b'\x1e'.join(docs) == (prepared / stage['file']).read_bytes()
        cumulative.extend(docs)
        assert b'\x1e'.join(cumulative) == (prepared / stage['cumulative_file']).read_bytes()
    _, _, probes, _ = examples(read('data/lessons-binding-v2.json'))
    heldout = {re.sub(r'\s+', ' ', r['context']).strip()
               for split in ('development', 'test') for r in probes[split]}
    normalized = re.sub(r'\s+', ' ', (prepared / 'train.dat').read_text(encoding='utf-8'))
    assert not any(context in normalized for context in heldout)
    injected = normalized + '\n' + sorted(heldout)[0]
    assert any(context in injected for context in heldout), 'Contamination control failed'

    # Guard the actual extraction path against wrong editions and truncated bodies.
    selected = next(r for r in manifest['sources'] if r['id'] == 5312)
    raw = Path('data/raw/5312.txt').read_bytes()
    rejected = []
    for label, candidate, source in [
        ('missing-wrapper', raw.replace(b'*** START OF', b'BROKEN START OF', 1), selected),
        ('wrong-body-boundary', raw, {**selected, 'body_start': '^Absent source boundary$'}),
        ('truncated-body', b'*** START OF THE PROJECT GUTENBERG EBOOK X ***\nshort\n'
                           b'*** END OF THE PROJECT GUTENBERG EBOOK X ***', {'id': 1})]:
        try:
            clean(candidate, source)
        except ValueError:
            rejected.append(label)
        else:
            raise AssertionError(label)
    evidence = dict(passed=True, model_training_performed=False, reserved_test_scored=False,
                    source_count=12, new_training_books=7, new_evaluation_books=2,
                    retained_evaluation_books=3, reproduced_files=compared,
                    outputs=manifest['outputs'], training_words=sum(r['approx_words'] for r in groups['train']),
                    retained_evaluation_bytes_identical=True, heldout_paragraph_overlap=0,
                    heldout_binding_contexts_checked=len(heldout), heldout_binding_contexts_in_training=0,
                    contamination_control_detected=True, rejected_invalid_editions=rejected,
                    cumulative_prefixes_verified=True, source_spec_sha256=manifest['source_spec_sha256'],
                    preparation_script_sha256=sha('scripts/prepare_corpus.py'),
                    verification_script_sha256=sha(Path(__file__)),
                    limitation='Exact normalized paragraphs of at least 120 characters and literal binding '
                               'contexts only; paraphrases, shared tales and shorter overlaps can remain.')
    write(out / 'verification.json', evidence)
    print(json.dumps(evidence, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--prepared', type=Path, default=Path('data/prepared/stories-v1'))
    args = parser.parse_args()
    audit(args.out, args.prepared)
