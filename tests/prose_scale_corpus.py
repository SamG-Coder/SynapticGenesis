"""Audit pinned prose, held-out protection and complete-exposure scheduling."""
import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from corpus.selection import require_training_spec
from native_experiment import read, sha, verified_book_manifest
from prepare_corpus import clean


def audit(prepared, schedule, output):
    spec_path = Path('data/sources-prose-scale-v1.json')
    spec = require_training_spec(spec_path)
    manifest = verified_book_manifest(prepared, spec_path)
    raw = Path('data/raw/gutenberg-scale-v1')
    reviews = read('reports/prose-scale-source-review.json')
    reviewed = {r['id']: r for r in reviews['records']}
    counts = {}
    protected = set()
    for record in spec['sources']:
        ident = record['id']
        assert sha(raw / f'{ident}.txt') == record['reviewed_raw_sha256']
        assert 'Public domain in the USA' in (raw / f'{ident}.html').read_text(encoding='utf-8')
        if ident in reviewed:
            review = reviewed[ident]
            assert review['admitted'] and review['raw_sha256'] == record['reviewed_raw_sha256']
            normalized = ' '.join(clean((raw / f'{ident}.txt').read_bytes(), record).split())
            assert all(sample in normalized for sample in review['reviewed_samples'])
        if record['split'] != 'train':
            body = (prepared / f'{ident}.txt').read_text(encoding='utf-8')
            protected.update(re.sub(r'\s+', ' ', p).strip() for p in re.split(r'\n\s*\n', body)
                             if len(re.sub(r'\s+', ' ', p).strip()) >= 120)
    for record in spec['sources']:
        if record['split'] == 'train':
            body = (prepared / f"{record['id']}.txt").read_text(encoding='utf-8')
            paragraphs = {re.sub(r'\s+', ' ', p).strip() for p in re.split(r'\n\s*\n', body)}
            assert not paragraphs & protected
    for name in ('train', 'validation', 'test'):
        text = (prepared / f'{name}.dat').read_text(encoding='utf-8')
        counts[name] = dict(**manifest['outputs'][name],
                           word_like_units=len(re.findall(r"[A-Za-z]+(?:['-][A-Za-z]+)*", text)))
    for ident in (13853, 15659, 12228, 11757, 902):
        assert (prepared / f'{ident}.txt').read_bytes() == (Path('data/prepared/stories-v1') / f'{ident}.txt').read_bytes()
    plan = read(schedule / 'protocol.json')
    assert sha(schedule / 'curriculum.sg') == plan['schedule_sha256']
    assert sha(prepared / 'manifest.json') == plan['prepared_manifest_sha256']
    end = 0
    for stage, row in zip(manifest['stages'], plan['stages']):
        docs = (prepared / stage['file']).read_bytes().split(b'\x1e')
        lengths = [len(d) - 1 for d in docs]
        windows = sum(len(range(0, length, plan['chunk'])) for length in lengths)
        assert row['source_pairs_per_pass'] == sum(lengths) and row['observations_per_pass'] == windows
        end += row['passes'] * windows
        assert row['end_update'] == end and sha(row['cumulative_source']) == row['source_sha256']
    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory)
        first = next(r for r in spec['sources'] if r['split'] == 'test')['id']
        shutil.copyfile(raw / f'{first}.html', temporary / f'{first}.html')
        (temporary / f'{first}.txt').write_bytes((raw / f'{first}.txt').read_bytes() + b'changed')
        result = subprocess.run([sys.executable, 'scripts/prepare_corpus.py', '--sources', str(spec_path),
                                 '--raw', str(temporary), '--out', str(temporary / 'out')], capture_output=True)
        assert result.returncode != 0 and b'Reviewed source bytes changed' in result.stderr
    report = dict(passed=True, source_spec_sha256=sha(spec_path), source_manifest_sha256=sha(prepared / 'manifest.json'),
        review_sha256=sha('reports/prose-scale-source-review.json'), counts=counts,
        reviewed_new_books=sum(r['admitted'] for r in reviewed.values()),
        existing_holdouts_byte_identical=True, training_protected_paragraph_overlap=0,
        changed_reviewed_bytes_rejected=True, complete_exposure_schedule_verified=True,
        schedule=plan, limitation='Sampled passage review; exact paragraph protection does not detect paraphrases, '
        'short matches or shared plots. Historical prose is not modern factual ground truth.')
    output.write_bytes((json.dumps(report, indent=2) + '\n').encode())
    print(json.dumps(counts, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepared', type=Path, default=Path('data/prepared/prose-scale-v1-pinned'))
    parser.add_argument('--schedule', type=Path, default=Path('runs/prose-scale-curriculum'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    audit(args.prepared, args.schedule, args.out)
