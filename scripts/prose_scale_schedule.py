"""Declare complete book exposure rather than a fixed tiny number of updates."""
import argparse
import json
import os
from pathlib import Path

from corpus.selection import require_training_spec
from native_experiment import sha, verified_book_manifest


def prepare(prepared, out):
    spec = Path('data/sources-prose-scale-v1.json')
    require_training_spec(spec)
    manifest = verified_book_manifest(prepared, spec)
    out.mkdir(parents=True, exist_ok=False)
    chunk, cumulative = 128, 0
    epochs = [4, 1, 1, 1]
    lines, rows = ['SGCURRICULUM3'], []
    for stage, passes in zip(manifest['stages'], epochs):
        documents = (prepared / stage['file']).read_bytes().split(b'\x1e')
        pairs = sum(len(d) - 1 for d in documents)
        observations = sum((len(d) - 1 + chunk - 1) // chunk for d in documents)
        cumulative += passes * observations
        source = prepared / stage['cumulative_file']
        path = Path(os.path.relpath(source.resolve(), out.resolve())).as_posix()
        lines.append(f'{cumulative} "{path}" 1 new 1')
        rows.append(dict(stage=stage['id'], passes=passes, documents=len(documents),
                         source_pairs_per_pass=pairs, observations_per_pass=observations,
                         end_update=cumulative, cumulative_source=str(source), source_sha256=sha(source)))
    schedule = out / 'curriculum.sg'
    schedule.write_bytes(('\n'.join(lines) + '\n').encode())
    result = dict(source_spec_sha256=sha(spec), prepared_manifest_sha256=sha(prepared / 'manifest.json'),
                  chunk=chunk, stages=rows, schedule_sha256=sha(schedule),
                  source_pairs=sum(r['passes'] * r['source_pairs_per_pass'] for r in rows),
                  online_updates=cumulative, policy='Complete four foundation passes and one pass of each '
                  'later stage, with new-document source traversal and ordinary balanced stage replay. '
                  'This is an initial exposure schedule, not a demonstrated optimum or stopping rule.')
    (out / 'protocol.json').write_bytes((json.dumps(result, indent=2) + '\n').encode())
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepared', type=Path, default=Path('data/prepared/prose-scale-v1-pinned'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.prepared, args.out)
