"""Publish audited reports while recording Windows runtime and Git LF identities."""
import argparse
import json
from pathlib import Path

from native_experiment import read, sha


def publish(root, plot, prefix):
    comparison = root / 'comparison.json'
    check = root / 'execution-check.json'
    summary = root / 'summary.json'
    plot_metadata = plot.with_name(plot.name + '.metadata.json')
    audit, measured, picture = read(check), read(summary), read(plot_metadata)
    identity = sha(comparison)
    if not audit['passed'] or audit['comparison_sha256'] != identity:
        raise ValueError('A matching successful execution audit is required')
    if (measured['comparison_sha256'] != identity or picture['comparison_sha256'] != identity
            or measured['execution_check_sha256'] != sha(check)
            or picture['execution_check_sha256'] != sha(check)
            or picture['image_sha256'] != sha(plot)):
        raise ValueError('Summary or plot provenance does not match the audited study')
    if plot.suffix.lower() != '.png':
        raise ValueError('The published comparison image must be PNG')
    pairs = [(comparison, Path(str(prefix) + '.json')),
             (check, Path(str(prefix) + '-execution.json')),
             (summary, Path(str(prefix) + '-summary.json')),
             (plot, Path(str(prefix) + '-comparison.png')),
             (plot_metadata, Path(str(prefix) + '-comparison.png.metadata.json'))]
    manifest = Path(str(prefix) + '-publication.json')
    if any(target.exists() for _, target in pairs) or manifest.exists():
        raise ValueError('Publication needs fresh output paths')
    entries = []
    for source, target in pairs:
        raw = source.read_bytes()
        output = raw.replace(b'\r\n', b'\n') if target.suffix == '.json' else raw
        if target.suffix == '.json' and json.loads(output) != json.loads(raw):
            raise ValueError('Newline normalization changed JSON data')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(output)
        entries.append(dict(runtime_path=source.resolve().as_posix(), runtime_sha256=sha(source),
                            published_path=target.as_posix(), published_sha256=sha(target),
                            newline_bytes_changed=(raw != output)))
    record = dict(status='Published matching audited artifacts',
                  study_status=measured['status'], trajectories=measured['trajectories'],
                  publisher_source_sha256=sha(__file__), artifacts=entries,
                  note='Published JSON uses LF to match repository attributes. Internal identities refer '
                       'to the untouched original runtime bytes. Both identities are recorded here; '
                       'JSON data is identical, and the PNG is copied byte for byte.')
    manifest.write_bytes((json.dumps(record, indent=2) + '\n').encode('utf-8'))
    print(manifest)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--plot', type=Path, required=True)
    parser.add_argument('--prefix', type=Path, required=True)
    args = parser.parse_args()
    publish(args.study, args.plot, args.prefix)
