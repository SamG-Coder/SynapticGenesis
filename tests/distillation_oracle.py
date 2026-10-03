"""Run every teacher-loss fixture and preserve both passes and strict failures."""
import argparse
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import torch

from oracle import check


def run(root):
    names = [f'cell-{cell}' for cell in range(1, 7)] + ['cell-6-warm-adam']
    records = []
    torch.set_num_threads(1)
    for name in names:
        with redirect_stdout(io.StringIO()):
            try:
                result = check(root / name)
            except AssertionError as error:
                result = (error.args[0] if error.args and isinstance(error.args[0], dict)
                          and 'reference' in error.args[0] else dict(passed=False, error=str(error)))
        records.append(dict(fixture=name, **result))
        print(name, {k: result[k] for k in ('passed', 'gradients_max_abs_error',
                                          'adam_update_max_abs_error') if k in result}, flush=True)
    result = dict(passed=all(r['passed'] for r in records), records=records,
                  native_controls=json.loads((root / 'result.json').read_text()),
                  limits='Synthetic mathematical checks, not retained language skills or live parent teaching. '
                         'Strict failures are retained and cause an unsuccessful exit.')
    (root / 'independent-oracle.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('build/distillation-test-results'))
    result = run(parser.parse_args().root)
    raise SystemExit(0 if result['passed'] else 1)
