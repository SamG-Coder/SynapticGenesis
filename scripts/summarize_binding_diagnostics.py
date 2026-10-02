"""Collect the completed optimization controls without changing model files."""
import json
from pathlib import Path

from audit_binding import audit


def collect():
    tiny = json.loads(Path('runs/binding-tiny-diagnostic/diagnostic.json').read_text())
    for row in tiny['runs']:
        row['probes'].pop('results')
    rates = json.loads(Path('runs/binding-rate-panel/comparison.json').read_text())
    for row in rates['runs']:
        folder = Path(f"runs/binding-rate-panel/{row['cell']}-{row['base_rate']:g}")
        row['response_audit'] = {split:audit(folder/f'{split}.json', split) for split in ('train','development')}
    report = dict(single_group=tiny, full_collection_rates=rates,
                  baseline_response_audits={cell:audit(Path(f'runs/binding-{cell}-1337/development.json'))
                                            for cell in ('lif','trace')},
                  conclusion='Both cells fit one complete group at all tested rates, but increasing later-stage '
                             'plasticity does not solve the complete binding collection under the tested exposure. '
                             'Changing questions/facts rarely changes baseline answers. This motivates testing '
                             'selective retrieval; it does not prove that architecture alone is responsible.')
    Path('reports/binding-optimization.json').write_text(json.dumps(report,indent=2)+'\n')
    for row in rates['runs']:
        print(row['cell'],row['actual_binding_rate'], row['train']['joint_accuracy'],
              row['development']['joint_accuracy'], row['session']['final_validation_loss'])


if __name__ == '__main__':
    collect()
