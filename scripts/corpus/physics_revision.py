"""Bind the reviewed follow-up errata to an explicitly admitted Physics revision."""
from copy import deepcopy
from pathlib import Path

from corpus.physics_edits import digest, require
from native_experiment import read, sha
from review_physics_errata import compiled as reviewed_errata


ERRATA = Path('data/physics-followup-errata-v1.json')
REVIEW = Path('reports/physics-followup-errata-review.json')
CHECKS = Path('reports/physics-followup-errata-checks.json')
NOTICE = (' Follow-up edition selected-physics-v2 applies seven separately reviewed text corrections '
          'in three training modules: the conserved-energy subtraction and its numerical substitution, '
          'total thermal energy versus a per-particle average, quantum ground-state motion at absolute '
          'zero, and the mass dependence of heat capacity. The exact before/after passages, original '
          'XML identities, research references and independent arithmetic checks accompany this '
          'revision. The original source files and all validation/test modules remain unchanged. '
          'This targeted errata set does not certify the rest of the textbook.')

REVISION_PROVENANCE = {
    'base_source_spec': 'base-source-spec.json',
    'base_preparation_audit': 'base-preparation-audit.json',
    'base_manifest': 'base-manifest.json',
    'followup_errata': 'followup-errata.json',
    'followup_review': 'followup-review.json',
    'followup_checks': 'followup-checks.json',
}


def reviewed_inputs():
    plan, base, changes, inputs = reviewed_errata(ERRATA)
    review, checks = read(REVIEW), read(CHECKS)
    require(review['complete'] and review['status'] == 'review_copy_not_admitted' and
            review['source_edition'] == base.spec['version'] == 'selected-physics-v1' and
            review['specification_sha256'] == sha(ERRATA) and
            (review['corrections'], review['changed_modules']) == (7, 3),
            'Follow-up review identity differs')
    require(checks['passed'] and checks['proposed_review_result_sha256'] == sha(REVIEW) and
            (checks['correction_count'], checks['modules_checked']) == (7, 3) and
            checks['full_originals_recovered'] and checks['question_and_solution_markers_preserved'] and
            checks['algebra']['proposed_equation_matches_conservation'], 'Follow-up audit differs')
    for change in changes:
        name = f"review-text/{change['id']}.txt"
        require(digest(change['corrected']) == review['output_sha256'][name],
                'Reconstructed correction differs from the reviewed copy')
    return plan, base, changes, [*inputs, REVIEW, CHECKS]


def selected_spec(plan, base, changes):
    """Derive the fixed revision metadata; old source records remain intact."""
    result = deepcopy(base.spec)
    result['version'] = 'selected-physics-v2'
    result['attribution'] += NOTICE
    result['limitations'].append('Seven follow-up text corrections affect three training modules; '
        'all prior selection limitations still apply. This is a revised edition, not additional independent text.')
    references = dict(base_source_spec=Path(plan['base_source_spec']),
        base_preparation_audit=Path(plan['base_preparation_audit']),
        base_manifest=base.prepared / 'manifest.json', followup_errata=ERRATA,
        followup_review=REVIEW, followup_checks=CHECKS)
    for key, path in references.items():
        result[key], result[key + '_sha256'] = path.as_posix(), sha(path)
    amended = {r['id']: r for r in changes}
    for row in result['sources']:
        if row['id'] in amended:
            change = amended[row['id']]
            require(row['split'] == 'train', 'A correction would rewrite a protected module')
            row['corrected_sha256'] = digest(change['corrected'])
            row['correction_ids'].extend(change['corrections'])
    return result


def compiled_revision(spec, cache, download=False):
    # Import here to reuse the v1 source reconstruction without a module cycle.
    from prepare_physics import compiled_selection
    plan, base, changes, _ = reviewed_inputs()
    require(spec == selected_spec(plan, base, changes), 'Physics revision differs from reviewed changes')
    bodies, auxiliary, edits, passage = compiled_selection(base.spec, cache, download)
    for row in base.manifest['sources']:
        require(digest(bodies[row['id']]) == row['clean_sha256'],
                'Original CNXML reconstruction differs from the prepared base')
    for change in changes:
        require(bodies[change['id']] == change['original'], 'Errata applies to another base module')
        bodies[change['id']] = change['corrected']
    require(sum(len(r['correction_ids']) for r in spec['sources']) == 12,
            'Existing or follow-up corrections were lost')
    return bodies, auxiliary, edits, passage
