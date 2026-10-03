"""Pure rendering of selected fact/query grids; no model or inferred labels."""
import itertools


def render(objects, locations, facts, queries, answer_template, partitions):
    prerequisites, training = set(), set()
    rows = {name: [] for name in partitions}
    for obj, loc, fact, query in itertools.product(objects, locations, facts, queries):
        prerequisites.add(fact.format(object=obj, location=loc) + '\n' + query.format(object=obj) +
                          answer_template.format(location=loc) + '\n')
    for a, b in itertools.combinations(objects, 2):
        partition = next(name for name, pairs in partitions.items() if frozenset((a, b)) in pairs)
        for l0, l1 in itertools.combinations(locations, 2):
            for order, style, query_form in itertools.product(range(2), range(len(facts)), range(len(queries))):
                group = f'{a}-{b}-{l0}-{l1}-o{order}-s{style}-q{query_form}'
                for reverse, target in itertools.product(range(2), range(2)):
                    mapping = {a: (l0, l1)[reverse], b: (l0, l1)[1-reverse]}
                    named = (a, b) if order == 0 else (b, a)
                    context = ' '.join(facts[style].format(object=obj, location=mapping[obj])
                                       for obj in named) + '\n'
                    queried = (a, b)[target]
                    query = queries[query_form].format(object=queried)
                    row = dict(id=f'{group}-r{reverse}-t{target}', pair=group, skill=f'binding-style-{style}',
                               context=context, query=query, correct=int(mapping[queried] == l1),
                               choice0=answer_template.format(location=l0), choice1=answer_template.format(location=l1))
                    rows[partition].append(row)
                    if partition == 'train':
                        training.add(context + query + answer_template.format(location=mapping[queried]) + '\n')
    # Search inside every generated document, including prerequisites. Checking
    # only equality of two-fact contexts would miss a context embedded in a
    # changed template. Separators cannot hide an in-document substring.
    generated_text = '\x1e'.join(prerequisites | training)
    heldout_contexts = {row['context'] for name in ('development', 'test') for row in rows[name]}
    if any(context in generated_text for context in heldout_contexts):
        raise ValueError('Held-out binding context overlaps training')
    return prerequisites, training, rows, partitions
