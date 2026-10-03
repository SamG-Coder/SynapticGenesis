"""Conservative text-overlap rejection for explicitly selected reading editions."""
import re


def words(document):
    return re.findall(r'\w+', document.decode('utf-8').casefold())


def signatures(document):
    tokens = words(document)
    sequences = {tuple(tokens[i:i+20]) for i in range(len(tokens)-19)}
    paragraphs = {value for p in re.split(rb'\n\s*\n', document)
                  if len(value := ' '.join(words(p))) >= 120}
    return tuple(tokens), sequences, paragraphs


def reject_overlap(documents, protected):
    """Reject new/new and new/old copies; report neither paraphrase detection nor scoring."""
    whole, sequences, paragraphs = set(), set(), set()
    for document in protected.values():
        full, grams, blocks = signatures(document)
        whole.add(full)
        sequences.update(grams)
        paragraphs.update(blocks)
    for identifier, document in documents.items():
        full, grams, blocks = signatures(document)
        if not full or full in whole or grams & sequences or blocks & paragraphs:
            raise ValueError('Selected source overlaps another edition: '+identifier)
        whole.add(full)
        sequences.update(grams)
        paragraphs.update(blocks)
