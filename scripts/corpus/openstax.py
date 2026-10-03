"""Text extraction for the reviewed Astronomy 2e CNXML edition.

MathML is serialized explicitly as LaTeX, never flattened into plain digits.
Unsupported mathematical structures fail closed; this is not a general-purpose
converter for every OpenStax book.
"""
from collections import Counter
import re
import xml.etree.ElementTree as ET


CN = '{http://cnx.rice.edu/cnxml}'
MATH = '{http://www.w3.org/1998/Math/MathML}'
OMITTED_NOTES = {'link-to-learning', 'making-connections', 'voyagers-in-astronomy', 'seeing-for-yourself'}


def compact(text):
    return re.sub(r'\s+', ' ', text).strip()


def tex_text(text):
    return ''.join({'\\': r'\textbackslash{}', '{': r'\{', '}': r'\}',
                    '$': r'\$', '%': r'\%', '&': r'\&', '#': r'\#',
                    '_': r'\_', '^': r'\textasciicircum{}'}.get(c, c) for c in text)


def math_text(node):
    if not node.tag.startswith(MATH):
        raise ValueError('Non-MathML element inside a formula')
    name = node.tag[len(MATH):]
    children = list(node)
    if name in ('mi', 'mn', 'mo', 'mtext'):
        if children:
            raise ValueError('Unexpected children in a MathML token')
        text = compact(node.text or '')
        return (r'\text{' + tex_text(text) + '}') if name == 'mtext' else tex_text(text)
    if name == 'mspace' and not children:
        return ' '
    if name in ('math', 'mrow', 'mtd'):
        if compact(node.text or '') or any(compact(c.tail or '') for c in children):
            raise ValueError('Unexpected text outside MathML tokens')
        return ' '.join(math_text(c) for c in children).strip()
    arities = {'mfrac': 2, 'msup': 2, 'msub': 2, 'msubsup': 3}
    if name in arities:
        if len(children) != arities[name]:
            raise ValueError(f'Invalid MathML arity for {name}')
        values = [math_text(c) for c in children]
        if name == 'mfrac':
            return r'\frac{' + values[0] + '}{' + values[1] + '}'
        suffix = {'msup': '^', 'msub': '_', 'msubsup': '_'}[name]
        result = '{' + values[0] + '}' + suffix + '{' + values[1] + '}'
        return result + ('^{' + values[2] + '}' if name == 'msubsup' else '')
    if name == 'msqrt' and children:
        return r'\sqrt{' + ' '.join(math_text(c) for c in children) + '}'
    if name == 'mtr' and all(c.tag == MATH + 'mtd' for c in children):
        return ' & '.join(math_text(c) for c in children)
    if name == 'mtable' and children and all(c.tag == MATH + 'mtr' for c in children):
        return r'\begin{matrix} ' + r' \\ '.join(math_text(c) for c in children) + r' \end{matrix}'
    raise ValueError(f'Unsupported MathML structure: {name}')


def extract(raw, module_titles=None):
    if b'<!DOCTYPE' in raw or b'<!ENTITY' in raw:
        raise ValueError('External/custom XML entities are unsupported')
    root = ET.fromstring(raw)
    if root.tag != CN + 'document':
        raise ValueError('Expected a CNXML document')
    title = root.find(CN + 'title')
    content = root.find(CN + 'content')
    if title is None or content is None:
        raise ValueError('Missing module title or content')
    module_titles = module_titles or {}
    labels, counts, stats = {}, Counter(), Counter()
    for node in content.iter():
        tag = node.tag.removeprefix(CN)
        if tag in ('figure', 'table', 'equation', 'example'):
            counts[tag] += 1
            if node.get('id'):
                labels[node.get('id')] = f'{tag.title()} {counts[tag]}'
        elif node.get('id'):
            heading = node.find(CN + 'title')
            if heading is not None:
                labels[node.get('id')] = compact(''.join(heading.itertext()))

    def contents(node):
        return (node.text or '') + ''.join(render(c) + (c.tail or '') for c in node)

    def block(value):
        return '\n\n' + value.strip() + '\n\n' if value.strip() else ''

    def table_group(node):
        width = int(node.get('cols', '0'))
        if not 1 <= width <= 64:
            raise ValueError('Invalid table width')
        names = {f'c{i}': i - 1 for i in range(1, width + 1)}
        for c in node.findall(CN + 'colspec'):
            names[c.get('colname')] = int(c.get('colnum')) - 1
        rows = [r for group in node if group.tag in (CN + 'thead', CN + 'tbody', CN + 'tfoot')
                for r in group if r.tag == CN + 'row']
        active, output = {}, []
        for index, row in enumerate(rows):
            cells = [active[i][1] if i in active and active[i][0] >= index else None for i in range(width)]
            cursor = 0
            for entry in row:
                if entry.tag != CN + 'entry':
                    raise ValueError('Unexpected table row content')
                while cursor < width and cells[cursor] is not None:
                    cursor += 1
                first, last = entry.get('namest', entry.get('colname')), entry.get('nameend')
                if any(name and name not in names for name in (first, last)):
                    raise ValueError('Undeclared table column in span')
                start = names[first] if first else cursor
                end = names[last] if last else start
                more = int(entry.get('morerows', '0'))
                if (not 0 <= start <= end < width or more < 0 or index + more >= len(rows)
                        or any(cells[i] is not None for i in range(start, end + 1))):
                    raise ValueError('Invalid or overlapping table span')
                value = compact(contents(entry))
                for column in range(start, end + 1):
                    cells[column] = value
                    active[column] = (index + more, value)
                if more or end != start:
                    stats['table_span_cells'] += 1
                cursor = end + 1
            # A full-width table heading remains a heading. Other merged cells
            # repeat into covered columns/rows to keep numeric labels aligned.
            if len(row) == 1 and len(set(cells)) == 1 and cells[0] is not None:
                output.append(cells[0])
            else:
                output.append(' | '.join(v or '' for v in cells))
        return block('\n'.join(output))

    def render(node):
        if node.tag == MATH + 'math':
            stats['math_expressions'] += 1
            return r'\(' + math_text(node) + r'\)'
        if not node.tag.startswith(CN):
            raise ValueError(f'Unsupported content namespace: {node.tag}')
        tag = node.tag[len(CN):]
        if tag == 'note' and OMITTED_NOTES.intersection(node.get('class', '').split()):
            stats['omitted_sidebars'] += 1
            return ''
        if tag in ('metadata', 'label', 'image', 'iframe', 'colspec'):
            stats['omitted_' + tag] += 1
            return ''
        if tag == 'media':
            alt = compact(node.get('alt', ''))
            if alt:
                stats['image_descriptions'] += 1
                return block('Image description: ' + alt)
            return ''
        if tag == 'link':
            label = compact(contents(node))
            if label:
                return label
            target = node.get('target-id', '')
            document = node.get('document', '')
            if document and document in module_titles:
                return module_titles[document]
            if target in labels:
                return labels[target]
            stats['generic_cross_references'] += 1
            return 'the referenced section'
        if tag in ('sup', 'sub'):
            return ('^' if tag == 'sup' else '_') + '(' + compact(contents(node)) + ')'
        if tag == 'newline':
            return '\n'
        if tag == 'footnote':
            return ' [Footnote: ' + compact(contents(node)) + '] '
        if tag == 'tgroup':
            return table_group(node)
        if tag == 'list':
            items = list(node)
            if any(c.tag != CN + 'item' for c in items):
                raise ValueError('Unexpected list child')
            enumeration = node.get('list-type') == 'enumerated'
            start = int(node.get('start-value', '1'))
            style = node.get('number-style', 'arabic')
            if enumeration and style not in ('arabic', 'lower-alpha', 'upper-alpha'):
                raise ValueError('Unsupported list numbering style')
            result = []
            for i, child in enumerate(items, start):
                prefix = '- '
                if enumeration:
                    prefix = (str(i) if style == 'arabic' else chr((97 if style == 'lower-alpha' else 65) + i - 1)) + '. '
                result.append(prefix + contents(child).strip())
            return block('\n'.join(result))
        if tag in ('problem', 'solution'):
            return block(('Question: ' if tag == 'problem' else 'Answer: ') + contents(node).strip())
        if tag in ('para', 'title', 'caption', 'meaning', 'quote', 'commentary'):
            return block(compact(contents(node)))
        if tag in ('figure', 'table', 'equation', 'example'):
            return block(labels.get(node.get('id'), tag.title()) + '\n' + contents(node).strip())
        if tag == 'definition':
            return block(contents(node))
        if tag in ('emphasis', 'span', 'term'):
            return contents(node)
        if tag in ('content', 'section', 'note', 'exercise', 'glossary'):
            return contents(node)
        raise ValueError(f'Unsupported CNXML element: {tag}')

    text = compact(''.join(title.itertext())) + '\n\n' + render(content)
    glossary = root.find(CN + 'glossary')
    if glossary is not None:
        text += '\n\nGlossary\n\n' + render(glossary)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r' *\n *', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text).strip() + '\n'
    if any(ord(c) < 32 and c not in '\n\t' for c in text) or '\ufffd' in text or len(text) < 60:
        raise ValueError('Invalid or empty extracted text')
    return text.encode('utf-8'), dict(stats)
