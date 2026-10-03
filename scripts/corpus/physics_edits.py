"""Apply individually reviewed Physics source edits to a copy of pinned CNXML."""
from copy import deepcopy
import hashlib
import xml.etree.ElementTree as ET

from corpus.openstax import CN, MATH, compact
from corpus.physics import remove_preserving_tail


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def edit_document(raw, module, records):
    require(b'<!DOCTYPE' not in raw and b'<!ENTITY' not in raw, 'Custom XML entities are unsupported')
    root = ET.fromstring(raw)
    require(root.tag == CN + 'document', 'Expected a CNXML document')
    selected = [r for r in records if r['module'] == module]
    require(len({r['element_id'] for r in selected}) == len(selected), 'Duplicate edited element')
    # Authenticate every original subtree before applying any transformation.
    originals = []
    for record in selected:
        require(record['source_sha256'] == digest(raw), 'Edited module source identity changed')
        matches = [e for e in root.iter() if e.get('id') == record['element_id']]
        require(len(matches) == 1, 'Missing or duplicate edited element')
        node = matches[0]
        require(digest(ET.tostring(node, encoding='utf-8')) == record['element_sha256'],
                'Reviewed element changed')
        require(record['reason'].strip() and record['references'], 'Missing review rationale')
        originals.append((record, node))
    parents = {child: parent for parent in root.iter() for child in parent}
    for record, node in originals:
        op = record['operation']
        if op == 'table_columns':
            require(node.tag == CN + 'table', 'Column correction needs a table')
            groups = node.findall(CN + 'tgroup')
            require(len(groups) == 1 and groups[0].get('cols') == str(record['old_columns']),
                    'Reviewed table declaration changed')
            rows = groups[0].findall('.//' + CN + 'row')
            width = record['new_columns']
            require(1 <= width <= 64 and rows and all(len(r) == width for r in rows),
                    'Reviewed table rows no longer have a uniform width')
            require(all(e.tag == CN + 'entry' and not e.attrib for row in rows for e in row),
                    'Column repair cannot infer spans or named entry placement')
            groups[0].set('cols', str(width))
        elif op == 'literal_list_labels':
            require(node.tag == CN + 'list' and node.get('list-type') == 'enumerated' and
                    node.get('number-style') == 'upper-roman' and node.get('mark-prefix') == '(' and
                    node.get('mark-suffix') == ')' and node.get('start-value', '1') == '1',
                    'Unexpected list labeling')
            require(record['labels'] == ['(I)', '(II)', '(III)', '(IV)'] and len(node) == 4 and
                    all(c.tag == CN + 'item' for c in node) and not compact(node.text or ''),
                    'Reviewed four-item list changed')
            node.tag = CN + 'section'
            node.attrib = {'id': record['element_id']}
            for label, child in zip(record['labels'], node):
                child.tag = CN + 'para'
                child.text = label + ' ' + (child.text or '')
        elif op == 'replace_formulas':
            require(node.tag == CN + 'para', 'Formula edit requires its reviewed paragraph')
            formulas = list(node.iter(MATH + 'math'))
            edits = record['formulas']
            require(len(formulas) == record['formula_count'] and
                    len({r['index'] for r in edits}) == len(edits), 'Formula positions changed')
            for edit in edits:
                require(0 <= edit['index'] < len(formulas), 'Formula index outside paragraph')
                old = formulas[edit['index']]
                require(digest(ET.tostring(old, encoding='utf-8')) == edit['before_sha256'],
                        'Reviewed formula changed')
                xml = edit['replacement_xml'].encode('utf-8')
                require(b'<!DOCTYPE' not in xml and b'<!ENTITY' not in xml, 'Custom formula entities')
                new = ET.fromstring(xml)
                require(new.tag == MATH + 'math' and all(e.tag.startswith(MATH) for e in new.iter()),
                        'Replacement must contain MathML only')
                parent = parents[old]
                position = list(parent).index(old)
                new.tail = old.tail
                parent.remove(old)
                parent.insert(position, deepcopy(new))
        elif op == 'media_description':
            require(node.tag == CN + 'media' and node.get('alt') == record['before'],
                    'Reviewed media description changed')
            require(record['after'].strip() and record['media_sha256'], 'Missing reviewed image evidence')
            node.set('alt', record['after'])
        else:
            raise ValueError('Unknown reviewed Physics edit: ' + op)
    return ET.tostring(root, encoding='utf-8'), [r['element_id'] for r in selected]


def omit_empty_exercises(raw):
    """Remove only empty wrappers left after external placeholders are omitted."""
    require(b'<!DOCTYPE' not in raw and b'<!ENTITY' not in raw, 'Custom XML entities are unsupported')
    root = ET.fromstring(raw)
    parents = {child: parent for parent in root.iter() for child in parent}
    removed = {'problem': [], 'exercise': []}
    for node in reversed(list(root.iter())):
        kind = node.tag.removeprefix(CN)
        if kind not in removed or node not in parents:
            continue
        descendants = list(node.iter())
        if (any(compact(e.text or '') or (e is not node and compact(e.tail or '')) for e in descendants) or
                any(e.tag not in {CN + 'exercise', CN + 'problem', CN + 'para'} for e in descendants) or
                any(set(e.attrib) - {'id', 'class'} for e in descendants)):
            continue
        removed[kind].append(node.get('id'))
        remove_preserving_tail(parents[node], node)
    return ET.tostring(root, encoding='utf-8'), removed
