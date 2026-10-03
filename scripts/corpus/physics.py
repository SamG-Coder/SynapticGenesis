"""Optional extraction policy for the pinned OpenStax Physics review.

This does not admit a source or change the astronomy/prealgebra converters.
Mathematical grouping, bold symbols and one left isotope-index pair are kept;
layout-only size hints are normalized rather than reproduced typographically.
"""
from collections import Counter
import re
import xml.etree.ElementTree as ET

from corpus.openstax import CN, MATH, compact, extract
from corpus.prealgebra_math import prealgebra_extension

OMITTED_NOTE_CLASSES = {'os-teacher', 'learning-objectives', 'watch-physics', 'snap-lab',
                       'os-interactive', 'virtual-physics'}
PUNCTUATION_REPAIRS = {'\u0092': '’', '\u0093': '“', '\u0094': '”', '\u0096': '–', '\u0097': '—'}


def physics_extension(node, render):
    name = node.tag.removeprefix(MATH)
    if name not in ('mstyle', 'mmultiscripts'):
        return prealgebra_extension(node, render)
    children = list(node)
    if not children or compact(node.text or '') or any(compact(c.tail or '') for c in children):
        raise ValueError(f'Invalid Physics MathML contents: {name}')
    if name == 'mstyle':
        attrs = node.attrib
        if attrs in ({'mathvariant': 'bold', 'mathsize': 'normal'}, {'mathvariant': 'bold'}):
            # Unlike mathbf, boldsymbol also preserves a bold Greek symbol.
            return r'\boldsymbol{' + ' '.join(render(c) for c in children) + '}'
        if attrs in ({'mathsize': 'normal'}, {'scriptlevel': '+1'}):
            return '{' + ' '.join(render(c) for c in children) + '}'
        return prealgebra_extension(node, render)
    # Reviewed isotope notation only, not a generic tensor-index converter.
    if node.attrib or len(children) != 4 or children[1].tag != MATH + 'mprescripts':
        raise ValueError('Unsupported Physics multiscript layout')
    marker = children[1]
    if marker.attrib or len(marker) or compact(marker.text or ''):
        raise ValueError('Invalid Physics prescript marker')

    def index(child):
        if child.tag == MATH + 'none':
            if child.attrib or len(child) or compact(child.text or ''):
                raise ValueError('Invalid empty Physics prescript')
            return ''
        value = render(child)
        if not value:
            raise ValueError('Empty Physics prescript expression')
        return value

    base = render(children[0])
    if not base:
        raise ValueError('Empty Physics multiscript base')
    sub, sup = index(children[2]), index(children[3])
    if not sub and not sup:
        raise ValueError('Physics multiscript has no indices')
    scripts = ('_{' + sub + '}' if sub else '') + ('^{' + sup + '}' if sup else '')
    return '{}' + scripts + '{' + base + '}'


def remove_preserving_tail(parent, child):
    position = list(parent).index(child)
    tail = child.tail or ''
    if position:
        previous = parent[position - 1]
        previous.tail = (previous.tail or '') + tail
    else:
        parent.text = (parent.text or '') + tail
    parent.remove(child)


def physics_document(raw):
    """Return a transformed copy plus counts; never alter cached source bytes."""
    if b'<!DOCTYPE' in raw or b'<!ENTITY' in raw:
        raise ValueError('External/custom XML entities are unsupported')
    root = ET.fromstring(raw)
    if root.tag != CN + 'document':
        raise ValueError('Expected a CNXML document')
    stats = Counter()

    def walk(parent):
        for child in list(parent):
            classes = set(child.get('class', '').split())
            if child.tag == CN + 'link' and classes & {'os-embed', 'os\u2010embed'}:
                if (set(child.attrib) != {'class', 'url'} or len(child) or compact(child.text or '') or
                        not re.fullmatch(r'#ost/api/ex/[A-Za-z0-9_-]+', child.get('url', ''))):
                    raise ValueError('Unsupported Physics exercise embed')
                # These empty links require an external exercise service. They
                # contain no question or answer in this pinned source snapshot.
                stats['omitted_external_exercise_placeholders'] += 1
                if 'os\u2010embed' in classes:
                    stats['external_exercise_unicode_hyphen_aliases'] += 1
                remove_preserving_tail(parent, child)
                continue
            if child.tag == CN + 'note' and classes & OMITTED_NOTE_CLASSES:
                stats['omitted_notes'] += 1
                for name in sorted(classes & OMITTED_NOTE_CLASSES):
                    stats['omitted_note_' + name] += 1
                remove_preserving_tail(parent, child)
                continue
            if child.tag == CN + 'table' and 'section-key-terms' in classes:
                # This is a vocabulary grid, not an explanatory data table.
                # Omit it explicitly; do not repair inconsistent column counts.
                stats['omitted_vocabulary_grids'] += 1
                remove_preserving_tail(parent, child)
                continue
            walk(child)
            if child.tag == CN + 'list' and child.find(CN + 'title') is not None:
                entries = list(child)
                if (not entries or entries[0].tag != CN + 'title' or
                        any(e.tag != CN + 'item' for e in entries[1:]) or
                        compact(child.text or '') or compact(entries[0].tail or '')):
                    raise ValueError('Unsupported titled Physics list')
                # Preserve its source title, item order, numbering and reference
                # ID in an enclosing section understood by the base extractor.
                position = list(parent).index(child)
                wrapper = ET.Element(CN + 'section')
                if 'id' in child.attrib:
                    wrapper.set('id', child.attrib.pop('id'))
                wrapper.tail, child.tail = child.tail, None
                child.remove(entries[0])
                wrapper.append(entries[0])
                wrapper.append(child)
                parent.remove(child)
                parent.insert(position, wrapper)
                stats['titled_lists_preserved'] += 1
    walk(root)
    return ET.tostring(root, encoding='utf-8'), dict(stats)


def extract_physics(raw, module_titles=None):
    transformed, policy = physics_document(raw)
    text, extraction = extract(transformed, module_titles, math_extension=physics_extension)
    decoded = text.decode('utf-8')
    repaired = {}
    for wrong, replacement in PUNCTUATION_REPAIRS.items():
        count = decoded.count(wrong)
        if count:
            repaired[f'U+{ord(wrong):04X}'] = count
            decoded = decoded.replace(wrong, replacement)
    if any(127 <= ord(c) <= 159 for c in decoded):
        raise ValueError('Unreviewed control character in Physics text')
    return decoded.encode('utf-8'), dict(policy=policy, extraction=extraction, punctuation_repairs=repaired)
