"""Explicit optional MathML notation for the pinned Prealgebra 2e review.

This extends extraction, not source admission. The existing Astronomy converter
keeps its original supported subset unless this callback is supplied. The
enclosure syntax uses MathJax's documented TeX ``enclose`` extension.
"""
from corpus.openstax import MATH, compact


def horizontal_bar(node):
    # Single-child grouping is equivalent only without extra group attributes.
    while node.tag == MATH + 'mrow' and len(node) == 1 and not node.attrib:
        node = node[0]
    return (node.tag in (MATH + 'mo', MATH + 'mtext') and not len(node)
            and bool(compact(node.text or ''))
            and set(compact(node.text or '')) <= set('_¯–—‾'))


def prealgebra_extension(node, render):
    """Preserve reviewed notation; reject unknown forms, attributes and arity."""
    name = node.tag.removeprefix(MATH)
    children = list(node)
    if not children or compact(node.text or '') or any(compact(c.tail or '') for c in children):
        raise ValueError(f'Invalid extended MathML contents: {name}')
    if name in ('mroot', 'mover', 'munder') and len(children) != 2:
        raise ValueError(f'Invalid extended MathML arity: {name}')
    if name in ('mroot', 'mphantom') and node.attrib:
        raise ValueError(f'Unreviewed extended MathML attributes: {name}')
    if name in ('mover', 'munder'):
        accent = 'accent' if name == 'mover' else 'accentunder'
        if set(node.attrib) - {accent} or node.get(accent, 'false') not in ('true', 'false'):
            raise ValueError(f'Unreviewed extended MathML attributes: {name}')
    values = [render(c) for c in children]
    if name == 'mroot':
        return r'\sqrt[' + values[1] + ']{' + values[0] + '}'
    if name in ('mover', 'munder'):
        if horizontal_bar(children[1]):
            command = r'\overline' if name == 'mover' else r'\underline'
            return command + '{' + values[0] + '}'
        command = r'\overset' if name == 'mover' else r'\underset'
        return command + '{' + values[1] + '}{' + values[0] + '}'
    body = ' '.join(values)
    if name == 'mstyle':
        styles = {(('fontweight', 'bold'),): r'\mathbf',
                  (('fontstyle', 'italic'),): r'\mathit'}
        command = styles.get(tuple(sorted(node.attrib.items())))
        if command is None:
            raise ValueError('Unreviewed MathML style')
        return command + '{' + body + '}'
    if name == 'mphantom':
        return r'\phantom{' + body + '}'
    if name == 'menclose':
        if set(node.attrib) != {'notation'} or node.get('notation') not in ('longdiv', 'updiagonalstrike'):
            raise ValueError('Unreviewed MathML enclosure')
        return r'\enclose{' + node.get('notation') + '}{' + body + '}'
    raise ValueError(f'Unsupported Prealgebra MathML structure: {name}')
