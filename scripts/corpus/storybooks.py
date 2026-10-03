"""Read explicitly selected Storybooks Canada editions without images or audio."""
import html
import re

from prepare_corpus import fetch, sha


def normalized(text):
    # The website typesets quotation marks and ellipses differently from its
    # Markdown source. This comparison never changes the prepared source text.
    typography = str.maketrans({'\u2018': "'", '\u2019': "'", '\u201c': '"',
                               '\u201d': '"', '\u2026': '...'})
    return re.sub(r'\s+', ' ', text.translate(typography)).strip()


def parse(raw, selected):
    if sha(raw) != selected['raw_sha256']:
        raise ValueError('Selected Markdown edition changed')
    text = raw.decode('utf-8').replace('\r\n', '\n')
    if any(ord(c) < 32 and c not in '\n\t' for c in text) or '\ufffd' in text:
        raise ValueError('Invalid source text character')
    sections = re.split(r'^##[ \t]*$', text, flags=re.M)
    if len(sections) < 3 or sections[0].strip() != '# '+selected['title']:
        raise ValueError('Missing title or page boundaries')
    metadata = {}
    for line in sections[-1].strip().splitlines():
        match = re.fullmatch(r'\* ([A-Za-z ]+): (.+)', line)
        if not match or match[1] in metadata:
            raise ValueError('Invalid or duplicated attribution field')
        metadata[match[1]] = match[2]
    if metadata != selected['attribution'] or metadata.get('Language') != 'en':
        raise ValueError('Attribution or language differs from selection')
    if metadata.get('License') != '[CC-BY]':
        raise ValueError('This selection permits only the declared CC BY editions')
    pages = [page.strip() for page in sections[1:-1]]
    if len(pages) != selected['pages'] or any(not page for page in pages):
        raise ValueError('Missing or unexpected story pages')
    if any(re.search(r'^[#*]', page, re.M) for page in pages):
        raise ValueError('Unexpected markup inside story text')
    document = ('\n\n'.join(pages)+'\n').encode('utf-8')
    if sha(document) != selected['body_sha256']:
        raise ValueError('Selected story body changed')
    return document, pages


def verify_publication(catalog, selected, pages):
    if sha(catalog) != selected['catalog_sha256']:
        raise ValueError('Selected publisher snapshot changed; review a new edition')
    text = catalog.decode('utf-8')
    licenses = set(re.findall(r'<a rel="license" href="([^"]+)"', text))
    if licenses != {selected['license_url']}:
        raise ValueError('Unexpected publisher license')
    if selected['license_url'] not in ('https://creativecommons.org/licenses/by/3.0/',
                                       'https://creativecommons.org/licenses/by/4.0/'):
        raise ValueError('Unselected license version')
    title = re.search(r'<h1><span class="def">(.*?)</span>', text, re.S)
    if not title or normalized(html.unescape(title[1])) != normalized(selected['title']):
        raise ValueError('Publisher title differs from selection')
    published = re.findall(r'<div class="[^"\n]*\blevel([1-5])-txt def"><h3>(.*?)</h3>', text, re.S)
    if len(published) != len(pages) or {int(level) for level, _ in published} != {selected['reading_level']}:
        raise ValueError('Publisher page count or reading level differs')
    for page, (_, content) in zip(pages, published):
        content = html.unescape(re.sub('<[^>]+>', ' ', content))
        if normalized(page) != normalized(content):
            raise ValueError('Source and published English pages differ')


def load(selected, commit, cache):
    if not re.fullmatch(r'[0-9a-f]{40}', commit):
        raise ValueError('Require a pinned upstream commit')
    name = selected['source_file']
    if not re.fullmatch(r'[0-9]{4}_[a-z0-9-]+\.md', name) or name[:4] != selected['id']:
        raise ValueError('Invalid selected source filename')
    source_url = f'https://raw.githubusercontent.com/global-asp/sbc-source/{commit}/en/{name}'
    catalog_url = f'https://storybookscanada.ca/stories/en/{selected["id"]}/'
    cache.mkdir(parents=True, exist_ok=True)
    raw = fetch(source_url, cache/name)
    catalog = fetch(catalog_url, cache/(selected['id']+'.html'))
    document, pages = parse(raw, selected)
    verify_publication(catalog, selected, pages)
    return document, dict(**selected, source_url=source_url, catalog_url=catalog_url,
                         clean_bytes=len(document), approx_words=len(document.decode('utf-8').split()))
