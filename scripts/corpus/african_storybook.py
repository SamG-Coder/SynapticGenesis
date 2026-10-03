"""Load only selected, revision-pinned African Storybook Markdown editions."""
import re

from corpus.storybooks import parse
from prepare_corpus import fetch, sha


LICENSES = {'https://creativecommons.org/licenses/by/3.0/',
            'https://creativecommons.org/licenses/by/4.0/'}


def index_entries(raw, expected_sha256):
    if sha(raw) != expected_sha256:
        raise ValueError('Pinned English source index changed')
    entries = {}
    pattern = r'(\d{4}) \| \[(.+)\]\(([^)]+)\) \| \[([^\]]+)\]\(([^)]+)\)'
    for line in raw.decode('utf-8').splitlines():
        if not re.match(r'^\d{4} \|', line):
            continue
        match = re.fullmatch(pattern, line)
        if not match or match[1] in entries:
            raise ValueError('Malformed or duplicated source-index entry')
        entries[match[1]] = dict(title=match[2], publisher_url=match[3],
                                  license=match[4], license_url=match[5])
    if not entries:
        raise ValueError('Empty source index')
    return entries


def verify_index_entry(selected, entries):
    entry = entries.get(selected['id'])
    if not entry or entry != selected['index_entry'] or entry['title'] != selected['title']:
        raise ValueError('Selected title or source-index metadata differs')
    if (entry['license'] != 'CC-BY' or entry['license_url'] not in LICENSES
            or selected['attribution'].get('License') != '[CC-BY]'):
        raise ValueError('Source and index must agree on a selected CC BY license')


def load(selected, commit, entries, cache):
    if not re.fullmatch(r'[0-9a-f]{40}', commit):
        raise ValueError('Require a pinned upstream commit')
    name = selected['source_file']
    if not re.fullmatch(r'[0-9]{4}_[a-z0-9-]+\.md', name) or name[:4] != selected['id']:
        raise ValueError('Invalid selected source filename')
    verify_index_entry(selected, entries)
    url = f'https://raw.githubusercontent.com/global-asp/asp-source/{commit}/en/{name}'
    raw = fetch(url, cache/name)
    document, _ = parse(raw, selected)
    return document, dict(**selected, source_url=url, clean_bytes=len(document),
                         approx_words=len(document.decode('utf-8').split()))
