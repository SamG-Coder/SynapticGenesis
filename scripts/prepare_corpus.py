"""Reproducible, explicit book acquisition and corpus preparation (stdlib only).

This script prepares data; it does not train a model. Every source is selected in
data/sources.json. Original files and catalogue pages are retained with SHA-256.
"""
import argparse
import hashlib
import html
import json
import re
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from corpus.selection import require_training_spec


def sha(data):
    return hashlib.sha256(data).hexdigest()


def fetch(url, destination):
    if destination.exists():
        return destination.read_bytes()
    request = urllib.request.Request(url, headers={'User-Agent': 'SynapticGenesis/0.1 selected research corpus'})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                raw = response.read()
            destination.write_bytes(raw)
            return raw
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))


def clean(raw, source):
    text = raw.decode('utf-8-sig').replace('\r\n', '\n').replace('\r', '\n')
    begin = re.search(r'\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK[^\n]*\n', text, re.I)
    end = re.search(r'\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK', text, re.I)
    if not begin or not end or begin.end() >= end.start():
        raise ValueError('Missing or invalid Gutenberg body delimiters')
    body = text[begin.end():end.start()]
    if source.get('body_start'):
        start = re.search(source['body_start'], body, re.M)
        if not start:
            raise ValueError(f"Selected body start not found for {source['id']}")
        body = body[start.start():]
    if source.get('body_end'):
        end = re.search(source['body_end'], body, re.M)
        if not end:
            raise ValueError(f"Selected body end not found for {source['id']}")
        body = body[:end.start()]
    body = re.sub(r'\[(?:Pg|Page)\s+[^\]]+\]', '', body, flags=re.I)
    body = re.sub(r'\[(?:Illustration|Picture:)[^\]]*\]', '', body, flags=re.I)
    # Accessible phonetic editions sometimes spell out combining marks as
    # editorial tags. Preserve the underlying letters, not the tag vocabulary.
    body = re.sub(r'\{~COMBINING [A-Z ]+~\}', '', body)
    body = re.sub(r'(?im)^[ \t]*(?:\d+[ \t]+)?(?:FIRST|SECOND|THIRD) READER\.?[ \t]*(?:\d+)?[ \t]*$', '', body)
    body = re.sub(r'[ \t]+\n', '\n', body)
    body = re.sub(r'\n{4,}', '\n\n\n', body).strip() + '\n'
    if '\ufffd' in body or '\x1e' in body or len(body) < 10000:
        raise ValueError('Invalid or suspicious text body')
    return body


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sources', type=Path, default=Path('data/sources.json'))
    parser.add_argument('--raw', type=Path, default=Path('data/raw'))
    parser.add_argument('--out', type=Path, default=Path('data/prepared/foundations-v1'))
    parser.add_argument('--download-only', action='store_true', help='Cache selected editions for inspection without preparing a corpus')
    args = parser.parse_args()
    spec_bytes = args.sources.read_bytes()
    spec = require_training_spec(args.sources)
    stage_ids = [s['id'] for s in spec['stages']]
    ids = [s['id'] for s in spec['sources']]
    if (len(stage_ids) != len(set(stage_ids)) or len(ids) != len(set(ids))
            or not stage_ids or not ids):
        raise ValueError('Stage and source IDs must be nonempty and unique')
    for source in spec['sources']:
        if (source['stage'] not in stage_ids or source['split'] not in ('train', 'validation', 'test')
                or not isinstance(source['id'], int) or source['id'] <= 0):
            raise ValueError(f'Invalid source: {source}')
    args.raw.mkdir(parents=True, exist_ok=True)
    if not args.download_only:
        if args.out.exists() and any(args.out.iterdir()):
            raise SystemExit('Output directory is not empty. Use a new --out directory for a new corpus version.')
        args.out.mkdir(parents=True, exist_ok=True)
    splits = {k: [] for k in ('train', 'validation', 'test')}
    stages = {stage: [] for stage in stage_ids}
    seen_paragraphs = set()
    records = []
    # Reserve held-out material first, then remove identical substantial paragraphs
    # from training. Whole-book splits also keep adjacent windows out of other splits.
    ordered = sorted(spec['sources'], key=lambda s: {'test': 0, 'validation': 1, 'train': 2}[s['split']])
    for source in ordered:
        book_id = source['id']
        page_url = f'https://www.gutenberg.org/ebooks/{book_id}'
        catalog = fetch(page_url, args.raw / f'{book_id}.html')
        page = catalog.decode('utf-8')
        if 'Public domain in the USA' not in page:
            raise ValueError(f'Unexpected rights metadata for {book_id}')
        title_match = re.search(r'<title>(.*?)</title>', page, re.S)
        catalog_title = html.unescape(title_match.group(1)) if title_match else ''
        if source['title'].casefold() not in catalog_title.casefold():
            raise ValueError(f'Title differs from selected edition: {catalog_title}')
        # Resolve the actual UTF-8 full-text link from the selected catalogue page.
        candidates = re.findall(r'href="([^"]+)"[^>]*[^\n]*?Plain Text', page)
        candidates += re.findall(r'href="([^"]*(?:\.txt(?:\.utf-8)?|/pg\d+\.txt))"', page)
        candidates = [u for u in candidates if '/ebooks/' in u or '/cache/' in u or '/files/' in u]
        if not candidates:
            raise ValueError(f'No text edition found for {book_id}: {catalog_title}')
        url = urllib.parse.urljoin(page_url, html.unescape(candidates[0]))
        if urllib.parse.urlparse(url).hostname not in ('www.gutenberg.org', 'gutenberg.org'):
            raise ValueError(f'Unexpected download host: {url}')
        raw = fetch(url, args.raw / f'{book_id}.txt')
        if args.download_only:
            print(f'{book_id}: {len(raw)} bytes cached; {catalog_title}', flush=True)
            continue
        body = clean(raw, source)
        kept = []
        duplicates = 0
        for paragraph in re.split(r'\n\s*\n', body):
            normalized = re.sub(r'\s+', ' ', paragraph).strip()
            digest = sha(normalized.encode())
            if len(normalized) >= 120 and digest in seen_paragraphs:
                duplicates += 1
                continue
            if len(normalized) >= 120:
                seen_paragraphs.add(digest)
            kept.append(paragraph)
        encoded = ('\n\n'.join(kept).strip() + '\n').encode('utf-8')
        splits[source['split']].append(encoded)
        if source['split'] == 'train':
            stages[source['stage']].append(encoded)
        (args.out / f'{book_id}.txt').write_bytes(encoded)
        records.append({**source, 'catalog_title': catalog_title, 'catalog_url': page_url,
                        'download_url': url, 'raw_sha256': sha(raw), 'clean_sha256': sha(encoded),
                        'catalog_sha256': sha(catalog),
                        'clean_bytes': len(encoded), 'approx_words': len(encoded.decode().split()),
                        'removed_duplicate_paragraphs': duplicates,
                        'catalog_rights': 'Public domain in the USA',
                        'edition_note': spec.get('edition_note', 'Historic reading textbook. Original notices are retained in the cached raw edition. The repository code license does not relicense source books.')})
        print(f"{source['split']:10s} {book_id:6d} {len(encoded):9d} bytes  {catalog_title}", flush=True)
    if args.download_only:
        return
    # Preserve the exact allowlist bytes as well as their hash. Windows working
    # copies and Git checkouts may use different line endings for identical JSON.
    (args.out / 'source-spec.json').write_bytes(spec_bytes)
    outputs = {}
    for name, documents in splits.items():
        payload = b'\x1e'.join(documents)
        (args.out / f'{name}.dat').write_bytes(payload)
        outputs[name] = {'bytes': len(payload), 'documents': len(documents), 'sha256': sha(payload)}
    stage_outputs = []
    cumulative = []
    schedule = ['SGCURRICULUM1']
    for stage in spec['stages']:
        documents = stages[stage['id']]
        if not documents:
            raise ValueError(f"No training sources in stage {stage['id']}")
        payload = b'\x1e'.join(documents)
        filename = f"stage-{stage['id']}.dat"
        (args.out / filename).write_bytes(payload)
        cumulative.extend(documents)
        cumulative_payload = b'\x1e'.join(cumulative)
        cumulative_file = f"through-stage-{stage['id']}.dat"
        (args.out / cumulative_file).write_bytes(cumulative_payload)
        schedule.append(f'{(len(stage_outputs)+1)*2000} "{cumulative_file}" 1')
        stage_outputs.append({**stage, 'file': filename, 'bytes': len(payload),
                              'documents': len(documents), 'sha256': sha(payload),
                              'cumulative_file': cumulative_file, 'cumulative_bytes': len(cumulative_payload),
                              'cumulative_sha256': sha(cumulative_payload)})
    (args.out / 'curriculum.sg').write_bytes(('\n'.join(schedule)+'\n').encode('utf-8'))
    manifest = {'version': spec['version'], 'created_utc': datetime.now(timezone.utc).isoformat(),
                'source_spec_sha256': sha(spec_bytes), 'tokenizer': 'UTF-8 bytes, fixed IDs 0..255, no learned tokenizer',
                'document_separator': '0x1e, excluded from sampled windows', 'outputs': outputs,
                'stages': stage_outputs, 'purpose': spec['purpose'], 'policy': spec['policy'],
                'stage_note': spec['stage_note'],
                'deduplication': 'Exact normalized paragraphs >=120 characters, held-out books reserved before training books',
                'limitations': 'Small historical reading curriculum; not a broad modern knowledge or instruction dataset. Historical moral and social assumptions remain in the text. Whole-book splits and exact-paragraph deduplication do not detect shared tales, paraphrases, or shorter overlap. The validation and test books cover different reading levels; stage-specific mastery evaluation is still needed.',
                'sources': records}
    (args.out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(outputs, indent=2))


if __name__ == '__main__':
    main()
