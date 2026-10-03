"""Prepare a self-contained, append-only edition of a selected live curriculum.

No model is loaded or trained here. The native --extend-curriculum operation
authenticates the previous schedule against the saved learning state.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def documents(raw):
    docs = raw.split(b'\x1e')
    if not docs or any(len(doc) < 2 for doc in docs):
        raise ValueError('Every document must contain at least two bytes; no empty separators')
    return docs


def check_answers(docs, scale):
    if scale > 1:
        marker = b'\nAnswer: '
        if any(doc.count(marker) != 1 or not doc.split(marker)[-1] for doc in docs):
            raise ValueError('Answer emphasis requires one nonempty Answer field per new document')


def read_schedule(path):
    raw = path.read_bytes()
    if len(raw) > 1024 * 1024:
        raise ValueError('Curriculum schedule is too large')
    lines = raw.decode('utf-8').splitlines()
    if not lines or lines[0] not in ('SGCURRICULUM1', 'SGCURRICULUM2', 'SGCURRICULUM3'):
        raise ValueError('Unsupported curriculum schedule')
    version = int(lines[0][-1])
    stages, previous = [], b''
    for line in lines[1:]:
        if not line.strip():
            continue
        # Match std::quoted: any escaped character loses its backslash inside
        # quotes; an unquoted filename is read as one literal whitespace token.
        match = re.fullmatch(r'\s*(\S+)\s+(?:"((?:\\.|[^"\\])*)"|(\S+))\s+(.*?)\s*', line)
        if not match:
            raise ValueError('Invalid curriculum row')
        filename = re.sub(r'\\(.)', r'\1', match[2]) if match[2] is not None else match[3]
        fields = [match[1], filename, *match[4].split()]
        if len(fields) != version + 2:
            raise ValueError('Invalid curriculum row')
        end, filename, rate = int(fields[0]), fields[1], float(fields[2])
        scope = fields[3] if version >= 2 else 'all'
        answer = float(fields[4]) if version >= 3 else 1
        if (not filename or not (stages[-1]['end_update'] if stages else 0) < end <= 100000000
                or not math.isfinite(rate) or not 0 < rate <= 10 or scope not in ('all', 'new')
                or not math.isfinite(answer) or not 1 <= answer <= 1000 or len(stages) >= 4096):
            raise ValueError('Invalid curriculum policy')
        source = (path.parent / filename).resolve()
        content = source.read_bytes()
        docs = documents(content)
        if previous and (not content.startswith(previous + b'\x1e') or len(docs) <= len(documents(previous))):
            raise ValueError('Old curriculum does not append complete unchanged documents')
        introduced = docs[len(documents(previous)):] if previous else docs
        check_answers(introduced, answer)
        stages.append(dict(end_update=end, source=source, content=content,
                           rate=fields[2], scope=scope, answer=fields[4] if version >= 3 else '1'))
        previous = content
    if not stages:
        raise ValueError('Curriculum has no stages')
    return raw, stages


def prepare(curriculum, data, out, end_update, rate_scale=.25, scope='new', answer_scale=1,
            holdouts=()):
    curriculum, data, out = Path(curriculum), Path(data), Path(out)
    parent_raw, stages = read_schedule(curriculum)
    if (not stages[-1]['end_update'] < end_update <= 100000000 or len(stages) >= 4096
            or not math.isfinite(rate_scale) or not 0 < rate_scale <= 10 or scope not in ('all', 'new')
            or not math.isfinite(answer_scale) or not 1 <= answer_scale <= 1000):
        raise ValueError('Extension must add a later stage within the native policy limits')
    selected = data.read_bytes()
    added = documents(selected)
    check_answers(added, answer_scale)
    earlier = documents(stages[-1]['content'])
    if len(set(added)) != len(added) or set(added).intersection(earlier):
        raise ValueError('Selected addition repeats an exact document; choose distinct new lessons')
    combined = stages[-1]['content'] + b'\x1e' + selected
    holdout_records = []
    all_docs = set(earlier + added)
    for path in map(Path, holdouts):
        raw = path.read_bytes()
        if all_docs.intersection(documents(raw)):
            raise ValueError('Extended learning corpus contains a held-out document')
        holdout_records.append(dict(source=str(path.resolve()), sha256=digest(raw)))
    stages.append(dict(end_update=end_update, source=data.resolve(), content=combined,
                       rate=format(rate_scale, '.17g'), scope=scope, answer=format(answer_scale, '.17g')))
    # Preflight finishes before creating output. Existing editions are never
    # overwritten. Copies keep the new schedule independent of old path layout.
    out.mkdir(parents=True, exist_ok=False)
    rows, records = [], []
    for index, stage in enumerate(stages, 1):
        name = f'edition-{index}.dat'
        (out / name).write_bytes(stage['content'])
        rows.append(f'{stage["end_update"]} "{name}" {stage["rate"]} {stage["scope"]} {stage["answer"]}')
        records.append(dict(file=name, sha256=digest(stage['content']), bytes=len(stage['content']),
                            documents=len(documents(stage['content'])), end_update=stage['end_update'],
                            rate_scale=float(stage['rate']), scope=stage['scope'],
                            answer_scale=float(stage['answer'])))
    schedule = ('SGCURRICULUM3\n' + '\n'.join(rows) + '\n').encode('utf-8')
    (out / 'curriculum.sg').write_bytes(schedule)
    manifest = dict(version='curriculum-extension-v1',
                    parent_schedule=str(curriculum.resolve()), parent_schedule_sha256=digest(parent_raw),
                    schedule_sha256=digest(schedule), previous_stages=len(stages)-1,
                    selected_source=str(data.resolve()), selected_sha256=digest(selected),
                    selected_bytes=len(selected), added_documents=len(added),
                    checked_holdouts=holdout_records, editions=records,
                    limits='Exact document checks only; no semantic overlap or curriculum mastery assessment.')
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    return manifest


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--curriculum', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--end-update', type=int, required=True, help='Cumulative online observation count')
    p.add_argument('--rate-scale', type=float, default=.25)
    p.add_argument('--scope', choices=['all', 'new'], default='new')
    p.add_argument('--answer-scale', type=float, default=1)
    p.add_argument('--holdout', type=Path, action='append', default=[])
    a = p.parse_args()
    result = prepare(a.curriculum, a.data, a.out, a.end_update, a.rate_scale, a.scope, a.answer_scale, a.holdout)
    print(json.dumps(result, indent=2))
