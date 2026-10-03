"""Keep reviewed documents and worked pairs intact while reserving held-out text."""
import re
import shlex


def normalized(text):
    # Preserve case, signs, decimal points and variables in mathematical text.
    return re.sub(r'\s+', ' ', text).strip()


def question(text):
    if text.count('\n\nQuestion: ') != 1 or text.count('\n\nAnswer: ') != 1:
        raise ValueError('Expected one question and one solution')
    title, rest = text.split('\n\nQuestion: ')
    prompt, answer = rest.split('\n\nAnswer: ')
    if not title.strip() or not prompt.strip() or not answer.removeprefix('Solution').strip():
        raise ValueError('Incomplete worked example')
    if '\x1e' in text:
        raise ValueError('A lesson cannot contain a document separator')
    return normalized(prompt)


def probe_contexts(raw):
    """Decode existing std::quoted probe records solely for context protection."""
    fields = shlex.split(raw.decode('utf-8'), comments=False, posix=True)
    if (len(fields) < 2 or fields[0] not in ('SGPROBE1', 'SGPROBE2')
            or len(fields) != 2 + 8 * int(fields[1])):
        raise ValueError('Invalid held-out probe file')
    result = []
    for start in range(2, len(fields), 8):
        if fields[start + 3] not in ('0', '1') or not fields[start + 4]:
            raise ValueError('Invalid held-out probe record')
        result.append(normalized(fields[start + 4]))
    return result


class PairReservations:
    """Exact full/question/long-paragraph checks; never remove part of a pair."""
    def __init__(self):
        self.whole, self.questions, self.paragraphs, self.contexts = {}, {}, {}, {}

    def add(self, text, owner, paired=False):
        self.whole.setdefault(normalized(text), owner)
        if paired:
            self.questions.setdefault(question(text), owner)
        for part in re.split(r'\n\s*\n', text):
            value = normalized(part)
            if len(value) >= 120:
                self.paragraphs.setdefault(value, owner)

    def conflict(self, text, *, paired=True):
        prompt, full = question(text) if paired else None, normalized(text)
        for kind, value, index in (('whole_lesson' if paired else 'whole_document', full, self.whole),
                                   ('question', prompt, self.questions)):
            if value is not None and value in index:
                return dict(kind=kind, reserved_by=index[value])
        for part in re.split(r'\n\s*\n', text):
            value = normalized(part)
            if len(value) >= 120 and value in self.paragraphs:
                return dict(kind='paragraph_at_least_120_characters', reserved_by=self.paragraphs[value])
        for context, owner in self.contexts.items():
            if context in full:
                return dict(kind='authored_probe_context', reserved_by=owner)
        return None


def select_documents(rows, protected, *, paired=False, reserve_rejected_holdouts=True):
    """Keep documents intact and reserve evaluation content before training."""
    if not paired:
        if len({r['id'] for r in rows}) != len(rows):
            raise ValueError('Duplicate document identity')
        for row in rows:
            if not row['text'].strip() or '\x1e' in row['text'] or row['split'] not in ('train', 'validation', 'test'):
                raise ValueError('Expected one complete document and a known split')
    reservations = PairReservations()
    for owner, raw, kind in protected:
        if kind == 'probe':
            for context in probe_contexts(raw):
                reservations.contexts.setdefault(context, owner)
        elif kind == 'documents':
            for text in raw.decode('utf-8').split('\x1e'):
                if not text.strip():
                    raise ValueError('Empty protected document')
                reservations.add(text, owner)
        else:
            raise ValueError('Unknown held-out source type')
    kept, omitted = [], []
    priority = {'test': 0, 'validation': 1, 'train': 2}
    for row in sorted(rows, key=lambda r: priority[r['split']]):
        text = row['text']
        conflict = reservations.conflict(text, paired=paired)
        if conflict:
            omitted.append(dict(id=row['id'], split=row['split'], **conflict))
        else:
            kept.append(row)
        if not conflict or (reserve_rejected_holdouts and row['split'] != 'train'):
            reservations.add(text, row['id'], paired=paired)
    return kept, omitted


def select_pairs(rows, protected):
    """Preserve the existing worked-lesson selection policy and output order."""
    return select_documents(rows, protected, paired=True, reserve_rejected_holdouts=False)
