"""Independent structured reference for grouped replay and sequential source exposure.

This simulates only source selection, reservoir admission, RNG and counters. It
does not implement neurons or a surrogate model for native learning quality.
"""
from dataclasses import dataclass
import struct

from adaptation_sources import fnv
from experiment_checkpoint import policy_checkpoint

MASK = (1 << 64) - 1


class Random64:
    def __init__(self, state):
        self.state = state

    def next(self):
        state = self.state ^ (self.state >> 12)
        state = (state ^ (state << 25)) & MASK
        self.state = state ^ (state >> 27)
        return (self.state * 2685821657736338717) & MASK

    def below(self, bound):
        assert bound > 0
        limit = ((-bound) & MASK) % bound
        while True:
            value = self.next()
            if value >= limit:
                return value % bound


@dataclass
class Group:
    end: int
    seen: int
    updates: int
    pairs: int
    items: list


def schedule_identity(raw, stages):
    value = fnv(raw)
    for stage in stages:
        value = fnv(struct.pack('<Q', fnv(stage['content'])), value)
    return value


class ReplayReference:
    def __init__(self, ancestor, every, schedule_hash, source_hash, document_lengths):
        meta, extra = policy_checkpoint(ancestor)
        assert meta[17] == 5 and extra[1] == 3 and not any(extra[9:14])
        self.meta, self.header = list(meta), list(extra[:16])
        self.random, self.speech = Random64(extra[4]), Random64(meta[23])
        self.groups = []
        position = 17 + 5 * extra[16]
        for i in range(extra[16]):
            end, seen, stored, updates, pairs = extra[17 + 5*i:22 + 5*i]
            items = [tuple(extra[j:j+3]) for j in range(position, position + 3*stored, 3)]
            self.groups.append(Group(end, seen, updates, pairs, items))
            position += 3 * stored
        assert position == len(extra)
        self.first = self.groups[-1].end
        self.lengths = document_lengths
        assert self.first < len(self.lengths) and min(self.lengths) >= 2
        self.header[2], self.header[14] = every, schedule_hash
        self.header[15] += 1
        self.meta[12] = source_hash
        self.meta[19], self.meta[20], self.meta[25] = self.first, 0, 1
        new_count = len(self.groups) + 1
        for index, group in enumerate(self.groups):
            quota = extra[3] // new_count + (index < extra[3] % new_count)
            while len(group.items) > quota:
                slot = self.random.below(len(group.items))
                group.items[slot] = group.items[-1]
                group.items.pop()
        self.groups.append(Group(len(document_lengths), 0, 0, 0, []))

    def run_until(self, end, on_replay=None):
        while self.meta[24] < end:
            document, offset = self.meta[19:21]
            size = min(self.meta[6], self.lengths[document] - 1 - offset)
            observed = (document, offset, size)
            self.meta[25] = 0
            self.meta[7] += 1
            self.meta[24] += 1
            self.meta[22] += size
            self.meta[20] += size
            if self.meta[20] == self.lengths[document] - 1:
                self.meta[20], self.meta[25] = 0, 1
                self.meta[19] += 1
                if self.meta[19] == len(self.lengths):
                    self.meta[19] = self.first
                    self.meta[21] += 1
            if self.meta[24] % self.header[2] == 0:
                available = [g for g in self.groups if g.items]
                if available:
                    group = available[self.random.below(len(available))] if len(available) > 1 else available[0]
                    episode = group.items[self.random.below(len(group.items))]
                    if on_replay is not None:
                        on_replay(self.meta[24], episode)
                    self.meta[7] += 1
                    self.header[6] += 1
                    self.header[7] += episode[2]
                    group.updates += 1
                    group.pairs += episode[2]
            self.header[5] += 1
            index = next(i for i, g in enumerate(self.groups) if document < g.end)
            group = self.groups[index]
            group.seen += 1
            quota = self.header[3] // len(self.groups) + (index < self.header[3] % len(self.groups))
            if len(group.items) < quota:
                group.items.append(observed)
            else:
                slot = self.random.below(group.seen)
                if slot < quota:
                    group.items[slot] = observed
            if self.meta[26] and self.meta[24] % self.meta[26] == 0:
                for _ in range(self.meta[27]):
                    self.speech.next()
                self.meta[30] += self.meta[27]

    def matches(self, path):
        actual, extra = policy_checkpoint(path)
        self.header[4], self.meta[23] = self.random.state, self.speech.state
        encoded = self.header + [len(self.groups)]
        for group in self.groups:
            encoded.extend((group.end, group.seen, len(group.items), group.updates, group.pairs))
        for group in self.groups:
            for item in group.items:
                encoded.extend(item)
        self.meta[31] = len(encoded)
        assert tuple(encoded) == extra, 'Replay policy, RNG, descriptors or counters differ'
        # Only the native learned-payload checksum is outside this host policy reference.
        assert all(a == b for i, (a, b) in enumerate(zip(actual, self.meta)) if i != 15), [
            (i, a, b) for i, (a, b) in enumerate(zip(actual, self.meta)) if i != 15 and a != b]
        return dict(source_cursor_exact=True, source_pairs_exact=True, replay_rng_exact=True,
                    replay_descriptors_exact=True, speech_rng_and_count_exact=True,
                    replay_slots=sum(len(g.items) for g in self.groups))
