// Host-side bounded replay policy. No model computation or source bytes live here.
// Live v5: common 16-word policy, group count, five words per source stage
// (exclusive document end, seen windows, stored windows, replay updates, pairs),
// then grouped three-word episode descriptors. All words are checksummed.
#pragma once
struct Episode {
    uint64_t document, offset, length;
};
struct StageReplayView {
    const State &s;
    static constexpr size_t prefix_words = 17, group_words = 5;
    explicit StageReplayView(const State &state) : s(state) {}
    uint64_t groups() const {
        return s.extra.at(16);
    }
    size_t field(size_t group, size_t column) const {
        return prefix_words + group_words * group + column;
    }
    uint64_t value(size_t group, size_t column) const {
        return s.extra.at(field(group, column));
    }
    size_t begin() const {
        return prefix_words + group_words * size_t(groups());
    }
    uint64_t count() const {
        return (s.extra.size() - begin()) / 3;
    }
    uint64_t quota(size_t group, uint64_t n) const {
        return s.extra[3] / n + (group < s.extra[3] % n);
    }
    size_t group_for(uint64_t document) const {
        for (size_t g = 0; g < groups(); ++g)
            if (document < value(g, 0))
                return g;
        throw std::runtime_error("Replay document outside introduced stages");
    }
    size_t offset(size_t group) const {
        size_t at = begin();
        for (size_t g = 0; g < group; ++g)
            at += size_t(value(g, 2)) * 3;
        return at;
    }
    Episode at(uint64_t index) const {
        size_t at = begin() + size_t(index) * 3;
        return {s.extra.at(at), s.extra.at(at + 1), s.extra.at(at + 2)};
    }
    void validate() const {
        if (s.meta[17] != 5 || s.extra.size() < prefix_words || s.extra[1] != 3 || groups() < 1 ||
            groups() > 4096 || groups() > s.extra[3] || s.extra[3] > 65536 || s.extra.size() < begin() ||
            (s.extra.size() - begin()) % 3)
            throw std::runtime_error("Invalid stage replay layout");
        uint64_t end = 0, seen = 0, stored = 0, updates = 0, pairs = 0;
        for (size_t g = 0; g < groups(); ++g) {
            if (value(g, 0) <= end || value(g, 0) > 1000000000 || value(g, 1) > s.extra[5] ||
                value(g, 2) != std::min(value(g, 1), quota(g, groups())) || value(g, 3) > s.extra[6] ||
                value(g, 4) > s.extra[7])
                throw std::runtime_error("Invalid stage replay group/counters");
            for (size_t j = 0; j < value(g, 2); ++j) {
                auto e = at(stored + j);
                if (e.document < end || e.document >= value(g, 0))
                    throw std::runtime_error("Replay window assigned to wrong source stage");
            }
            end = value(g, 0);
            seen += value(g, 1);
            stored += value(g, 2);
            updates += value(g, 3);
            pairs += value(g, 4);
        }
        if (seen != s.extra[5] || stored != count() || updates != s.extra[6] || pairs != s.extra[7] ||
            groups() != s.extra[15] + 1)
            throw std::runtime_error("Stage replay totals differ from live policy");
    }
    void report(std::ostream &out) const {
        out << ",\"replay_groups\":[";
        for (size_t g = 0; g < groups(); ++g) {
            if (g)
                out << ',';
            out << "{\"stage\":" << g + 1 << ",\"document_end\":" << value(g, 0)
                << ",\"seen_windows\":" << value(g, 1) << ",\"stored_windows\":" << value(g, 2)
                << ",\"quota\":" << quota(g, groups()) << ",\"replay_updates\":" << value(g, 3)
                << ",\"replay_pairs\":" << value(g, 4) << '}';
        }
        out << ']';
    }
};
struct StageReplay : StageReplayView {
    State &s;
    explicit StageReplay(State &state) : StageReplayView(state), s(state) {}
    uint64_t random(uint64_t bound) {
        if (!bound)
            throw std::runtime_error("Empty stage replay selection");
        uint64_t minimum = (~bound + 1) % bound, v;
        do {
            v = rnd(s.extra[4]);
        } while (v < minimum);
        return v % bound;
    }
    void add_group(uint64_t document_end) {
        uint64_t n = groups() + 1;
        if (n > 4096 || n > s.extra[3] || !document_end || (n > 1 && document_end <= value(size_t(n - 2), 0)))
            throw std::runtime_error("Stage replay needs one slot per introduced source stage");
        std::vector<std::array<uint64_t, group_words>> metadata;
        std::vector<uint64_t> episodes;
        for (size_t g = 0; g + 1 < n; ++g) {
            std::array<uint64_t, group_words> record;
            for (size_t j = 0; j < group_words; ++j)
                record[j] = value(g, j);
            size_t start = offset(g);
            std::vector<Episode> pool;
            for (size_t j = 0; j < record[2]; ++j)
                pool.push_back(
                    {s.extra[start + 3 * j], s.extra[start + 3 * j + 1], s.extra[start + 3 * j + 2]});
            // Uniformly shrink each existing reservoir. Its lifetime seen count
            // remains intact if that source reappears in a later all-scope stage.
            while (pool.size() > quota(g, n)) {
                pool[size_t(random(pool.size()))] = pool.back();
                pool.pop_back();
            }
            record[2] = pool.size();
            metadata.push_back(record);
            for (const auto &e : pool)
                episodes.insert(episodes.end(), {e.document, e.offset, e.length});
        }
        metadata.push_back({document_end, 0, 0, 0, 0});
        // Copy after shrinking so the advanced RNG is preserved.
        std::vector<uint64_t> next(s.extra.begin(), s.extra.begin() + 16);
        next.push_back(n);
        for (const auto &record : metadata)
            next.insert(next.end(), record.begin(), record.end());
        next.insert(next.end(), episodes.begin(), episodes.end());
        s.extra = std::move(next);
    }
    void initialize(uint64_t document_end) {
        if (s.extra.size() != 16 || s.extra[5] || s.extra[6] || s.extra[7])
            throw std::runtime_error("Stage replay must be selected before observing the stream");
        s.meta[17] = 5;
        s.extra.push_back(0);
        add_group(document_end);
    }
    void adopt_first_stage(uint64_t document_end) {
        // All previously observed windows belong to this one source group.
        // Later conversion cannot reconstruct lost per-group exposure counts.
        if (s.meta[17] != 4 || s.extra.size() < 16 || s.extra[1] != 1 || s.extra[15] != 0 ||
            (s.extra.size() - 16) % 3)
            throw std::runtime_error("Stage replay conversion requires a first-stage curriculum reservoir");
        State candidate = s;
        std::vector<uint64_t> next(s.extra.begin(), s.extra.begin() + 16);
        next[1] = 3;
        next.insert(next.end(),
                    {1, document_end, s.extra[5], (s.extra.size() - 16) / 3, s.extra[6], s.extra[7]});
        next.insert(next.end(), s.extra.begin() + 16, s.extra.end());
        candidate.meta[17] = 5;
        candidate.extra = std::move(next);
        StageReplayView(candidate).validate();
        s = std::move(candidate);
    }
    void remember(const Episode &e) {
        size_t g = group_for(e.document), start = offset(g);
        uint64_t seen = ++s.extra[field(g, 1)], stored = value(g, 2), cap = quota(g, groups());
        if (stored < cap) {
            s.extra.insert(s.extra.begin() + start + 3 * size_t(stored), {e.document, e.offset, e.length});
            ++s.extra[field(g, 2)];
        } else {
            uint64_t slot = random(seen);
            if (slot < cap) {
                start += 3 * size_t(slot);
                s.extra[start] = e.document;
                s.extra[start + 1] = e.offset;
                s.extra[start + 2] = e.length;
            }
        }
    }
    Episode choose() {
        size_t nonempty = 0;
        for (size_t g = 0; g < groups(); ++g)
            nonempty += value(g, 2) != 0;
        size_t selected = nonempty == 1 ? 0 : size_t(random(nonempty));
        for (size_t g = 0; g < groups(); ++g) {
            if (!value(g, 2))
                continue;
            if (selected--)
                continue;
            size_t start = offset(g) + 3 * size_t(random(value(g, 2)));
            return {s.extra[start], s.extra[start + 1], s.extra[start + 2]};
        }
        throw std::runtime_error("Stage replay has no observed windows");
    }
    void completed(const Episode &e) {
        size_t g = group_for(e.document);
        ++s.extra[field(g, 3)];
        s.extra[field(g, 4)] += e.length;
    }
};
