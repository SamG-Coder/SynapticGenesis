// Reservoir memory holds only descriptors into the selected, immutable learning
// corpus. No generated speech or held-out data enters this memory. Replaying a
// descriptor reconstructs a reset-state training window, not an old GPU state.
#pragma once
struct Episode {
    uint64_t document, offset, length;
};
struct ReplayMemory {
    State &s;
    static constexpr uint64_t magic = 0x3159414c504552ull;
    static constexpr size_t header_words = 16;
    explicit ReplayMemory(State &state) : s(state) {}
    bool extended() const {
        return s.meta[17] == 2 || s.meta[17] == 3;
    }
    uint64_t mode() const {
        return extended() ? s.extra[1] : 0;
    }
    uint64_t every() const {
        return extended() ? s.extra[2] : 0;
    }
    uint64_t count() const {
        return extended() ? (s.extra.size() - header_words) / 3 : 0;
    }
    uint64_t updates() const {
        return extended() ? s.extra[6] : 0;
    }
    uint64_t pairs() const {
        return extended() ? s.extra[7] : 0;
    }
    bool graph() const {
        return extended() && s.extra[8];
    }
    float core_scale() const {
        return extended() ? s.hp[6] : 1.f;
    }
    static void configure(State &s, const Args &args) {
        s.meta[17] = 2;
        s.extra.assign(header_words, 0);
        auto mode = args.get("replay", "none");
        if (mode != "none" && mode != "reservoir" && mode != "recent")
            throw std::runtime_error("--replay must be none, reservoir or recent");
        int capacity = args.num("replay-capacity", mode == "reservoir" ? 1024 : 0);
        int every = args.num("replay-every", mode == "none" ? 0 : 4);
        int seed = args.num("replay-seed", 1777);
        if (capacity < 0 || capacity > 65536 || (mode == "reservoir" && !capacity) ||
            (mode != "reservoir" && capacity) || seed <= 0 || every < 0 ||
            (mode == "none" ? every != 0 : every < 1))
            throw std::runtime_error("Invalid replay capacity, interval or seed");
        s.extra[0] = magic;
        s.extra[1] = mode == "reservoir" ? 1 : (mode == "recent" ? 2 : 0);
        s.extra[2] = every;
        s.extra[3] = capacity;
        s.extra[4] = seed;
        s.extra[8] = !args.get("graph").empty();
        s.hp[6] = args.real("core-scale", 1.f);
        if (s.hp[6] < 0 || s.hp[6] > 1)
            throw std::runtime_error("--core-scale must be in [0,1]");
    }
    Episode at(uint64_t index) const {
        size_t start = header_words + size_t(index) * 3;
        return {s.extra.at(start), s.extra.at(start + 1), s.extra.at(start + 2)};
    }
    static void validate_episode(const Episode &episode, const LiveCorpus &data, int chunk) {
        if (episode.document >= data.docs.size())
            throw std::runtime_error("Replay document outside corpus");
        auto doc = data.docs[size_t(episode.document)];
        uint64_t size = doc.second - doc.first;
        if (!episode.length || episode.length > uint64_t(chunk) || episode.offset >= size - 1 ||
            episode.length > size - 1 - episode.offset)
            throw std::runtime_error("Replay window crosses a document boundary");
    }
    void validate(const LiveCorpus &data, int chunk) const {
        if (!extended()) {
            if (s.meta[17] != 1 || !s.extra.empty())
                throw std::runtime_error("Unsupported live policy version");
            return;
        }
        if (s.extra.size() < header_words || (s.extra.size() - header_words) % 3 || s.extra[0] != magic)
            throw std::runtime_error("Invalid live replay layout");
        if (mode() > 2 || s.extra[3] > 65536 || !s.extra[4] || count() > s.extra[3] || count() > s.extra[5] ||
            s.extra[5] != s.meta[24] || s.extra[8] > 1 || updates() > s.meta[24] || updates() > s.meta[7] ||
            (mode() == 0 ? every() != 0 : (every() < 1 || every() > 1000000000ull)) ||
            (mode() != 1 ? (s.extra[3] != 0 || count() != 0) : s.extra[3] == 0) || !std::isfinite(s.hp[6]) ||
            s.hp[6] < 0 || s.hp[6] > 1)
            throw std::runtime_error("Invalid live replay settings/counters");
        size_t reserved_start = s.meta[17] == 3 ? 14 : 9;
        if (s.meta[17] == 3 &&
            (s.extra[9] != 1 || s.extra[12] > s.meta[24] || s.extra[13] != s.meta[24] + updates()))
            throw std::runtime_error("Invalid consolidation policy/counters");
        for (size_t i = reserved_start; i < header_words; ++i)
            if (s.extra[i])
                throw std::runtime_error("Unknown live policy field");
        for (uint64_t i = 0; i < count(); ++i)
            validate_episode(at(i), data, chunk);
    }
    uint64_t random(uint64_t bound) {
        // Rejection sampling avoids modulo bias in reservoir replacement.
        uint64_t minimum = (~bound + 1) % bound, value;
        do {
            value = rnd(s.extra[4]);
        } while (value < minimum);
        return value % bound;
    }
    void remember(const Episode &episode) {
        if (!extended())
            return;
        uint64_t seen = ++s.extra[5];
        if (mode() != 1)
            return;
        uint64_t slot = count();
        if (slot < s.extra[3]) {
            s.extra.push_back(episode.document);
            s.extra.push_back(episode.offset);
            s.extra.push_back(episode.length);
        } else {
            slot = random(seen);
            if (slot < s.extra[3]) {
                size_t start = header_words + size_t(slot) * 3;
                s.extra[start] = episode.document;
                s.extra[start + 1] = episode.offset;
                s.extra[start + 2] = episode.length;
            }
        }
    }
    bool due() const {
        return every() && s.meta[24] % every() == 0 && (mode() == 2 || count() > 0);
    }
    Episode choose(const Episode &current) {
        return mode() == 2 ? current : at(random(count()));
    }
    void completed(size_t n) {
        ++s.extra[6];
        s.extra[7] += n;
    }
};
