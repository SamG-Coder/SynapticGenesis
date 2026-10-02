// A bounded exposure schedule over cumulative, append-only source corpora.
// Stage changes reuse the same mutable model, replay reservoir and SI history.
#pragma once
struct CurriculumStage {
    uint64_t end_update = 0;
    fs::path corpus;
    float rate_scale = 1;
    uint64_t corpus_hash = 0;
};
struct LiveCurriculum {
    std::vector<CurriculumStage> stages;
    uint64_t hash = 0;
    explicit LiveCurriculum(const fs::path &path) {
        std::ifstream f(path, std::ios::binary);
        require(bool(f), "Cannot open curriculum schedule");
        require(fs::file_size(path) <= 1024 * 1024, "Curriculum schedule is too large");
        std::string bytes((std::istreambuf_iterator<char>(f)), {});
        require(bytes.size() <= 1024 * 1024, "Curriculum schedule is too large");
        hash = hash_bytes(bytes.data(), bytes.size());
        require(hash != 0, "Invalid curriculum hash");
        std::istringstream input(bytes);
        std::string magic, line;
        std::getline(input, magic);
        if (!magic.empty() && magic.back() == '\r')
            magic.pop_back();
        require(magic == "SGCURRICULUM1", "Unsupported curriculum schedule");
        while (std::getline(input, line)) {
            if (line.find_first_not_of(" \t\r") == std::string::npos)
                continue;
            CurriculumStage stage;
            std::string filename, extra;
            std::istringstream row(line);
            row >> stage.end_update >> std::quoted(filename) >> stage.rate_scale;
            require(bool(row) && !(row >> extra) && !filename.empty() &&
                        stage.end_update > (stages.empty() ? 0 : stages.back().end_update) &&
                        stage.end_update <= 100000000 && std::isfinite(stage.rate_scale) &&
                        stage.rate_scale > 0 && stage.rate_scale <= 10 && stages.size() < 4096,
                    "Invalid curriculum row: expected increasing end-update, quoted path and LR scale");
            stage.corpus = (path.parent_path() / fs::path(filename)).lexically_normal();
            stages.push_back(stage);
        }
        require(!stages.empty(), "Curriculum has no stages");
        // Bind every scheduled edition, including future stages, to the policy.
        // Paths alone would allow a future file to change unnoticed on resume.
        std::unique_ptr<LiveCorpus> previous;
        for (auto &stage : stages) {
            auto current = std::make_unique<LiveCorpus>(stage.corpus);
            if (previous)
                validate_append(*previous, *current);
            stage.corpus_hash = current->hash;
            hash = hash_bytes(&stage.corpus_hash, sizeof(stage.corpus_hash), hash);
            previous = std::move(current);
        }
        require(hash != 0, "Invalid combined curriculum hash");
    }
    static void validate_append(const LiveCorpus &old, const LiveCorpus &next) {
        require(next.bytes.size() > old.bytes.size() && next.bytes[old.bytes.size()] == 30 &&
                    std::equal(old.bytes.begin(), old.bytes.end(), next.bytes.begin()) &&
                    next.docs.size() > old.docs.size() &&
                    std::equal(old.docs.begin(), old.docs.end(), next.docs.begin()),
                "Curriculum stages must append complete documents without changing earlier bytes");
    }
    void rate(State &s, float base) const {
        require(std::isfinite(base) && base > 0 && base <= .1f, "Invalid curriculum base learning rate");
        for (const auto &stage : stages)
            require(base * stage.rate_scale <= .1f, "Curriculum learning rate exceeds limit");
        s.hp[7] = base;
        s.hp[0] = base * stages.at(size_t(s.extra[15])).rate_scale;
    }
    void initialize(State &s) const {
        require(s.extra.size() >= 16 && (s.meta[17] == 2 || s.meta[17] == 3),
                "Curriculum needs an initialized live policy");
        s.meta[17] = 4;
        s.extra[14] = hash;
        s.extra[15] = 0;
        rate(s, s.hp[0]);
    }
    void validate(const State &s) const {
        validate_curriculum_state(s);
        require(s.meta[17] == 4 && s.extra[14] == hash && s.extra[15] < stages.size(),
                "Resume requires the identical curriculum schedule and all source editions");
        size_t index = size_t(s.extra[15]);
        uint64_t start = index ? stages[index - 1].end_update : 0;
        require(s.meta[24] >= start && s.meta[24] <= stages[index].end_update &&
                    s.meta[12] == stages[index].corpus_hash && s.hp[0] == s.hp[7] * stages[index].rate_scale,
                "Curriculum stage, source, learning rate or update counter is inconsistent");
    }
    LiveCorpus corpus(const State &s) const {
        validate(s);
        LiveCorpus result(stages[size_t(s.extra[15])].corpus);
        require(result.hash == stages[size_t(s.extra[15])].corpus_hash,
                "Curriculum source changed during this invocation");
        return result;
    }
    bool advance(LiveEngine &engine, LiveCorpus &data, State &s, std::ostream *events = nullptr) const {
        validate(s);
        size_t index = size_t(s.extra[15]);
        if (s.meta[24] < stages[index].end_update || index + 1 == stages.size())
            return false;
        LiveCorpus next(stages[index + 1].corpus);
        require(next.hash == stages[index + 1].corpus_hash,
                "Curriculum source changed during this invocation");
        validate_append(data, next);
        data.validate(s);
        ReplayMemory(s).validate(data, engine.root.T);
        // Close a partially observed document's SI trajectory at the explicit
        // stage boundary. A completed document already consolidated in live_tick.
        if (engine.root.synapses->active() && !s.meta[25]) {
            engine.root.synapses->consolidate(engine.root.w);
            s.extra[12] = engine.root.synapses->boundaries;
        }
        uint64_t old_hash = data.hash, old_document = s.meta[19], old_offset = s.meta[20];
        s.meta[19] = data.docs.size(); // Begin with newly introduced content.
        s.meta[20] = 0;
        s.meta[25] = 1; // Usual document boundary reset on the next tick.
        s.meta[12] = next.hash;
        ++s.extra[15];
        rate(s, s.hp[7]);
        data = std::move(next);
        validate(s);
        data.validate(s);
        ReplayMemory(s).validate(data, engine.root.T);
        if (events)
            *events << std::setprecision(10)
                    << "{\"event\":\"curriculum_transition\",\"online_update\":" << s.meta[24]
                    << ",\"stage\":" << s.extra[15] + 1 << ",\"previous_corpus_hash\":\"" << old_hash
                    << "\",\"corpus_hash\":\"" << data.hash << "\",\"previous_document\":" << old_document
                    << ",\"previous_offset\":" << old_offset << ",\"new_document\":" << s.meta[19]
                    << ",\"replay_windows_preserved\":" << ReplayMemory(s).count()
                    << ",\"learning_rate\":" << s.hp[0]
                    << ",\"consolidation_events\":" << engine.root.synapses->boundaries << "}\n";
        return true;
    }
};
