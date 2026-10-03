// Immutable selected-source and frozen-model package. Neural math is shared.
#pragma once
namespace teachers {
uint64_t file_hash(const fs::path &path) {
    std::ifstream file(path, std::ios::binary);
    if (!file)
        throw std::runtime_error("Cannot open teacher bundle file: " + path.string());
    std::array<char, 65536> buffer{};
    uint64_t result = 14695981039346656037ull;
    while (file) {
        file.read(buffer.data(), buffer.size());
        result = hash_bytes(buffer.data(), size_t(file.gcount()), result);
    }
    if (!file.eof())
        throw std::runtime_error("Cannot read teacher bundle file");
    return result;
}
bool valid_origin(const std::string &id) {
    return !id.empty() && id.size() <= 64 && std::all_of(id.begin(), id.end(), [](char c) {
        return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
               (c >= '0' && c <= '9') || c == '-' || c == '_';
    });
}
struct Origin {
    fs::path checkpoint;
    std::string member;
    uint64_t member_hash = 0;
};
struct Bundle {
    fs::path directory;
    LiveCorpus source;
    TeachingState identity;
    std::vector<std::string> members;
    std::vector<std::vector<float>> weights;
    explicit Bundle(const fs::path &path) : directory(fs::absolute(path)), source(directory / "source.dat") {
        fs::path manifest = directory / "teachers.sg";
        require(fs::file_size(manifest) <= 16384, "Teacher manifest is too large");
        std::ifstream in(manifest);
        std::string magic;
        float temperature, strength, mixture;
        uint64_t count, scope, source_hash, documents;
        require(bool(in >> magic >> count >> temperature >> strength >> mixture >> scope >> source_hash >> documents) &&
                    magic == "SGTEACH1" && count >= 1 && count <= 2 && scope <= 1,
                "Invalid teacher manifest header");
        auto &w = identity.words;
        w[0] = TeachingState::magic;
        w[1] = 1;
        w[2] = 1;
        w[3] = count;
        w[4] = file_hash(manifest);
        w[5] = source_hash;
        w[6] = documents;
        w[11] = float_word(temperature);
        w[12] = float_word(strength);
        w[13] = float_word(mixture);
        w[14] = scope;
        require(std::isfinite(temperature) && temperature >= .25f && temperature <= 16 &&
                    std::isfinite(strength) && strength >= 0 && strength <= 100 &&
                    std::isfinite(mixture) && mixture >= 0 && mixture <= 1 &&
                    source.hash == source_hash && source.docs.size() == documents &&
                    source.docs.front().first == 0 && source.docs.back().second == source.bytes.size(),
                "Teacher source identity or objective settings changed");
        for (size_t i = 0; i < count; ++i) {
            size_t at = identity.entry(i);
            std::string member;
            require(bool(in >> std::quoted(member)), "Missing teacher origin");
            for (size_t j = 0; j < 8; ++j)
                require(bool(in >> w[at + j]), "Missing teacher identity field");
            require(scope ? (valid_origin(member) && w[at + 7]) : (member.empty() && !w[at + 7]),
                    "Invalid teacher population origin");
            auto checkpoint = directory / ("teacher-" + std::to_string(i) + ".ckpt");
            require(file_hash(checkpoint) == w[at], "Teacher snapshot file changed");
            auto stored = read_checkpoint(checkpoint);
            require(!stored.state.membrane.present(),
                    "Experimental membrane policy is not admitted as a teacher");
            auto &m = stored.state.meta;
            require(m[15] == w[at + 1] && m[14] == w[at + 2] && m[2] == w[at + 3] &&
                        m[3] == w[at + 4] && m[4] == w[at + 5] && m[1] == w[at + 6],
                    "Teacher checkpoint identity differs from manifest");
            identity.config(i); // Bounded integer conversion, also checked by the native reader.
            members.push_back(member);
            weights.push_back(std::move(stored.weights));
        }
        in >> std::ws;
        require(in.eof(), "Unexpected teacher manifest fields");
        if (count == 2)
            require(w[16] != w[24] && (!scope || members[0] != members[1]),
                    "Two teachers must be distinct snapshots and population members");
    }
    void selected_prefix(const LiveCorpus &selected) const {
        require(source.bytes.size() <= selected.bytes.size() && source.docs.size() <= selected.docs.size() &&
                    std::equal(source.bytes.begin(), source.bytes.end(), selected.bytes.begin()),
                "Teacher source is not an exact prefix of the selected curriculum");
        for (size_t i = 0; i < source.docs.size(); ++i)
            require(source.docs[i] == selected.docs[i], "Teacher source changes document boundaries");
    }
    void bind(State &s) const {
        require(has_grouped_replay(s), "Teacher replay requires a curriculum with stage replay");
        if (live_version(s) == 6) {
            for (size_t i : std::array<size_t, 10>{0, 1, 3, 4, 5, 6, 11, 12, 13, 14})
                require(s.teaching.words[i] == identity.words[i], "Resume requires the identical teacher bundle");
            for (size_t i = 16; i < TeachingState::count_words; ++i)
                require(s.teaching.words[i] == identity.words[i], "Resume teacher identity changed");
        } else {
            set_live_version(s, 6);
            s.teaching = identity;
            s.teaching.words[7] = s.meta[24];
            s.teaching.words[8] = s.extra[6];
            s.teaching.words[15] = s.extra[7];
        }
        validate_teaching_state(s);
    }
};

// Population callers implement ownership/lifespan checks while holding their lock.
struct Authority {
    virtual void validate(const Bundle &bundle, bool attaching, bool active) const = 0;
    virtual ~Authority() = default;
};

void create_bundle(const fs::path &out, const fs::path &source_path, const std::vector<Origin> &origins,
                   float temperature, float strength, float mixture, bool population) {
    require(!fs::exists(out), "Teacher bundle output already exists");
    require(origins.size() >= 1 && origins.size() <= 2 && std::isfinite(temperature) &&
                temperature >= .25f && temperature <= 16 && std::isfinite(strength) && strength >= 0 &&
                strength <= 100 && std::isfinite(mixture) && mixture >= 0 && mixture <= 1,
            "Invalid teacher bundle settings");
    LiveCorpus source(source_path);
    require(source.docs.front().first == 0 && source.docs.back().second == source.bytes.size(),
            "Teacher source must contain complete documents");
    std::vector<std::array<uint64_t, 8>> records;
    for (const auto &origin : origins) {
        require(population ? (valid_origin(origin.member) && origin.member_hash)
                           : (origin.member.empty() && !origin.member_hash), "Invalid teacher origin");
        auto stored = read_checkpoint(origin.checkpoint); // Full validation without a GPU allocation.
        require(!stored.state.membrane.present(),
                "Experimental membrane policy is not admitted as a teacher");
        auto &m = stored.state.meta;
        records.push_back({file_hash(origin.checkpoint), m[15], m[14], m[2], m[3], m[4], m[1], origin.member_hash});
    }
    require(records.size() == 1 || (records[0][0] != records[1][0] &&
                                   (!population || origins[0].member != origins[1].member)),
            "Two teachers must be distinct snapshots and population members");
    fs::create_directories(out);
    dump(out / "source.dat", source.bytes);
    for (size_t i = 0; i < origins.size(); ++i) {
        auto copied = out / ("teacher-" + std::to_string(i) + ".ckpt");
        fs::copy_file(origins[i].checkpoint, copied);
        require(file_hash(copied) == records[i][0], "Teacher changed during snapshot copying");
    }
    // Publish the manifest last; an incomplete copy is not a readable bundle.
    std::ofstream file(out / "teachers.sg", std::ios::binary);
    file << "SGTEACH1\n" << origins.size() << ' ' << std::setprecision(std::numeric_limits<float>::max_digits10)
         << temperature << ' ' << strength << ' ' << mixture << ' ' << int(population) << ' '
         << source.hash << ' ' << source.docs.size() << '\n';
    for (size_t i = 0; i < origins.size(); ++i) {
        file << std::quoted(origins[i].member);
        for (auto value : records[i])
            file << ' ' << value;
        file << '\n';
    }
    file.close();
    require(bool(file), "Cannot write teacher manifest");
    Bundle verified(out);
    std::cout << "teacher_bundle=" << out.string() << " teachers=" << origins.size()
              << " eligible_documents=" << source.docs.size() << " identity=" << verified.identity.words[4] << '\n';
}
} // namespace teachers
