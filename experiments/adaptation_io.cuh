// Diagnostic source admission and restart identity. No model equations.
#pragma once
namespace adaptation {
std::string bytes(const fs::path &path, uint64_t maximum = 512ull * 1024 * 1024) {
    require(fs::is_regular_file(path) && fs::file_size(path) <= maximum, "Missing or oversized probe input");
    std::ifstream file(path, std::ios::binary);
    require(bool(file), "Cannot open probe input");
    std::string result((std::istreambuf_iterator<char>(file)), {});
    require(result.size() == fs::file_size(path), "Incomplete probe input read");
    return result;
}
uint64_t identity(const fs::path &path) {
    auto data = bytes(path);
    return hash_bytes(data.data(), data.size());
}
struct Source {
    std::vector<std::string> documents;
    uint64_t hash;
    explicit Source(const fs::path &path) {
        auto raw = bytes(path, 64ull * 1024 * 1024);
        hash = hash_bytes(raw.data(), raw.size());
        size_t begin = 0;
        for (;;) {
            auto end = raw.find(char(30), begin);
            if (end == std::string::npos)
                end = raw.size();
            require(end - begin >= 2, "Empty or one-byte probe document");
            documents.push_back(raw.substr(begin, end - begin));
            if (end == raw.size())
                break;
            begin = end + 1;
        }
    }
};
struct Window {
    size_t document, offset;
};
struct Schedule {
    std::vector<Window> windows;
    uint64_t hash;
    Schedule(const fs::path &path, const Source &source) {
        auto raw = bytes(path, 16ull * 1024 * 1024);
        hash = hash_bytes(raw.data(), raw.size());
        std::istringstream input(raw);
        std::string magic;
        uint64_t context, count, source_hash;
        input >> magic >> context >> count >> source_hash;
        require(bool(input) && magic == "SGADAPT1" && context == 128 && count > 0 && count <= 65536 &&
                    source_hash == source.hash, "Invalid adaptation schedule header/source identity");
        for (uint64_t i = 0; i < count; ++i) {
            uint64_t document, offset, expected;
            input >> document >> offset >> expected;
            require(bool(input) && document < source.documents.size(), "Invalid adaptation source index");
            const auto &text = source.documents[size_t(document)];
            require(offset <= text.size() && text.size() - size_t(offset) >= 129,
                    "Adaptation window crosses its source document");
            require(hash_bytes(text.data() + offset, 129) == expected, "Adaptation window bytes changed");
            windows.push_back({size_t(document), size_t(offset)});
        }
        require(!(input >> magic), "Unexpected adaptation schedule content");
    }
};
// Sidecars authenticate the entire ordinary checkpoint, including its header.
// [0] magic, [1] format, [2] completed probe updates, [3] schedule hash,
// [4] mode, [5] original checkpoint hash, [6] seed, [7] initial Adam step,
// [8] initial weight hash, [9] initial moment hash, [10] float LR bits,
// [11] saved checkpoint hash, [12] train hash, [13] schedule count,
// [14] development hash, [15] checksum of words 0..14.
using Identity = std::array<uint64_t, 16>;
constexpr uint64_t magic = 0x3154504144475355ull;
void save_identity(const fs::path &checkpoint, Identity value) {
    value[11] = identity(checkpoint);
    value[15] = hash_bytes(value.data(), 15 * sizeof(uint64_t));
    auto sidecar = checkpoint;
    sidecar += ".sgadapt";
    std::ofstream file(sidecar, std::ios::binary);
    write_raw(file, value.data(), value.size());
    file.close();
    require(bool(file), "Adaptation sidecar flush failed");
}
Identity restore_identity(const fs::path &checkpoint, const Identity &expected) {
    auto sidecar = checkpoint;
    sidecar += ".sgadapt";
    auto raw = bytes(sidecar, sizeof(Identity));
    require(raw.size() == sizeof(Identity), "Invalid adaptation sidecar length");
    Identity result{};
    std::memcpy(result.data(), raw.data(), raw.size());
    require(result[0] == magic && result[1] == 1 &&
                result[15] == hash_bytes(result.data(), 15 * sizeof(uint64_t)) &&
                result[11] == identity(checkpoint), "Adaptation restart identity mismatch");
    for (size_t i = 3; i <= 14; ++i)
        if (i != 11)
            require(result[i] == expected[i], "Adaptation restart policy/source/ancestor changed");
    require(result[2] <= result[13], "Invalid completed adaptation count");
    return result;
}
uint64_t tensor_hash(const Buf &buffer, uint64_t seed = 14695981039346656037ull) {
    auto data = buffer.host();
    return hash_bytes(data.data(), data.size() * sizeof(float), seed);
}
uint64_t learning_hash(const Model &model) {
    return tensor_hash(model.v, tensor_hash(model.m, tensor_hash(model.w)));
}
} // namespace adaptation
