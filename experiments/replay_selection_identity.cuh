// Experimental policy is authenticated beside each ordinary native checkpoint.
#pragma once
namespace replay_priority {
using SelectionIdentity = std::array<uint64_t, 11>;
SelectionIdentity identity(const fs::path &base, const LiveCurriculum &curriculum,
                           const State &ancestor, uint64_t mode, uint64_t seed) {
    return {0x3150524c45534753ull, 1, file_identity(base), curriculum.hash, mode, seed,
            ancestor.meta[24], ancestor.meta[7], 1, 0, 0}; // Graph scoring version 1.
}
void save_identity(const fs::path &checkpoint, SelectionIdentity id) {
    id[9] = file_identity(checkpoint);
    id[10] = hash_bytes(id.data(), 10 * sizeof(uint64_t));
    std::ofstream out(checkpoint.string() + ".sgpriority", std::ios::binary);
    write_raw(out, id.data(), id.size());
    out.close();
    require(bool(out), "Replay policy sidecar flush failed");
}
void validate_identity(const fs::path &checkpoint, const SelectionIdentity &expected) {
    fs::path path = checkpoint.string() + ".sgpriority";
    require(fs::exists(path) && fs::file_size(path) == sizeof(SelectionIdentity), "Missing or malformed replay policy sidecar");
    SelectionIdentity actual;
    std::ifstream in(path, std::ios::binary);
    read_raw(in, actual.data(), actual.size());
    require(actual[10] == hash_bytes(actual.data(), 10 * sizeof(uint64_t)) &&
                actual[9] == file_identity(checkpoint), "Replay policy sidecar checksum mismatch");
    require(std::equal(actual.begin(), actual.begin() + 9, expected.begin()),
            "Replay resume requires the original ancestor, sources and selection policy");
}
} // namespace replay_priority
