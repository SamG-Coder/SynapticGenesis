// Durable teacher identity/counters; no model computation or file paths.
#pragma once
struct TeachingState {
    static constexpr uint64_t magic = 0x3148434145544753ull; // SGTEACH1
    static constexpr size_t count_words = 32;
    std::array<uint64_t, count_words> words{};
    // 0 magic, 1 schema, 2 active, 3 teacher count, 4 manifest hash,
    // 5 selected-prefix hash, 6 eligible documents, 7 online count at admission,
    // 8 replay count at admission, 9 assisted updates, 10 assisted pairs,
    // 11 temperature, 12 strength, 13 mixture (float bits), 14 population scope,
    // 15 replay pairs at admission. Two eight-word teacher records follow:
    // full-file hash, payload hash, parameters, C/H/L/cell, member-record hash.
    bool active() const { return words[2] == 1; }
    size_t count() const { return size_t(words[3]); }
    float temperature() const { return word_float(words[11]); }
    float strength() const { return word_float(words[12]); }
    float mixture() const { return word_float(words[13]); }
    size_t entry(size_t i) const { return 16 + 8 * i; }
    Config config(size_t i) const {
        size_t at = entry(i);
        for (size_t j = 3; j <= 6; ++j)
            if (words[at + j] > 1000000000ull)
                throw std::runtime_error("Invalid teacher dimensions");
        return {int(words[at + 3]), int(words[at + 4]), int(words[at + 5]), int(words[at + 6])};
    }
};
