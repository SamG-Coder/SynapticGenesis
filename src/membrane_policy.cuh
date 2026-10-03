// Optional learning policy, separate from architecture and biological age.
#pragma once
struct MembranePolicy {
    static constexpr uint64_t live_extension = 7, magic = 0x314d454d504753ull; // SGPMEM1
    static constexpr size_t count_words = 8;
    // magic, schema, underlying live version, cost/band float bits, reserved[3].
    std::array<uint64_t, count_words> words{};
    bool present() const {
        return std::any_of(words.begin(), words.end(), [](uint64_t w) { return w != 0; });
    }
    float cost() const { return present() ? word_float(words[3]) : 0.f; }
    float band() const { return present() ? word_float(words[4]) : 1.5f; }
    static void validate_options(float cost, float band) {
        if (!std::isfinite(cost) || cost < 0 || cost > 100 || !std::isfinite(band) || band <= 1 || band >= 2)
            throw std::runtime_error("Membrane cost must be in [0,100] and band in (1,2)");
    }
    static MembranePolicy make(uint64_t base, float cost, float band) {
        validate_options(cost, band);
        if (base < 1 || base > 6 || cost == 0)
            throw std::runtime_error("Membrane policy requires a live base version and positive cost");
        MembranePolicy p;
        p.words = {magic, 1, base, float_word(cost), float_word(band), 0, 0, 0};
        return p;
    }
    void validate(int cell) const {
        if (words[0] != magic || words[1] != 1 || words[2] < 1 || words[2] > 6 ||
            words[3] > UINT32_MAX || words[4] > UINT32_MAX || words[5] || words[6] || words[7] ||
            cell < 3 || cell > 6)
            throw std::runtime_error("Invalid membrane policy or incompatible cell");
        validate_options(cost(), band());
        if (cost() == 0)
            throw std::runtime_error("Disabled membrane policy must use the legacy live format");
    }
    void report(std::ostream &out) const {
        if (present())
            out << ",\"membrane_cost\":" << cost() << ",\"membrane_band\":" << band();
    }
};
