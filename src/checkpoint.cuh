// Checkpoint state, bounded loading and atomic replacement.
#pragma once
#include "teaching_state.cuh"

struct State {
    std::array<uint64_t, 32> meta{};
    std::array<float, 8> hp{};
    // Version-2 live policy and bounded replay descriptors. meta[31] stores
    // the number of uint64 words; the payload is appended after membranes.
    std::vector<uint64_t> extra;
    // Live extension v3 appends importance, trajectory estimate, reference values.
    std::vector<float> synaptic;
    TeachingState teaching;
    State() {
        meta[0] = 0x314d4c53434e5042ull;
        meta[1] = 1;
        hp[0] = .001f;
        hp[1] = .01f;
        hp[2] = 1.f;
        hp[3] = std::numeric_limits<float>::max();
    }
};
bool has_grouped_replay(const State &s) {
    return s.meta[17] == 5 || s.meta[17] == 6;
}
#include "stage_replay.cuh"
bool has_curriculum(const State &s) {
    return s.meta[17] == 4 || has_grouped_replay(s);
}
bool has_synaptic_history(const State &s) {
    return s.meta[17] == 3 || (has_curriculum(s) && s.extra.size() >= 16 && s.extra[9] == 1);
}
void validate_curriculum_state(const State &s) {
    if (!has_curriculum(s))
        return;
    if (s.extra.size() < 16 || s.extra[9] > 1 || !s.extra[14] || s.extra[15] >= 4096 ||
        !std::isfinite(s.hp[7]) || s.hp[7] <= 0 || s.hp[7] > .1f)
        throw std::runtime_error("Invalid curriculum checkpoint policy");
    if (!s.extra[9])
        for (int i = 10; i <= 13; ++i)
            if (s.extra[i])
                throw std::runtime_error("Disabled curriculum consolidation has nonempty history");
}
void validate_teaching_state(const State &s) {
    const auto &p = s.teaching;
    const auto &w = p.words;
    if (s.meta[17] != 6) {
        if (std::any_of(w.begin(), w.end(), [](uint64_t value) { return value != 0; }))
            throw std::runtime_error("Teacher state requires live checkpoint version 6");
        return;
    }
    if (s.extra.size() < 17 || s.extra[1] != 3 || w[0] != TeachingState::magic || w[1] != 1 ||
        w[2] > 1 || w[3] < 1 || w[3] > 2 || !w[4] || !w[5] || !w[6] || w[6] > 1000000000ull ||
        w[7] > s.meta[24] || w[8] > s.extra[6] || w[15] > s.extra[7] ||
        w[9] > s.extra[6] - w[8] || w[10] > s.extra[7] - w[15] ||
        s.meta[5] != 1 || s.meta[6] < 1 || s.meta[6] > 4096 || s.meta[24] > 1000000000ull ||
        s.extra[6] > s.meta[24] || (w[9] == 0) != (w[10] == 0) ||
        w[10] < w[9] || w[10] > w[9] * s.meta[6] || w[14] > 1 ||
        w[11] > UINT32_MAX || w[12] > UINT32_MAX || w[13] > UINT32_MAX ||
        !std::isfinite(p.temperature()) || p.temperature() < .25f || p.temperature() > 16 ||
        !std::isfinite(p.strength()) || p.strength() < 0 || p.strength() > 100 ||
        !std::isfinite(p.mixture()) || p.mixture() < 0 || p.mixture() > 1)
        throw std::runtime_error("Invalid teacher policy or counters");
    for (size_t i = 0; i < p.count(); ++i) {
        size_t at = p.entry(i);
        Config q = p.config(i);
        if (!w[at] || !w[at + 1] || Layout(q).n != w[at + 2] ||
            (w[14] == 1 ? !w[at + 7] : w[at + 7] != 0))
            throw std::runtime_error("Invalid teacher checkpoint identity");
    }
    if (p.count() == 1 && std::any_of(w.begin() + 24, w.end(), [](uint64_t value) { return value != 0; }))
        throw std::runtime_error("Unused teacher identity must be empty");
    if (p.count() == 2 && w[16] == w[24])
        throw std::runtime_error("Duplicate teacher checkpoint identities");
}
// meta:
// magic/version/C/H/L/B/T/step/target_steps/warmup/RNG/seed/train_hash/val_hash/num_params/payload_hash/fast_math.
template <class T> void write_raw(std::ofstream &f, const T *p, size_t n) {
    f.write(reinterpret_cast<const char *>(p), std::streamsize(n * sizeof(T)));
    if (!f)
        throw std::runtime_error("File write failed");
}
template <class T> void read_raw(std::ifstream &f, T *p, size_t n) {
    f.read(reinterpret_cast<char *>(p), std::streamsize(n * sizeof(T)));
    if (!f)
        throw std::runtime_error("Truncated or unreadable file");
}
// Live checkpoints append the recurrent state and also protect the cursor,
// hyperparameters and RNG metadata. Ordinary v1 checkpoints stay compatible.
uint64_t live_hash(const State &s, const std::vector<float> &membranes, uint64_t hash) {
    if (!s.meta[17])
        return hash;
    hash = hash_bytes(membranes.data(), membranes.size() * 4, hash);
    auto meta = s.meta;
    meta[15] = 0;
    hash = hash_bytes(meta.data(), meta.size() * 8, hash);
    hash = hash_bytes(s.hp.data(), s.hp.size() * 4, hash);
    hash = hash_bytes(s.extra.data(), s.extra.size() * 8, hash);
    if (s.meta[17] == 6)
        hash = hash_bytes(s.teaching.words.data(), TeachingState::count_words * 8, hash);
    return hash_bytes(s.synaptic.data(), s.synaptic.size() * 4, hash);
}
void save(const fs::path &path, Model &m, State &s) {
    auto w = m.w.host(), mo = m.m.host(), vo = m.v.host();
    auto membranes = s.meta[17] ? m.membranes() : std::vector<float>{};
    s.meta[1] = m.q.cell;
    s.meta[2] = m.q.c;
    s.meta[3] = m.q.h;
    s.meta[4] = m.q.l;
    s.meta[5] = m.B;
    s.meta[6] = m.T;
    s.meta[14] = m.a.n;
    s.meta[18] = membranes.size();
    if (s.meta[17] >= 2)
        s.meta[31] = s.extra.size();
    else if (!s.extra.empty())
        throw std::runtime_error("Extended state requires live checkpoint version 2 through 6");
    if (s.meta[17] > 6)
        throw std::runtime_error("Unsupported live checkpoint version");
    validate_curriculum_state(s);
    validate_teaching_state(s);
    if (has_grouped_replay(s))
        StageReplay(s).validate();
    if (has_synaptic_history(s)) {
        if (!m.synapses->active() || s.extra.size() < 16)
            throw std::runtime_error("Missing synaptic state for enabled live consolidation");
        s.extra[9] = 1; // SI policy version.
        s.extra[10] = float_word(m.synapses->strength);
        s.extra[11] = float_word(m.synapses->damping);
        s.extra[12] = m.synapses->boundaries;
        s.extra[13] = m.synapses->updates;
        s.synaptic = m.synapses->host();
    } else if (m.synapses->active() || !s.synaptic.empty())
        throw std::runtime_error("Synaptic state requires an enabled consolidation policy");
    uint64_t hash = hash_bytes(w.data(), w.size() * 4);
    hash = hash_bytes(mo.data(), mo.size() * 4, hash);
    hash = hash_bytes(vo.data(), vo.size() * 4, hash);
    s.meta[15] = live_hash(s, membranes, hash);
    if (!path.parent_path().empty())
        fs::create_directories(path.parent_path());
    auto temp = path;
    temp += ".tmp";
    std::ofstream f(temp, std::ios::binary | std::ios::trunc);
    write_raw(f, s.meta.data(), 32);
    write_raw(f, s.hp.data(), 8);
    write_raw(f, w.data(), w.size());
    write_raw(f, mo.data(), mo.size());
    write_raw(f, vo.data(), vo.size());
    if (!membranes.empty())
        write_raw(f, membranes.data(), membranes.size());
    if (!s.extra.empty())
        write_raw(f, s.extra.data(), s.extra.size());
    if (s.meta[17] == 6)
        write_raw(f, s.teaching.words.data(), TeachingState::count_words);
    if (!s.synaptic.empty())
        write_raw(f, s.synaptic.data(), s.synaptic.size());
    f.close();
    if (!f)
        throw std::runtime_error("Checkpoint flush failed");
    // Keep the previous checkpoint if replacing an existing file fails.
    if (fs::exists(path)) {
        auto prev = path;
        prev += ".previous";
        if (fs::exists(prev))
            fs::remove(prev);
        fs::rename(path, prev);
        try {
            fs::rename(temp, path);
        } catch (...) {
            fs::rename(prev, path);
            throw;
        }
        fs::remove(prev);
    } else
        fs::rename(temp, path);
}
State header(const fs::path &path) {
    std::ifstream f(path, std::ios::binary);
    State s;
    read_raw(f, s.meta.data(), 32);
    read_raw(f, s.hp.data(), 8);
    if (s.meta[0] != 0x314d4c53434e5042ull || s.meta[1] < 1 || s.meta[1] > 6)
        throw std::runtime_error("Unsupported checkpoint architecture/version");
    for (int i = 2; i <= 9; ++i)
        if (s.meta[i] > 1000000000ull)
            throw std::runtime_error("Invalid checkpoint metadata");
    return s;
}
struct StoredCheckpoint {
    State state;
    std::vector<float> weights, first, second, membranes;
};
StoredCheckpoint read_checkpoint(const fs::path &path) {
    State s = header(path);
    Config q{int(s.meta[2]), int(s.meta[3]), int(s.meta[4]), int(s.meta[1])};
    Layout layout(q);
    if (s.meta[14] != layout.n)
        throw std::runtime_error("Checkpoint parameter count differs from its dimensions");
    if (s.meta[17] > 6 || (!s.meta[17] && s.meta[18]) ||
        (s.meta[17] &&
         (s.meta[5] < 1 || s.meta[5] > 256 || s.meta[18] != q.recurrent_per_layer(int(s.meta[5])) * q.l)))
        throw std::runtime_error("Invalid live checkpoint dimensions/version");
    if ((s.meta[17] < 2 && s.meta[31]) || s.meta[31] > (has_grouped_replay(s) ? 17 + 5 * 4096 : 16) + 3 * 65536)
        throw std::runtime_error("Invalid extended checkpoint length");
    // Read the bounded policy before deciding whether a v4 file has SI arrays.
    std::ifstream f(path, std::ios::binary);
    f.seekg(std::streamoff(288 + 12 * layout.n + 4 * s.meta[18]));
    s.extra.resize(size_t(s.meta[31]));
    if (!s.extra.empty())
        read_raw(f, s.extra.data(), s.extra.size());
    size_t teaching_bytes = s.meta[17] == 6 ? TeachingState::count_words * 8 : 0;
    if (teaching_bytes)
        read_raw(f, s.teaching.words.data(), TeachingState::count_words);
    validate_curriculum_state(s);
    validate_teaching_state(s);
    if (has_grouped_replay(s))
        StageReplay(s).validate();
    size_t synaptic_count = has_synaptic_history(s) ? 3 * layout.n : 0;
    if (fs::file_size(path) != 288 + 12 * layout.n + 4 * s.meta[18] + 8 * s.meta[31] + teaching_bytes + 4 * synaptic_count)
        throw std::runtime_error("Checkpoint length mismatch");
    f.seekg(288);
    std::vector<float> w(layout.n), mo(layout.n), vo(layout.n), membranes(size_t(s.meta[18]));
    read_raw(f, w.data(), w.size());
    read_raw(f, mo.data(), mo.size());
    read_raw(f, vo.data(), vo.size());
    if (!membranes.empty())
        read_raw(f, membranes.data(), membranes.size());
    f.seekg(std::streamoff(288 + 12 * layout.n + 4 * s.meta[18] + 8 * s.meta[31] + teaching_bytes));
    s.synaptic.resize(synaptic_count);
    if (synaptic_count)
        read_raw(f, s.synaptic.data(), s.synaptic.size());
    uint64_t h = hash_bytes(w.data(), w.size() * 4);
    h = hash_bytes(mo.data(), mo.size() * 4, h);
    h = hash_bytes(vo.data(), vo.size() * 4, h);
    h = live_hash(s, membranes, h);
    if (h != s.meta[15])
        throw std::runtime_error("Checkpoint checksum mismatch");
    for (auto *v : {&w, &mo, &vo, &membranes, &s.synaptic})
        for (float x : *v)
            if (!std::isfinite(x))
                throw std::runtime_error("Nonfinite checkpoint parameter");
    if (synaptic_count) {
        if (s.extra.size() < 16 || s.extra[9] != 1 || (s.meta[17] == 3 && (s.extra[14] || s.extra[15])) ||
            s.extra[12] > s.meta[24] || s.extra[13] != s.meta[24] + s.extra[6])
            throw std::runtime_error("Invalid synaptic checkpoint policy/counters");
        float strength = word_float(s.extra[10]), damping = word_float(s.extra[11]);
        if (!std::isfinite(strength) || strength < 0 || strength > 1000000 || !std::isfinite(damping) ||
            damping <= 0 || damping > 1 ||
            std::any_of(s.synaptic.begin(), s.synaptic.begin() + layout.n, [](float x) { return x < 0; }))
            throw std::runtime_error("Invalid synaptic checkpoint values");
    }
    return {std::move(s), std::move(w), std::move(mo), std::move(vo), std::move(membranes)};
}
void load(const fs::path &path, Model &m, State &s, bool restore_runtime = false) {
    auto stored = read_checkpoint(path);
    s = std::move(stored.state);
    if (s.meta[1] != uint64_t(m.q.cell) || s.meta[2] != uint64_t(m.q.c) || s.meta[3] != uint64_t(m.q.h) ||
        s.meta[4] != uint64_t(m.q.l) || s.meta[14] != m.a.n)
        throw std::runtime_error("Checkpoint model dimensions differ");
    if (restore_runtime && (!s.meta[17] || s.meta[5] != uint64_t(m.B)))
        throw std::runtime_error("No compatible live state in checkpoint");
    // Loading into an existing shared runtime must not keep stale importance.
    *m.synapses = SynapticMemory{};
    m.w.put(stored.weights);
    m.m.put(stored.first);
    m.v.put(stored.second);
    if (restore_runtime) {
        m.membranes(stored.membranes);
        if (!s.synaptic.empty()) {
            m.synapses->initialize(m.w, word_float(s.extra[10]), word_float(s.extra[11]));
            m.synapses->put(s.synaptic);
            m.synapses->boundaries = s.extra[12];
            m.synapses->updates = s.extra[13];
        }
    }
}
