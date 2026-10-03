// Host-only file/policy checks. CUDA is compiled but never called by this main.
#define main synapticgenesis_cli_entry
#include "../src/spike_lm.cu"
#undef main
#include "membrane_learning.cuh"

namespace membrane_checkpoint_test {
int checks = 0, rejected = 0, fixtures = 0;
void check(bool good, const char *message) {
    require(good, message);
    ++checks;
}
template <class F> void rejects(F fn, const std::string &message) {
    bool failed = false;
    try { fn(); } catch (const std::runtime_error &e) {
        failed = std::string(e.what()).find(message) != std::string::npos;
        if (!failed)
            throw;
    }
    require(failed, "Malformed fixture was accepted or produced the wrong diagnostic");
    ++rejected;
}
void wrap(State &s, float cost = .01f, float band = 1.5f) {
    s.membrane = MembranePolicy::make(live_version(s), cost, band);
    s.meta[17] = 7;
}
StoredCheckpoint fixture(int base, int cell, bool si = false) {
    Config q{8, 8, 2, cell};
    Layout layout(q);
    StoredCheckpoint r;
    auto &s = r.state;
    s.meta[1] = cell;
    s.meta[2] = q.c;
    s.meta[3] = q.h;
    s.meta[4] = q.l;
    s.meta[5] = 1;
    s.meta[6] = 4;
    s.meta[14] = layout.n;
    s.meta[17] = base;
    s.meta[18] = base ? q.l * q.recurrent_per_layer() : 0;
    r.weights = initialize(q, layout, 1337);
    r.first.assign(layout.n, .125f);
    r.second.assign(layout.n, .0625f);
    r.membranes.assign(size_t(s.meta[18]), .25f);
    if (base >= 2) {
        Args args(0, nullptr);
        if (base >= 5) {
            args.values["--replay"] = "stage";
            args.values["--replay-capacity"] = "8";
        }
        ReplayMemory::configure(s, args);
        set_live_version(s, uint64_t(base));
    }
    if (base >= 4) {
        s.extra[14] = 12345;
        s.hp[7] = .001f;
    }
    if (base >= 5) {
        StageReplay(s).initialize(2);
        set_live_version(s, uint64_t(base));
    }
    if (base == 3 || si) {
        s.extra[9] = 1;
        s.extra[10] = float_word(.1f);
        s.extra[11] = float_word(.001f);
        s.synaptic.assign(layout.n * 3, .25f);
    }
    if (base == 6) {
        auto &w = s.teaching.words;
        w[0] = TeachingState::magic;
        w[1] = 1; w[3] = 1; w[4] = 99; w[5] = 101; w[6] = 2;
        w[11] = float_word(1.f); w[12] = float_word(.2f); w[13] = float_word(1.f);
        w[16] = 123; w[17] = 321; w[18] = Layout(Config{8,8,1,1}).n;
        w[19] = 8; w[20] = 8; w[21] = 1; w[22] = 1;
    }
    s.meta[31] = s.extra.size();
    return r;
}
// Independent fixture serializer, with explicit on-disk ordering and offsets.
// The production header/read_checkpoint functions consume the resulting files.
void write_fixture(const fs::path &path, StoredCheckpoint &r) {
    auto &s = r.state;
    uint64_t h = hash_bytes(r.weights.data(), r.weights.size() * 4);
    h = hash_bytes(r.first.data(), r.first.size() * 4, h);
    h = hash_bytes(r.second.data(), r.second.size() * 4, h);
    s.meta[15] = live_hash(s, r.membranes, h);
    std::ofstream f(path, std::ios::binary);
    write_raw(f, s.meta.data(), 32);
    write_raw(f, s.hp.data(), 8);
    if (s.meta[17] == 7)
        write_raw(f, s.membrane.words.data(), 8);
    for (const auto *v : {&r.weights, &r.first, &r.second, &r.membranes})
        write_raw(f, v->data(), v->size());
    write_raw(f, s.extra.data(), s.extra.size());
    if (live_version(s) == 6)
        write_raw(f, s.teaching.words.data(), 32);
    write_raw(f, s.synaptic.data(), s.synaptic.size());
    f.close();
    require(bool(f), "Fixture flush failed");
}
void compare(const StoredCheckpoint &a, const StoredCheckpoint &b) {
    check(a.state.meta == b.state.meta && a.state.hp == b.state.hp &&
          a.state.membrane.words == b.state.membrane.words && a.state.extra == b.state.extra &&
          a.state.teaching.words == b.state.teaching.words && a.state.synaptic == b.state.synaptic &&
          a.weights == b.weights && a.first == b.first && a.second == b.second && a.membranes == b.membranes,
          "Fixture differs after native read");
}
void patch_word(const fs::path &path, size_t offset, uint64_t word) {
    std::fstream f(path, std::ios::binary | std::ios::in | std::ios::out);
    f.seekp(std::streamoff(offset));
    f.write(reinterpret_cast<const char *>(&word), 8);
    require(bool(f), "Cannot mutate fixture");
}
void run(const fs::path &out) {
    require(!out.empty() && !fs::exists(out), "Host checks need a fresh output directory");
    fs::create_directories(out);
    for (int cell = 3; cell <= 6; ++cell)
        for (int base = 0; base <= 6; ++base) {
            for (int si = 0; si <= int(base >= 4); ++si) {
                auto legacy = fixture(base, cell, si != 0), extended = legacy;
                auto stem = std::to_string(cell) + "-" + std::to_string(base) + "-" + std::to_string(si);
                auto old_path = out / (stem + "-legacy.ckpt");
                write_fixture(old_path, legacy);
                compare(legacy, read_checkpoint(old_path));
                ++fixtures;
                if (!base)
                    continue;
                wrap(extended.state);
                auto path = out / (stem + "-membrane.ckpt");
                write_fixture(path, extended);
                compare(extended, read_checkpoint(path));
                ++fixtures;
                auto head = header(path);
                check(head.membrane.words == extended.state.membrane.words && live_version(head) == uint64_t(base),
                      "Lightweight header lost the base live policy");
                check(fs::file_size(path) == fs::file_size(old_path) + 64, "Extension did not add exactly 64 bytes");
                check(has_curriculum(head) == has_curriculum(legacy.state) &&
                      has_grouped_replay(head) == has_grouped_replay(legacy.state), "Feature detection differs");
            }
        }
    auto source = fixture(5, 6);
    wrap(source.state);
    auto good = out / "valid.ckpt", bad = out / "invalid.ckpt";
    write_fixture(good, source);
    // Header corruption is rejected before reading/allocating learned arrays.
    const std::vector<std::pair<size_t, uint64_t>> bad_words = {
        {0, 0}, {1, 2}, {2, 0}, {2, 7}, {3, 0}, {3, float_word(-1.f)},
        {3, float_word(101.f)}, {3, float_word(std::numeric_limits<float>::infinity())},
        {3, float_word(std::numeric_limits<float>::quiet_NaN())}, {3, 1ull << 32},
        {4, float_word(1.f)}, {4, float_word(2.f)}, {4, 1ull << 32},
        {4, float_word(std::numeric_limits<float>::quiet_NaN())}, {5, 1}, {6, 1}, {7, 1}};
    for (const auto &entry : bad_words) {
        fs::copy_file(good, bad, fs::copy_options::overwrite_existing);
        patch_word(bad, 288 + entry.first * 8, entry.second);
        rejects([&] { header(bad); }, entry.first == 3 && entry.second == 0 ? "Disabled membrane" :
            (entry.first == 3 && entry.second <= UINT32_MAX) ||
            (entry.first == 4 && entry.second <= UINT32_MAX) ? "Membrane cost" : "Invalid membrane policy");
    }
    for (uint64_t cell : {1ull, 2ull}) {
        fs::copy_file(good, bad, fs::copy_options::overwrite_existing);
        patch_word(bad, 8, cell);
        rejects([&] { header(bad); }, "incompatible cell");
    }
    fs::copy_file(good, bad, fs::copy_options::overwrite_existing);
    patch_word(bad, 17 * 8, 8);
    rejects([&] { header(bad); }, "Unsupported live version");
    for (size_t length : {size_t(288), size_t(351)}) {
        fs::copy_file(good, bad, fs::copy_options::overwrite_existing);
        fs::resize_file(bad, length);
        rejects([&] { header(bad); }, "Truncated");
    }
    fs::copy_file(good, bad, fs::copy_options::overwrite_existing);
    patch_word(bad, 288 + 3 * 8, float_word(.02f));
    rejects([&] { read_checkpoint(bad); }, "checksum mismatch");
    fs::copy_file(good, bad, fs::copy_options::overwrite_existing);
    patch_word(bad, 352, 0); // finite but changed learned values
    rejects([&] { read_checkpoint(bad); }, "checksum mismatch");
    fs::copy_file(good, bad, fs::copy_options::overwrite_existing);
    fs::resize_file(bad, fs::file_size(bad) + 1);
    rejects([&] { read_checkpoint(bad); }, "length mismatch");
    fs::copy_file(good, bad, fs::copy_options::overwrite_existing);
    fs::resize_file(bad, fs::file_size(bad) - 1);
    rejects([&] { read_checkpoint(bad); }, "Truncated");
    State stray;
    stray.membrane = source.state.membrane;
    rejects([&] { validate_membrane_state(stray); }, "stray membrane policy");
    for (uint64_t base = 1; base <= 6; ++base) {
        State s = source.state;
        auto words = s.membrane.words;
        set_live_version(s, base);
        words[2] = base;
        check(s.meta[17] == 7 && s.membrane.words == words, "Transition discarded the objective");
        State disabled;
        set_live_version(disabled, base);
        check(disabled.meta[17] == base && !disabled.membrane.present(), "Disabled transition changed format");
    }
    auto text_path = out / "host-fixture.dat";
    std::vector<unsigned char> bytes{'a','b','c','d',30,'e','f','g','h'};
    dump(text_path, bytes);
    LiveCorpus data(text_path);
    Args args(0, nullptr);
    args.values = {{"--membrane-cost", ".01"}, {"--membrane-band", "1.75"}};
    State s;
    s.meta[1] = 6;
    configure_live(s, args, data, "a");
    check(s.meta[17] == 7 && live_version(s) == 2 && s.membrane.band() == 1.75f, "New policy not configured");
    auto words = s.membrane.words;
    Args inherited(0, nullptr);
    inherited.values["--consolidation"] = "si";
    configure_live(s, inherited, data, "a");
    words[2] = 3;
    check(s.membrane.words == words && has_synaptic_history(s), "New stream failed to inherit objective");
    Args off(0, nullptr);
    off.values["--membrane-cost"] = "0";
    configure_live(s, off, data, "a");
    check(s.meta[17] == 2 && !s.membrane.present(), "Explicit disable retained extension");
    off.values["--membrane-band"] = "1.5";
    rejects([&] { requested_membrane_policy(s, off); }, "positive --membrane-cost");
    auto convert = fixture(4, 6).state;
    convert.extra[1] = 1; convert.extra[2] = 4; convert.extra[3] = 8;
    wrap(convert);
    StageReplay(convert).adopt_first_stage(2);
    check(convert.meta[17] == 7 && live_version(convert) == 5 && StageReplay(convert).groups() == 1,
          "Stage conversion discarded the wrapper");
    ReplayMemory(convert).validate(data, 4);
    check(convert.membrane.cost() == .01f, "Stage conversion discarded cost");
    {
        std::ofstream f(out / "host-curriculum.sg", std::ios::binary);
        f << "SGCURRICULUM2\n5 \"host-fixture.dat\" 1 all\n";
    }
    LiveCurriculum curriculum(out / "host-curriculum.sg");
    teachers::create_bundle(out / "teachers", text_path,
                            {{out / "3-0-0-legacy.ckpt", "", 0}}, 1.f, .2f, 1.f, false);
    teachers::Bundle bundle(out / "teachers");
    for (bool si : {false, true}) {
        Args options(0, nullptr);
        options.values = {{"--membrane-cost", ".02"}, {"--replay", "stage"}, {"--replay-capacity", "8"}};
        if (si)
            options.values["--consolidation"] = "si";
        State transition;
        transition.meta[1] = 6; transition.meta[5] = 1; transition.meta[6] = 4;
        configure_live(transition, options, data, "a");
        curriculum.initialize(transition);
        curriculum.validate(transition);
        ReplayMemory(transition).validate(data, 4);
        check(transition.meta[17] == 7 && live_version(transition) == 5 &&
              has_synaptic_history(transition) == si, "Curriculum initialization lost wrapped features");
        bundle.bind(transition);
        validate_teaching_state(transition);
        validate_membrane_state(transition);
        ReplayMemory(transition).validate(data, 4);
        auto saved = transition.teaching.words;
        bundle.bind(transition);
        check(transition.meta[17] == 7 && live_version(transition) == 6 &&
              transition.membrane.cost() == .02f && transition.teaching.words == saved &&
              has_synaptic_history(transition) == si, "Teacher bind or rebind lost wrapped features");
    }
    rejects([&] { teachers::create_bundle(out / "rejected-teacher", text_path,
                         {{good, "", 0}}, 1.f, .2f, 1.f, false); }, "not admitted as a teacher");
    check(!fs::exists(out / "rejected-teacher"), "Rejected teacher package was created");
    for (const auto *cell : {"lif", "alif"}) {
        Args invalid(0, nullptr);
        invalid.values = {{"--cell", cell}, {"--membrane-cost", ".01"},
                          {"--out", (out / "invalid-live").string()}};
        rejects([&] { live_command(invalid); }, "requires trace");
    }
    for (const auto *option : {"--membrane-cost", "--membrane-band"}) {
        Args invalid(0, nullptr);
        invalid.values = {{"--resume", (out / "6-2-0-membrane.ckpt").string()},
                          {"--out", (out / "invalid-resume").string()},
                          {option, std::string(option) == "--membrane-cost" ? ".01" : "1.5"}};
        rejects([&] { live_command(invalid); }, "Live resume preserves membrane");
    }
    evolution::Candidate parent;
    parent.state = source.state;
    parent.state.meta[7] = 1;
    parent.parameters = 100; parent.score = 0;
    check(!evolution::eligible(parent, 1000), "Experimental parent became eligible");
    uint64_t rng = 42;
    evolution::Rules rules;
    rejects([&] { evolution::child_state(parent, parent, rules, rng); }, "not admitted to reproduction");
    std::ofstream report(out / "result.json", std::ios::binary);
    report << "{\"passed\":true,\"cuda_executed\":false,\"fixtures\":" << fixtures
           << ",\"checks\":" << checks << ",\"rejections\":" << rejected
           << ",\"legacy_versions\":[0,1,2,3,4,5,6],\"wrapped_live_versions\":[1,2,3,4,5,6],"
              "\"wrapped_cells\":[3,4,5,6],\"added_bytes\":64,\"gpu_restart_verified\":false}\n";
    report.close();
    require(bool(report), "Cannot write host result");
    std::cout << "PASS: " << fixtures << " native-reader fixtures, " << checks << " checks, "
              << rejected << " rejections; no CUDA calls.\n";
}
} // namespace membrane_checkpoint_test
int main(int argc, char **argv) {
    try {
        Args args(argc, argv);
        require(argc >= 2, "Expected host-test or gpu-test --out PATH");
        args.allow({"out"});
        if (std::string(argv[1]) == "host-test")
            membrane_checkpoint_test::run(args.get("out"));
        else if (std::string(argv[1]) == "gpu-test")
            membrane_learning_test::run(args.get("out"));
        else
            throw std::runtime_error("Expected host-test or gpu-test --out PATH");
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
}
