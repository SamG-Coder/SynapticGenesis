#pragma once
#include "live.cuh"

void synaptic_test(const Args &args) {
    args.allow({"out"});
    fs::path out = args.get("out", "reports/synaptic-tests");
    fs::create_directories(out);
    Config q{32, 64, 2};
    Model fixture(q, 1, 8);
    auto weights = initialize(q, fixture.a, 31);
    fixture.w.put(weights);
    fixture.synapses->initialize(fixture.w, .3f, .01f);
    std::vector<float> gradient(weights.size()), moment(weights.size()), variance(weights.size());
    auto plastic = fixture.synapses->host();
    for (size_t i = 0; i < weights.size(); ++i) {
        gradient[i] = .003f * std::sin(float(i % 83));
        moment[i] = .0001f * std::cos(float(i % 41));
        variance[i] = .00001f * (1 + float(i % 7));
        plastic[i] = .1f + .03f * float(i % 9);
        plastic[weights.size() + i] = .001f * (float(i % 5) - 2);
        plastic[2 * weights.size() + i] = weights[i] + .007f * (float(i % 3) - 1);
    }
    fixture.g.put(gradient);
    fixture.m.put(moment);
    fixture.v.put(variance);
    fixture.synapses->put(plastic);
    dump(out / "weights.f32", weights);
    dump(out / "task-gradient.f32", gradient);
    dump(out / "moment.f32", moment);
    dump(out / "variance.f32", variance);
    dump(out / "decay-mask.f32", fixture.decay.host());
    dump(out / "synapses-before.f32", plastic);
    float norm = fixture.update(5, .001f, .01f, .02f, .25f);
    dump(out / "updated.f32", fixture.w.host());
    dump(out / "moment-after.f32", fixture.m.host());
    dump(out / "variance-after.f32", fixture.v.host());
    dump(out / "clipped-gradient.f32", fixture.g.host());
    dump(out / "synapses-after-update.f32", fixture.synapses->host());
    fixture.synapses->consolidate(fixture.w);
    dump(out / "synapses-after-boundary.f32", fixture.synapses->host());
    std::ofstream config(out / "fixture.json");
    config << std::setprecision(10) << "{\"parameters\":" << weights.size()
           << ",\"core_end\":" << fixture.a.final_gain
           << ",\"strength\":0.3,\"damping\":0.01,\"step\":5,\"lr\":0.001,\"wd\":0.01,"
              "\"clip\":0.02,\"core_scale\":0.25,\"gradient_norm\":"
           << norm << "}\n";
    config.close();

    std::string text = "A girl counts toys.\x1eThe child sees the bird.\x1e"
                       "A child waits.";
    dump(out / "corpus.dat", std::vector<char>(text.begin(), text.end()));
    LiveCorpus data(out / "corpus.dat");
    float resume_error = 0, zero_error = 0, protection_effect = 0;
    for (int cell : {1, 2}) {
        q.cell = cell;
        Args settings = args;
        settings.values["--replay"] = "reservoir";
        settings.values["--replay-capacity"] = "5";
        settings.values["--replay-every"] = "2";
        settings.values["--graph"] = "1";
        settings.values["--consolidation"] = "si";
        settings.values["--si-strength"] = "10";
        State s;
        configure_live(s, settings, data, "A");
        s.meta[26] = 3;
        s.meta[27] = 7;
        s.hp[0] = .001f;
        LiveEngine first(q, 8, false), resumed(q, 8, false), zero(q, 8, false), ordinary(q, 8, false);
        auto initial = initialize(q, first.root.a, 31);
        first.root.w.put(initial);
        first.root.synapses->initialize(first.root.w, 10, .001f);
        require(first.root.synapses == first.speaker.synapses, "Speaker lost shared synaptic memory");
        for (int i = 0; i < 5; ++i)
            live_tick(first, data, s, "A");
        require(first.root.synapses->boundaries == 1, "Wrong document consolidation count");
        auto importance = first.root.synapses->importance.host();
        require(*std::max_element(importance.begin(), importance.end()) > 0,
                "Useful trajectory did not acquire importance");
        auto checkpoint = out / ("resume-" + std::to_string(cell) + ".ckpt");
        save(checkpoint, first.root, s);
        State restored;
        load(checkpoint, resumed.root, restored, true);
        require(s.meta[17] == 3 && s.extra == restored.extra &&
                    maxdiff(first.root.synapses->host(), resumed.root.synapses->host()) == 0,
                "Synaptic checkpoint did not roundtrip");
        for (int i = 0; i < 19; ++i) {
            auto a = live_tick(first, data, s, "A"), b = live_tick(resumed, data, restored, "A");
            require(a.speech == b.speech && s.extra == restored.extra,
                    "Consolidation resume changed speech or policy");
        }
        for (auto pair : {std::make_pair(first.root.w.host(), resumed.root.w.host()),
                          std::make_pair(first.root.m.host(), resumed.root.m.host()),
                          std::make_pair(first.root.v.host(), resumed.root.v.host()),
                          std::make_pair(first.root.membranes(), resumed.root.membranes()),
                          std::make_pair(first.root.synapses->host(), resumed.root.synapses->host())})
            resume_error = std::max(resume_error, maxdiff(pair.first, pair.second));
        require(resume_error < 3e-5, "Consolidated runtime resume differs");
        require(first.replay_view(4).synapses == first.root.synapses, "Replay lost shared synaptic memory");
        auto before = first.root.synapses->host();
        first.speak("A", s);
        require(maxdiff(before, first.root.synapses->host()) == 0,
                "Generated speech changed synaptic learning history");

        // Zero strength keeps all bookkeeping but must recover the original
        // optimizer, even after several consolidation boundaries and replays.
        State zs, os;
        settings.values["--si-strength"] = "0";
        configure_live(zs, settings, data, "A");
        settings.values["--consolidation"] = "none";
        settings.values.erase("--si-strength");
        configure_live(os, settings, data, "A");
        zs.meta[26] = os.meta[26] = 3;
        zs.meta[27] = os.meta[27] = 7;
        zs.hp[0] = os.hp[0] = .001f;
        zero.root.w.put(initial);
        ordinary.root.w.put(initial);
        zero.root.synapses->initialize(zero.root.w, 0, .001f);
        for (int i = 0; i < 24; ++i) {
            auto a = live_tick(zero, data, zs, "A"), b = live_tick(ordinary, data, os, "A");
            require(a.speech == b.speech, "Zero-strength consolidation changed speech");
        }
        zero_error = std::max(zero_error, maxdiff(zero.root.w.host(), ordinary.root.w.host()));
        require(zero_error < 3e-5, "Zero-strength consolidation changed weights");
        protection_effect = std::max(protection_effect, maxdiff(first.root.w.host(), zero.root.w.host()));

        std::ifstream in(checkpoint, std::ios::binary);
        std::vector<char> bytes((std::istreambuf_iterator<char>(in)), {});
        bytes.back() ^= 1;
        dump(out / "corrupt.ckpt", bytes);
        bool rejected = false;
        try {
            load(out / "corrupt.ckpt", resumed.root, restored, true);
        } catch (const std::exception &) {
            rejected = true;
        }
        require(rejected, "Corrupt synaptic payload was accepted");

        // A correct checksum must not make a negative importance acceptable.
        auto invalid = first.root.synapses->importance.host();
        invalid[0] = -1;
        first.root.synapses->importance.put(invalid);
        save(out / "invalid.ckpt", first.root, s);
        rejected = false;
        try {
            load(out / "invalid.ckpt", resumed.root, restored, true);
        } catch (const std::exception &) {
            rejected = true;
        }
        require(rejected, "Negative synaptic importance was accepted");
    }
    require(protection_effect > 1e-6, "Consolidation penalty had no learning effect");
    std::ofstream result(out / "native.json");
    result << "{\"passed\":true,\"resume_max_error\":" << resume_error
           << ",\"zero_strength_max_error\":" << zero_error
           << ",\"penalty_weight_difference\":" << protection_effect
           << ",\"both_cells_checked\":true,\"speech_identical_after_resume\":true,"
              "\"speech_preserves_synaptic_history\":true,\"invalid_payloads_rejected\":true}\n";
    std::cout << "PASS synaptic memory: both cells, live/replay/graph sharing, full resume, "
                 "zero-strength control, corruption rejection. Resume error "
              << resume_error << "\n";
}
