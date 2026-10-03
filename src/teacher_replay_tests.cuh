// Mechanism checks, not language-quality evidence.
#pragma once
#include "frozen_model_tests.cuh"
void teacher_replay_test(const Args &args) {
    args.allow({"out"});
    fs::path out = args.get("out", "reports/teacher-replay-tests");
    fs::create_directories(out);
    frozen_model_test(out);
    auto rejects = [](auto action) {
        bool failed = false;
        try { action(); } catch (const std::exception &) { failed = true; }
        require(failed, "Invalid teacher replay input accepted");
    };
    // Hold the matrix shape constant while changing every future byte. This
    // tests causality without confusing it with GEMM shape/threshold rounding.
    for (int cell = 1; cell <= 6; ++cell) {
        Model teacher({8, 16, 2, cell}, 1, 16);
        auto weights = initialize(teacher.q, teacher.a, 321);
        activate_association_fixture(teacher.q, teacher.a, weights);
        teacher.w.put(weights);
        std::vector<int> x(16);
        std::iota(x.begin(), x.end(), 31);
        distillation::Targets targets(16, 2.f);
        auto allocation = targets.gpu_bytes();
        for (int n : {1, 7, 16, 3, 15}) {
            teacher.forward(x);
            targets.set_rows(n);
            rejects([&]() { targets.probabilities(); });
            targets.from_logits(teacher.logits, nullptr, .5f, true);
            auto first = targets.probabilities();
            auto changed = x;
            for (int i = n; i < 16; ++i) changed[i] = (changed[i] + 113) % 256;
            teacher.forward(changed);
            targets.from_logits(teacher.logits, nullptr, .5f, true);
            require(first == targets.probabilities(), "Future padding changed teacher prefix");
            require(first.size() == size_t(n) * 256 && targets.gpu_bytes() == allocation,
                    "Teacher tail resized or exposed stale targets");
            if (n < 16) rejects([&]() { targets.from_logits(teacher.logits); });
        }
        rejects([&]() { targets.set_rows(0); });
        rejects([&]() { targets.set_rows(17); });
        require(teacher.w.host() == weights && teacher.m.host() == std::vector<float>(teacher.a.n, 0) &&
                    teacher.v.host() == std::vector<float>(teacher.a.n, 0), "Teacher parameters mutated");
    }
    std::vector<std::string> docs{"A bird.",
                                 "The key is in the box.\nAnswer: box.",
                                 "The hat is in the bag.\nAnswer: bag."};
    for (size_t n = 1; n <= docs.size(); ++n) {
        std::ofstream file(out / (std::to_string(n) + ".dat"), std::ios::binary);
        for (size_t i = 0; i < n; ++i) file << (i ? std::string(1, char(30)) : "") << docs[i];
    }
    auto schedule = out / "curriculum.sg";
    {
        std::ofstream file(schedule);
        file << "SGCURRICULUM3\n8 \"1.dat\" 1 all 1\n16 \"2.dat\" .5 new 64\n28 \"3.dat\" .25 new 64\n";
    }
    std::vector<teachers::Origin> origins;
    for (int i = 0; i < 2; ++i) {
        Model parent({8 + 8*i, 16 + 8*i, 1 + i, 1 + 5*i}, 1, 8);
        parent.w.put(initialize(parent.q, parent.a, 91 + i));
        State s;
        auto path = out / ("parent-" + std::to_string(i) + ".ckpt");
        save(path, parent, s);
        origins.push_back({path, "", 0});
    }
    auto bundle_path = out / ("bundle-" + std::to_string(teachers::file_hash(out / "2.dat")));
    if (!fs::exists(bundle_path))
        teachers::create_bundle(bundle_path, out / "2.dat", origins, 2.f, .7f, .3f, false);
    teachers::Bundle immutable(bundle_path);
    for (size_t i = 0; i < origins.size(); ++i)
        require(teachers::file_hash(origins[i].checkpoint) == immutable.identity.words[immutable.identity.entry(i)],
                "Test bundle has stale parents");
    uint64_t budget = 32ull * 1024 * 1024, free_before = 0;
    rejects([&]() { teachers::Replay::check_budget(immutable, 8, 0, free_before); });
    LiveCurriculum curriculum(schedule);
    int cases = 0;
    uint64_t total_taught = 0, total_pairs = 0;
    for (int cell = 1; cell <= 6; ++cell)
        for (bool si : {false, true}) {
            Args options = args;
            options.values["--replay"] = "stage";
            options.values["--replay-capacity"] = "9";
            options.values["--replay-every"] = "2";
            options.values["--graph"] = "1";
            options.values["--speak-every"] = "2";
            options.values["--tokens"] = "7";
            if (si) options.values["--consolidation"] = "si";
            Config q{8, 16, 2, cell};
            LiveEngine first(q, 8, false), resumed(q, 8, false);
            State s, restored;
            LiveCorpus data(curriculum.stages.front().corpus);
            first.root.w.put(initialize(q, first.root.a, 123));
            configure_live(s, options, data, "A");
            curriculum.initialize(s);
            s.meta[5] = 1; s.meta[6] = 8;
            if (si) first.root.synapses->initialize(first.root.w, .1f, .001f);
            teachers::Bundle bundle(bundle_path);
            bundle.selected_prefix(LiveCorpus(out / "3.dat"));
            bundle.bind(s);
            first.teacher_replay = teachers::Replay::create(bundle, 8, budget);
            auto tick = [&](LiveEngine &engine, LiveCorpus &source, State &state) {
                curriculum.advance(engine, source, state);
                auto result = live_tick(engine, source, state, "A");
                ReplayMemory(state).validate(source, 8);
                validate_teaching_state(state);
                // All eligible documents coincide with the first two source
                // groups: independently accumulated replay counts must agree.
                uint64_t updates = 0, pairs = 0;
                StageReplayView groups(state);
                for (size_t g = 0; g < std::min(size_t(2), size_t(groups.groups())); ++g) {
                    updates += groups.value(g, 3); pairs += groups.value(g, 4);
                }
                require(state.teaching.words[9] == updates && state.teaching.words[10] == pairs,
                        "Teacher assistance included an ineligible or unobserved window");
                return result;
            };
            for (int i = 0; i < 11; ++i) tick(first, data, s);
            auto label = std::to_string(cell) + "-" + std::to_string(si);
            auto path = out / (label + "-split.ckpt");
            save(path, first.root, s);
            load(path, resumed.root, restored, true);
            auto other = curriculum.corpus(restored);
            curriculum.apply_feedback(other, size_t(restored.extra[15]));
            teachers::Bundle again(bundle_path);
            again.bind(restored);
            resumed.teacher_replay = teachers::Replay::create(again, 8, budget);
            for (int i = 11; i < 28; ++i) {
                auto a = tick(first, data, s), b = tick(resumed, other, restored);
                require(a.loss == b.loss && a.teacher_penalty == b.teacher_penalty && a.speech == b.speech &&
                            s.extra == restored.extra && s.teaching.words == restored.teaching.words,
                        "Teacher replay split trajectory differs");
            }
            auto full = out / (label + "-full.ckpt"), split = out / (label + "-resumed.ckpt");
            save(full, first.root, s); save(split, resumed.root, restored);
            require(teachers::file_hash(full) == teachers::file_hash(split), "Full teacher checkpoints differ");
            require(s.teaching.words[9] > 0 && s.teaching.words[9] < s.extra[6] &&
                        s.teaching.words[10] < 8 * s.teaching.words[9], "Fixture missed exclusions or short tails");
            // Replay shares parameters, but updates must not overwrite any live
            // membrane, trace, associative matrix or already captured decoder.
            auto live_state = first.root.membranes();
            auto &replay = first.replay_view(3);
            std::vector<int> x{65, 66, 67}, y{66, 67, 68};
            replay.forward(x, &y);
            auto replay_state = replay.membranes();
            auto a = first.teacher_replay->apply(replay, x, {1, 64, 1}, s);
            auto gradient = replay.dlogits.host();
            replay.forward(x, &y);
            auto b = first.teacher_replay->apply(replay, x, {1, 64, 1}, s);
            require(a.total_loss == b.total_loss && gradient == replay.dlogits.host() &&
                        replay.membranes() == replay_state && first.root.membranes() == live_state,
                        "Frozen teacher reset or learner recurrence isolation failed");
            replay.backward(); replay.update(int(++s.meta[7]), s.hp[0], s.hp[1], s.hp[2]);
            require(first.root.membranes() == live_state, "Teacher update changed live recurrence");
            ++cases;
            total_taught += restored.teaching.words[9]; total_pairs += restored.teaching.words[10];
        }
    teachers::Bundle after(bundle_path);
    require(after.identity.words == immutable.identity.words && after.weights == immutable.weights,
            "Learning changed the frozen bundle");
    std::ofstream report(out / "result.json");
    report << "{\"passed\":true,\"restart_cases\":" << cases
           << ",\"full_checkpoint_resume_exact\":true,\"cells\":[1,2,3,4,5,6],\"optional_si\":true,"
              "\"graph_speech_exact\":true,\"weighted_answers\":true,\"future_padding_causal\":true,"
              "\"bounded_tail_allocation\":true,\"live_recurrence_isolated\":true,\"teacher_weights_frozen\":true,"
              "\"eligibility_counts_match_source_groups\":true,\"zero_memory_budget_rejected\":true,"
              "\"teacher_updates_across_restart_cases\":" << total_taught
           << ",\"teacher_pairs_across_restart_cases\":" << total_pairs << "}\n";
    std::cout << "PASS teacher replay: causal prefixes, source eligibility, exact restart and shared learner\n";
}
