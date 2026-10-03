#pragma once
void stage_replay_test(const Args &args) {
    args.allow({"out"});
    fs::path out = args.get("out", "reports/stage-replay-tests");
    fs::create_directories(out);
    auto policy = [&](int seed, const char *mode) {
        Args options = args;
        options.values["--replay"] = mode;
        options.values["--replay-capacity"] = "6";
        options.values["--replay-seed"] = std::to_string(seed);
        options.values["--replay-every"] = "2";
        State s;
        ReplayMemory::configure(s, options);
        if (s.extra[1] == 3) {
            s.extra[14] = 1;
            StageReplay(s).initialize(1);
        }
        return s;
    };
    // Same first-stage reservoir and random draw sequence as the older policy.
    State grouped = policy(1777, "stage"), uniform = policy(1777, "reservoir");
    for (uint64_t i = 0; i < 100; ++i) {
        Episode e{0, i, 1};
        ++grouped.meta[24];
        ++uniform.meta[24];
        if (ReplayMemory(grouped).due()) {
            auto a = ReplayMemory(grouped).choose(e), b = ReplayMemory(uniform).choose(e);
            require(a.document == b.document && a.offset == b.offset && a.length == b.length,
                    "One-stage replay changed the reservoir sequence");
            ReplayMemory(grouped).completed(a);
            ReplayMemory(uniform).completed(b);
        }
        ReplayMemory(grouped).remember(e);
        ReplayMemory(uniform).remember(e);
        require(grouped.extra[4] == uniform.extra[4], "One-stage replay consumed different RNG draws");
        for (size_t j = 0; j < ReplayMemory(grouped).count(); ++j)
            require(ReplayMemory(grouped).at(j).offset == ReplayMemory(uniform).at(j).offset,
                    "One-stage reservoirs differ");
    }
    StageReplayView(grouped).validate();
    State conversion = uniform;
    conversion.meta[17] = 4;
    conversion.extra[14] = 1;
    auto old_meta = conversion.meta;
    StageReplay(conversion).adopt_first_stage(1);
    old_meta[17] = 5;
    require(conversion.meta == old_meta && conversion.hp == uniform.hp && conversion.extra == grouped.extra,
            "First-stage policy conversion changed learning history or replay state");

    // A returning earlier source remains a reservoir over all its observations,
    // including before and after quota shrink. Check inclusion frequencies.
    std::array<int, 24> inclusion{};
    constexpr int trials = 4000;
    for (int seed = 1; seed <= trials; ++seed) {
        State s = policy(seed, "stage");
        auto observe = [&](Episode e) {
            ++s.meta[24];
            ReplayMemory(s).remember(e);
        };
        for (uint64_t i = 0; i < 12; ++i)
            observe({0, i, 1});
        ++s.extra[15];
        StageReplay(s).add_group(2);
        for (uint64_t i = 0; i < 100; ++i)
            observe({1, i, 1});
        for (uint64_t i = 12; i < 24; ++i)
            observe({0, i, 1});
        StageReplayView(s).validate();
        require(StageReplayView(s).value(0, 2) == 3 && StageReplayView(s).value(1, 2) == 3,
                "New source frequency changed balanced quotas");
        for (size_t i = 0; i < 3; ++i)
            ++inclusion[size_t(ReplayMemory(s).at(i).offset)];
    }
    for (int n : inclusion)
        require(n > 380 && n < 650, "Stage reservoir inclusion is strongly biased");
    ++grouped.extra[15];
    StageReplay(grouped).add_group(2);
    for (uint64_t i = 0; i < 1000; ++i) {
        ++grouped.meta[24];
        ReplayMemory(grouped).remember({1, i, 1});
    }
    std::array<int, 2> chosen{};
    for (int i = 0; i < 10000; ++i)
        ++chosen[size_t(StageReplay(grouped).choose().document)];
    require(chosen[0] > 4750 && chosen[0] < 5250, "Replay did not balance source selection");
    for (int fault = 0; fault < 4; ++fault) {
        State bad = grouped;
        if (fault == 0)
            bad.extra[16] = 4097;
        if (fault == 1)
            ++bad.extra[StageReplayView(bad).field(0, 1)];
        if (fault == 2)
            bad.extra[StageReplayView(bad).begin()] = 1;
        if (fault == 3)
            bad.extra.pop_back();
        bool rejected = false;
        try {
            StageReplayView(bad).validate();
        } catch (const std::exception &) {
            rejected = true;
        }
        require(rejected, "Malformed grouped replay layout accepted");
    }

    std::vector<std::string> docs{"A child reads a book beside a tree. ",
                                  "The key is in the box.\nAnswer: box.",
                                  "The hat is in the bag.\nAnswer: bag."};
    for (size_t n = 1; n <= 3; ++n) {
        std::ofstream f(out / (std::to_string(n) + ".dat"), std::ios::binary);
        for (size_t i = 0; i < n; ++i) {
            if (i)
                f << char(30);
            f << docs[i];
        }
    }
    auto schedule = out / "curriculum.sg";
    {
        std::ofstream f(schedule);
        f << "SGCURRICULUM3\n4 \"1.dat\" 1 all 1\n8 \"2.dat\" .5 new 64\n12 \"3.dat\" .25 all 64\n";
    }
    LiveCurriculum curriculum(schedule);
    float error = 0;
    int cases = 0;
    for (int cell : {1, 2, 3, 4, 5, 6})
        for (bool si : {false, true}) {
            Args options = args;
            options.values["--replay"] = "stage";
            options.values["--replay-capacity"] = "6";
            options.values["--replay-every"] = "2";
            options.values["--graph"] = "1";
            options.values["--speak-every"] = "2";
            options.values["--tokens"] = "7";
            if (si) {
                options.values["--consolidation"] = "si";
                options.values["--si-strength"] = ".02";
            }
            Config q{8, 16, 2, cell};
            LiveEngine first(q, 8, false), resumed(q, 8, false);
            State s, restored;
            LiveCorpus data(curriculum.stages.front().corpus);
            first.root.w.put(initialize(q, first.root.a, 123));
            configure_live(s, options, data, "A");
            curriculum.initialize(s);
            if (si)
                first.root.synapses->initialize(first.root.w, .02f, .001f);
            auto tick = [&](LiveEngine &engine, LiveCorpus &source, State &state) {
                curriculum.advance(engine, source, state);
                auto result = live_tick(engine, source, state, "A");
                ReplayMemory(state).validate(source, 8);
                return result;
            };
            for (int i = 0; i < 7; ++i)
                tick(first, data, s);
            auto path = out / (std::to_string(cell) + "-" + std::to_string(si) + ".ckpt");
            save(path, first.root, s);
            load(path, resumed.root, restored, true);
            auto other = curriculum.corpus(restored);
            for (int i = 7; i < 12; ++i) {
                auto a = tick(first, data, s), b = tick(resumed, other, restored);
                require(a.speech == b.speech && s.extra == restored.extra, "Stage replay restart differs");
                error = std::max(error, std::abs(a.loss - b.loss));
            }
            for (auto pair : {std::make_pair(&first.root.w, &resumed.root.w),
                              std::make_pair(&first.root.m, &resumed.root.m),
                              std::make_pair(&first.root.v, &resumed.root.v)})
                error = std::max(error, maxdiff(pair.first->host(), pair.second->host()));
            error = std::max(error, maxdiff(first.root.membranes(), resumed.root.membranes()));
            if (si)
                error = std::max(error, maxdiff(first.root.synapses->host(), resumed.root.synapses->host()));
            require(error < 3e-5 && StageReplayView(s).groups() == 3 && ReplayMemory(s).count() == 6,
                    "Stage replay resume or bounded memory failed");
            ++cases;
        }
    std::ofstream f(out / "native.json");
    f << "{\"passed\":true,\"restart_cases\":" << cases << ",\"resume_max_error\":" << error
      << ",\"one_stage_matches_reservoir\":true,\"quota_shrink_and_returning_source_checked\":true,"
         "\"inclusion_frequency_trials\":"
      << trials
      << ",\"equal_group_selection_checked\":true,"
         "\"invalid_payload_rejected\":true,\"cells\":[1,2,3,4,5,6],\"optional_si\":true,\"graph_speech_"
         "identical\":true}\n";
    std::cout << "PASS stage replay: bounded quotas, selection, returning sources, typed restart and shared "
                 "live state\n";
}
