// Matched-observation continual-language experiment through the real live engine.
// Every arm starts from one checked checkpoint; only its declared memory/plasticity
// intervention differs. Held-out text is evaluated with independent recurrence.
#pragma once
#include "live.cuh"
void retention_bench(const Args &args) {
    args.allow({"train-a",
                "train-b",
                "validation-a",
                "validation-b",
                "out",
                "steps-a",
                "steps-b",
                "seed",
                "channels",
                "hidden",
                "layers",
                "cell",
                "chunk",
                "lr",
                "fast",
                "eval-batches",
                "replay-capacity",
                "replay-every",
                "si-strength",
                "speak-every",
                "tokens"});
    fs::path out = args.get("out", "runs/retention");
    require(!fs::exists(out), "Retention experiment exists; choose a new output directory");
    int a_steps = args.num("steps-a", 6000), b_steps = args.num("steps-b", 6000),
        chunk = args.num("chunk", 128), seed = args.num("seed", 1337), batches = args.num("eval-batches", 32);
    require(a_steps > 0 && b_steps > 0 && a_steps <= 10000000 && b_steps <= 10000000 && chunk >= 1 &&
                chunk <= 4096 && seed > 0 && batches >= 1 && batches <= 10000,
            "Invalid retention experiment limits");
    Config q{args.num("channels", 256), args.num("hidden", 512), args.num("layers", 4), cell_version(args)};
    LiveCorpus a(args.get("train-a")), b(args.get("train-b")), va(args.get("validation-a")),
        vb(args.get("validation-b"));
    validate_live_holdout(a, va);
    validate_live_holdout(a, vb);
    validate_live_holdout(b, va);
    validate_live_holdout(b, vb);
    require(a.hash != b.hash && va.hash != vb.hash, "Retention phases and held-out domains must differ");
    Data validation_a(args.get("validation-a"), 128), validation_b(args.get("validation-b"), 128);
    int every = args.num("replay-every", 4), capacity = args.num("replay-capacity", 1024);
    require(every > 0 && every <= a_steps && every <= b_steps && capacity > 0 && capacity <= 65536,
            "Invalid retention replay budget");
    float strength = args.real("si-strength", .001f);
    require(strength > 0 && strength <= 1000000, "Invalid retention SI strength");
    fs::create_directories(out);
    dump(out / "a.dat", a.bytes);
    auto combined = a.bytes;
    combined.push_back(30);
    combined.insert(combined.end(), b.bytes.begin(), b.bytes.end());
    dump(out / "ab.dat", combined);
    {
        std::ofstream f(out / "curriculum.sg", std::ios::binary);
        f << "SGCURRICULUM2\n"
          << a_steps << " \"a.dat\" 1 all\n"
          << a_steps + b_steps << " \"ab.dat\" 1 new\n";
    }
    LiveCurriculum curriculum(out / "curriculum.sg");
    Args options = args;
    options.values["--lr"] = args.get("lr", "0.0003");
    options.values["--seed"] = std::to_string(seed);
    options.values["--replay"] = "reservoir";
    options.values["--replay-every"] = std::to_string(every);
    options.values["--replay-capacity"] = std::to_string(capacity);
    options.values["--consolidation"] = "si";
    options.values["--si-strength"] = "0";
    options.values["--speak-every"] = args.get("speak-every", "500");
    options.values["--tokens"] = args.get("tokens", "96");
    options.values["--graph"] = "1";
    const std::string prompt = "The bird ";
    bool fast = !args.get("fast").empty();
    State trained;
    double initial_a, initial_b, learned_a, learned_b, a_seconds;
    size_t parameters = Layout(q).n;
    {
        LiveEngine engine(q, chunk, fast);
        engine.root.w.put(initialize(q, engine.root.a, uint64_t(seed)));
        trained.meta[10] = trained.meta[11] = uint64_t(seed);
        trained.meta[16] = fast;
        configure_live(trained, options, a, prompt);
        curriculum.initialize(trained);
        engine.root.synapses->initialize(engine.root.w, 0, .001f);
        Model evaluator(q, 16, 128);
        evaluator.w.share(engine.root.w);
        initial_a = evaluate(evaluator, validation_a, batches);
        initial_b = evaluate(evaluator, validation_b, batches);
        save(out / "initial.ckpt", engine.root, trained);
        auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < a_steps; ++i) {
            live_tick(engine, a, trained, prompt);
            if (trained.meta[24] % 1000 == 0)
                std::cout << "retention common update=" << trained.meta[24] << '\n';
        }
        ck(cudaDeviceSynchronize());
        a_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        save(out / "common.ckpt", engine.root, trained);
        learned_a = evaluate(evaluator, validation_a, batches);
        learned_b = evaluate(evaluator, validation_b, batches);
    }
    std::ofstream report(out / "result.json");
    report << std::setprecision(10) << "{\"seed\":" << seed << ",\"parameters\":" << parameters
           << ",\"cell\":\"" << q.name() << "\",\"steps_a\":" << a_steps << ",\"steps_b\":" << b_steps
           << ",\"chunk\":" << chunk << ",\"fast_math\":" << (fast ? "true" : "false")
           << ",\"replay_every\":" << every << ",\"replay_capacity\":" << capacity
           << ",\"common_checkpoint_payload_hash\":\"" << trained.meta[15] << "\",\"train_a_hash\":\""
           << a.hash << "\",\"train_b_hash\":\"" << b.hash << "\",\"validation_a_hash\":\"" << va.hash
           << "\",\"validation_b_hash\":\"" << vb.hash
           << "\",\"evaluation_targets_per_domain\":" << 16ull * 128 * batches
           << ",\"initial_old_loss\":" << initial_a << ",\"initial_new_loss\":" << initial_b
           << ",\"after_a_old_loss\":" << learned_a << ",\"after_a_new_loss\":" << learned_b
           << ",\"common_training_seconds\":" << a_seconds << ",\"arms\":[";
    const std::vector<std::string> arms{
        "none",       "current", "reservoir", "reservoir_si", "reservoir_slow_core", "reservoir_low_lr",
        "none_low_lr"};
    uint64_t expected_observations = 0;
    for (size_t arm = 0; arm < arms.size(); ++arm) {
        LiveEngine engine(q, chunk, fast);
        State s;
        load(out / "common.ckpt", engine.root, s, true);
        require(s.meta[15] == trained.meta[15], "Retention arm has a different starting model");
        LiveCorpus data = curriculum.corpus(s);
        // Explicit experimental ablations, not a silently altered resume policy.
        // No-replay and current-window controls intentionally discard the old
        // reservoir; source cursor, weights, moments, SI trace and speech RNG stay.
        if (arm < 2 || arm == 6) {
            s.extra.resize(16);
            s.extra[1] = arm == 1 ? 2 : 0;
            s.extra[2] = arm == 1 ? every : 0;
            s.extra[3] = 0;
        }
        if (arm == 3) {
            engine.root.synapses->strength = strength;
            s.extra[10] = float_word(strength);
        }
        if (arm == 4)
            s.hp[6] = .25f;
        if (arm >= 5)
            curriculum.rate(s, s.hp[7] * .25f);
        ReplayMemory(s).validate(data, chunk);
        fs::path directory = out / arms[arm];
        fs::create_directories(directory);
        std::ofstream metrics(directory / "metrics.jsonl"),
            transcript(directory / "transcript.txt", std::ios::binary);
        require(bool(metrics) && bool(transcript), "Cannot write retention arm logs");
        curriculum.advance(engine, data, s, &metrics);
        require(data.first_document == a.docs.size(), "Retention online stream silently revisits old data");
        Model evaluator(q, 16, 128);
        evaluator.w.share(engine.root.w);
        uint64_t observed = s.meta[22], replay_pairs = s.extra[7], replay_updates = s.extra[6],
                 generated = s.meta[30];
        double seconds = 0, old_loss = learned_a, new_loss = learned_b;
        LiveLatency ticks, speech_ticks;
        for (int i = 0; i < b_steps; ++i) {
            auto start = std::chrono::steady_clock::now();
            auto result = live_tick(engine, data, s, prompt);
            ck(cudaDeviceSynchronize());
            double ms =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
            seconds += ms * .001;
            (result.speech.empty() ? ticks : speech_ticks).add(ms);
            if (!result.speech.empty())
                transcript << "\n[update " << s.meta[24] << "]\n" << result.speech << '\n';
            require(s.meta[19] >= a.docs.size(), "Old data entered the new online stream");
            if ((i + 1) % 1000 == 0 || i + 1 == b_steps) {
                old_loss = evaluate(evaluator, validation_a, batches);
                new_loss = evaluate(evaluator, validation_b, batches);
                metrics << std::setprecision(10) << "{\"phase_b_update\":" << i + 1
                        << ",\"old_loss\":" << old_loss << ",\"new_loss\":" << new_loss
                        << ",\"forgetting\":" << old_loss - learned_a << "}\n";
                metrics.flush();
                std::cout << arms[arm] << " update=" << i + 1 << " old_loss=" << old_loss
                          << " new_loss=" << new_loss << '\n';
            }
        }
        uint64_t delta_observed = s.meta[22] - observed;
        if (!arm)
            expected_observations = delta_observed;
        require(delta_observed == expected_observations,
                "Retention arms saw different online source exposure");
        save(directory / "latest.ckpt", engine.root, s);
        report << (arm ? "," : "") << "{\"name\":\"" << arms[arm] << "\",\"old_loss\":" << old_loss
               << ",\"new_loss\":" << new_loss << ",\"old_forgetting\":" << old_loss - learned_a
               << ",\"new_transfer_change\":" << new_loss - learned_b
               << ",\"observed_pairs\":" << delta_observed
               << ",\"replay_pairs\":" << s.extra[7] - replay_pairs
               << ",\"replay_updates\":" << s.extra[6] - replay_updates
               << ",\"generated_bytes\":" << s.meta[30] - generated << ",\"core_scale\":" << s.hp[6]
               << ",\"si_strength\":" << engine.root.synapses->strength << ",\"learning_rate\":" << s.hp[0]
               << ",\"update_and_speech_seconds\":" << seconds
               << ",\"observed_pairs_per_second\":" << delta_observed / seconds
               << ",\"ordinary_tick_p50_ms\":" << ticks.percentile(.5)
               << ",\"ordinary_tick_p95_ms\":" << ticks.percentile(.95)
               << ",\"speech_tick_p95_ms\":" << speech_ticks.percentile(.95) << ",\"saved_payload_hash\":\""
               << s.meta[15] << "\"}";
        report.flush();
    }
    report << "],\"matched_online_exposure\":true,\"si_history_tracked_in_all_arms\":true,"
              "\"generated_bytes_are_not_targets\":true,\"new_online_scope_excludes_old_documents\":true}\n";
    report.close();
    require(bool(report), "Cannot finish retention report");
    std::cout << "Retention experiment saved: " << (out / "result.json").string() << '\n';
}
