// Development must preserve learned history, including across process restarts.
#pragma once
void curriculum_test(const Args &args) {
    args.allow({"out"});
    fs::path out = args.get("out", "reports/curriculum-tests");
    fs::create_directories(out);
    auto write = [&](const char *name, const std::string &text) {
        dump(out / name, std::vector<char>(text.begin(), text.end()));
    };
    std::string first = "The child sees the bird in the tree. The bird sings to the child. ";
    std::string second = first + char(30) + "A seed needs water and sunlight. The child waters the garden. ";
    std::string third =
        second + char(30) + "Birds and plants live near the stream. Water flows down the hill. ";
    write("one.dat", first);
    write("two.dat", second);
    write("three.dat", third);
    write("curriculum.sg", "SGCURRICULUM1\n3 \"one.dat\" 1\n6 \"two.dat\" 0.5\n9 \"three.dat\" 0.25\n");
    LiveCurriculum curriculum(out / "curriculum.sg");
    write("earlier.sg", "SGCURRICULUM2\n3 \"one.dat\" 1 all\n6 \"two.dat\" 0.5 all\n");
    LiveCurriculum earlier(out / "earlier.sg");
    float error = 0;
    size_t cases = 0, transitions = 0;
    auto rejects = [](auto operation) {
        bool rejected = false;
        try {
            operation();
        } catch (const std::exception &) {
            rejected = true;
        }
        require(rejected, "Invalid curriculum or checkpoint accepted");
    };
    for (int cell : {1, 2, 3, 4})
        for (bool si : {false, true})
            for (int split : {2, 3, 4}) {
                Config q{8, 16, 2, cell};
                LiveEngine uninterrupted(q, 8, false), resumed(q, 8, false);
                uninterrupted.root.w.put(initialize(q, uninterrupted.root.a, 1337));
                Args options = args;
                options.values["--replay"] = "reservoir";
                options.values["--replay-capacity"] = "16";
                options.values["--replay-every"] = "2";
                options.values["--lr"] = "0.001";
                options.values["--graph"] = "1";
                options.values["--speak-every"] = "2";
                options.values["--tokens"] = "7";
                if (si) {
                    options.values["--consolidation"] = "si";
                    options.values["--si-strength"] = "0.02";
                }
                LiveCorpus data(out / "one.dat");
                State s;
                configure_live(s, options, data, "A");
                if (si)
                    uninterrupted.root.synapses->initialize(uninterrupted.root.w, .02f, .001f);
                // Attach the curriculum to an already learning v2/v3 runtime.
                live_tick(uninterrupted, data, s, "A");
                auto old_extra = s.extra;
                auto old_w = uninterrupted.root.w.host(), old_m = uninterrupted.root.m.host();
                curriculum.initialize(s);
                require(s.meta[24] == 1 && s.hp[7] == .001f && s.extra[5] == old_extra[5] &&
                            std::equal(old_extra.begin() + 16, old_extra.end(), s.extra.begin() + 16),
                        "Binding a curriculum erased replay history");
                require(maxdiff(old_w, uninterrupted.root.w.host()) == 0 &&
                            maxdiff(old_m, uninterrupted.root.m.host()) == 0,
                        "Binding a curriculum changed parameters or optimizer");
                float *weight_pointer = uninterrupted.root.w.p, *moment_pointer = uninterrupted.root.m.p;
                auto advance = [&](LiveEngine &engine, LiveCorpus &corpus, State &state) {
                    auto descriptors = state.extra;
                    auto weights = engine.root.w.host(), moments = engine.root.m.host();
                    auto membranes = engine.root.membranes();
                    size_t old_docs = corpus.docs.size();
                    uint64_t old_boundaries = engine.root.synapses->boundaries;
                    bool partial = !state.meta[25];
                    if (curriculum.advance(engine, corpus, state)) {
                        ++transitions;
                        require(state.meta[19] == old_docs && state.meta[20] == 0 && state.meta[25] == 1,
                                "Development did not start at the first new document");
                        require(descriptors.size() == state.extra.size() &&
                                    descriptors[4] == state.extra[4] &&
                                    std::equal(descriptors.begin() + 16, descriptors.end(),
                                               state.extra.begin() + 16),
                                "Development lost replay descriptors or RNG");
                        require(maxdiff(weights, engine.root.w.host()) == 0 &&
                                    maxdiff(moments, engine.root.m.host()) == 0 &&
                                    maxdiff(membranes, engine.root.membranes()) == 0,
                                "Development unexpectedly changed learned parameters or runtime state");
                        require(engine.root.synapses->boundaries == old_boundaries + (si && partial ? 1 : 0),
                                "Development consolidated the same trajectory twice");
                    }
                };
                while (s.meta[24] < uint64_t(split)) {
                    advance(uninterrupted, data, s);
                    live_tick(uninterrupted, data, s, "A");
                }
                fs::path checkpoint = out / ("resume-" + std::to_string(cell) + "-" + std::to_string(si) +
                                             "-" + std::to_string(split) + ".ckpt");
                save(checkpoint, uninterrupted.root, s);
                // Extension changes only the schedule identity, even midway
                // through a document. Different text versions may encode the
                // identical earlier policy. Rejections must be transactional.
                State extending = s;
                extending.extra[14] = earlier.hash;
                earlier.validate(extending);
                curriculum.extend(extending, earlier);
                require(extending.meta == s.meta && extending.hp == s.hp && extending.extra == s.extra &&
                            extending.synaptic == s.synaptic,
                        "Curriculum extension reset live history");
                State rejected = s;
                rejected.extra[14] = earlier.hash;
                State before_rejection = rejected;
                LiveCurriculum changed = curriculum;
                changed.stages[1].rate_scale = 1;
                rejects([&] { changed.extend(rejected, earlier); });
                require(rejected.meta == before_rejection.meta && rejected.hp == before_rejection.hp &&
                            rejected.extra == before_rejection.extra &&
                            rejected.synaptic == before_rejection.synaptic,
                        "Rejected extension changed live state");
                require(fs::file_size(checkpoint) == 288 + 12 * uninterrupted.root.a.n + 4 * s.meta[18] +
                                                         8 * s.extra.size() +
                                                         (si ? 12 * uninterrupted.root.a.n : 0),
                        "Curriculum allocated an incorrect optional SI payload");
                State restored;
                load(checkpoint, resumed.root, restored, true);
                LiveCorpus restored_data = curriculum.corpus(restored);
                while (s.meta[24] < 9) {
                    advance(uninterrupted, data, s);
                    advance(resumed, restored_data, restored);
                    auto a = live_tick(uninterrupted, data, s, "A");
                    auto b = live_tick(resumed, restored_data, restored, "A");
                    require(a.speech == b.speech && a.replayed == b.replayed,
                            "Resumed development changed speech or replay");
                    error = std::max(error, std::abs(a.loss - b.loss));
                }
                for (auto buffers : {std::make_pair(&uninterrupted.root.w, &resumed.root.w),
                                     std::make_pair(&uninterrupted.root.m, &resumed.root.m),
                                     std::make_pair(&uninterrupted.root.v, &resumed.root.v)})
                    error = std::max(error, maxdiff(buffers.first->host(), buffers.second->host()));
                error = std::max(error, maxdiff(uninterrupted.root.membranes(), resumed.root.membranes()));
                if (si)
                    error = std::max(
                        error, maxdiff(uninterrupted.root.synapses->host(), resumed.root.synapses->host()));
                require(s.meta == restored.meta && s.hp == restored.hp && s.extra == restored.extra,
                        "Resumed development changed cursors, policy or RNG");
                require(s.extra[15] == 2 && s.hp[0] == .00025f && s.hp[7] == .001f &&
                            uninterrupted.root.w.p == weight_pointer &&
                            uninterrupted.root.m.p == moment_pointer,
                        "Development replaced the runtime or used the wrong learning rate");
                curriculum.rate(s, .0002f);
                require(s.hp[0] == .00005f, "Base learning rate override lost the stage multiplier");
                State bad = s;
                bad.extra[15] = 4096;
                rejects([&] { curriculum.validate(bad); });
                bad = s;
                bad.meta[24] = 10;
                rejects([&] { curriculum.validate(bad); });
                bad = s;
                bad.extra[14] ^= 1;
                rejects([&] { curriculum.validate(bad); });
                // A payload edit must fail before it becomes training state.
                fs::path corrupt = out / "corrupt.ckpt";
                fs::copy_file(checkpoint, corrupt, fs::copy_options::overwrite_existing);
                {
                    std::fstream f(corrupt, std::ios::in | std::ios::out | std::ios::binary);
                    f.seekp(256 + 7 * 4);
                    float rate = .0009f;
                    f.write(reinterpret_cast<const char *>(&rate), 4);
                }
                rejects([&] { load(corrupt, resumed.root, restored, true); });
                ++cases;
            }
    require(error < 3e-5f, "Curriculum resume changed learned state beyond tolerance");
    for (const std::string &invalid :
         {"changed" + second, first + "added without separator", first + char(30) + "x", first}) {
        write("invalid.dat", invalid);
        rejects([&] {
            LiveCurriculum::validate_append(LiveCorpus(out / "one.dat"), LiveCorpus(out / "invalid.dat"));
        });
    }
    LiveCorpus final_data(out / "three.dat"), holdout(out / "one.dat");
    rejects([&] { validate_live_holdout(final_data, holdout); });
    // Future editions are part of the identity even before they are introduced.
    write("three.dat", third + "new");
    LiveCurriculum changed(out / "curriculum.sg");
    require(changed.hash != curriculum.hash, "Future data edition was not bound to the schedule");
    write("three.dat", third);
    std::ofstream report(out / "native.json");
    report
        << "{\"passed\":true,\"resume_cases\":" << cases << ",\"stage_transitions_checked\":" << transitions
        << ",\"resume_max_error\":" << error
        << ",\"cells\":[1,2,3,4],\"optional_si\":true,\"graph_speech_identical\":true,"
           "\"replay_preserved\":true,\"same_gpu_allocations\":true,\"future_source_identity_checked\":true,"
           "\"heldout_document_rejected\":true,\"base_rate_override_checked\":true,"
           "\"extension_preserves_history\":true,\"extension_rejection_transactional\":true}\n";
    std::cout << "PASS curriculum: " << cases << " resume cases, max error=" << error << "\n";
}
