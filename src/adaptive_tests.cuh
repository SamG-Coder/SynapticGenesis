#pragma once
#include "live.cuh"
void adaptive_test(const Args &args, int cell = 2) {
    args.allow({"out"});
    fs::path out = args.get("out", cell == 3 ? "reports/trace-tests" : "reports/adaptive-tests");
    fs::create_directories(out);
    Config q{32, 64, 2, cell};
    Model gradient(q, 2, 16);
    auto weights = initialize(q, gradient.a, 123);
    gradient.w.put(weights);
    std::vector<float> initial(size_t(q.l) * 2 * 2 * q.h);
    for (int l = 0; l < q.l; ++l)
        for (int j = 0; j < 2 * q.h; ++j) {
            initial[l * 4 * q.h + j] = 1.7f * std::sin(float(j + l * q.h) * .39f);
            initial[l * 4 * q.h + 2 * q.h + j] =
                q.traced() ? .4f * std::sin(float(j) * .17f) : .35f + .2f * std::cos(float(j) * .17f);
        }
    gradient.membranes(initial);
    std::vector<int> x(32), y(32);
    for (int i = 0; i < 32; ++i) {
        x[i] = (i * 37 + 91) % 256;
        y[i] = (i * 19 + 13) % 256;
    }
    float loss = gradient.forward(x, &y, true);
    dump(out / "weights.f32", weights);
    dump(out / "initial_state.f32", initial);
    dump(out / "final_state.f32", gradient.membranes());
    dump(out / "logits.f32", gradient.logits.host());
    dump(out / "inputs.i32", x);
    dump(out / "targets.i32", y);
    gradient.backward();
    dump(out / "gradients.f32", gradient.g.host());
    gradient.backward(1);
    dump(out / "gradients_regularized.f32", gradient.g.host());
    gradient.backward();
    gradient.update(1, .001f, 0, 0);
    dump(out / "updated.f32", gradient.w.host());
    std::ofstream conf(out / "fixture.json");
    conf << "{\"channels\":32,\"hidden\":64,\"layers\":2,\"cell\":" << cell
         << ",\"batch\":2,\"context\":16,\"loss\":" << std::setprecision(10) << loss
         << ",\"parameters\":" << gradient.a.n << "}";
    conf.close();

    Model chunk(q, 1, 4), step(q, 1, 1), captured(q, 1, 1);
    auto state = chunk.membranes();
    for (int l = 0; l < q.l; ++l) {
        std::copy_n(initial.data() + l * 4 * q.h, q.h, state.data() + l * 2 * q.h);
        std::copy_n(initial.data() + l * 4 * q.h + 2 * q.h, q.h, state.data() + l * 2 * q.h + q.h);
    }
    for (Model *m : {&chunk, &step, &captured}) {
        m->w.put(weights);
        m->membranes(state);
    }
    GraphDecoder graph(captured);
    require(maxdiff(state, captured.membranes()) == 0, "Adaptive capture changed incoming state");
    float chunk_error = 0, graph_error = 0;
    for (int t = 0; t < 32; t += 4) {
        chunk.forward(std::vector<int>(x.begin() + t, x.begin() + t + 4), nullptr, true);
        auto logits = chunk.logits.host();
        for (int j = 0; j < 4; ++j) {
            step.forward({x[t + j]}, nullptr, true);
            chunk_error = std::max(
                chunk_error, maxdiff(step.logits.host(), std::vector<float>(logits.begin() + j * 256,
                                                                            logits.begin() + (j + 1) * 256)));
            graph_error = std::max(graph_error, maxdiff(step.logits.host(), graph.step(x[t + j])));
        }
        chunk_error = std::max(chunk_error, maxdiff(chunk.membranes(), step.membranes()));
        graph_error = std::max(graph_error, maxdiff(captured.membranes(), step.membranes()));
    }
    require(chunk_error < 3e-5 && graph_error < 3e-5, "Adaptive streaming/capture disagrees");

    // Matched initialization, and a secondary-path-disabled negative control.
    Config oldq{q.c, q.h, q.l, 1};
    Model old(oldq, 1, 1);
    auto oldw = initialize(oldq, old.a, 123), disabled = weights;
    require(std::equal(oldw.begin(), oldw.begin() + q.c * 256, weights.begin()), "Embedding RNG changed");
    for (int l = 0; l < q.l; ++l) {
        auto a = old.a.layers[l], b = step.a.layers[l];
        require(std::equal(oldw.begin() + a.gain, oldw.begin() + a.leak + q.h, weights.begin() + b.gain),
                "Common block initialization differs");
        std::fill_n(disabled.data() + b.adapt_scale, q.h, -100.f);
    }
    require(std::equal(oldw.begin() + old.a.final_gain, oldw.end(), weights.begin() + step.a.final_gain),
            "Readout RNG changed");
    old.w.put(oldw);
    step.w.put(disabled);
    step.reset();
    float disabled_error = 0;
    for (int byte : x) {
        old.forward({byte}, nullptr, true);
        step.forward({byte}, nullptr, true);
        disabled_error = std::max(disabled_error, maxdiff(old.logits.host(), step.logits.host()));
    }
    require(disabled_error < 3e-5, "Disabled adaptation does not recover LIF output");

    std::string text = "A girl counts the seeds.\x1eThe child sees the bird.";
    dump(out / "corpus.dat", std::vector<char>(text.begin(), text.end()));
    LiveCorpus data(out / "corpus.dat");
    Args settings = args;
    settings.values["--replay"] = "reservoir";
    settings.values["--replay-capacity"] = "5";
    settings.values["--replay-every"] = "2";
    settings.values["--graph"] = "1";
    State s;
    configure_live(s, settings, data, "A");
    s.meta[26] = 3;
    s.meta[27] = 7;
    s.hp[0] = .001f;
    LiveEngine first(q, 8, false), second(q, 8, false);
    first.root.w.put(weights);
    for (int i = 0; i < 5; ++i)
        live_tick(first, data, s, "A");
    save(out / "resume.ckpt", first.root, s);
    State restored;
    load(out / "resume.ckpt", second.root, restored, true);
    require(s.meta[1] == uint64_t(cell) && s.meta[18] == uint64_t(q.l * q.h * 2),
            "Secondary-state checkpoint layout incorrect");
    require(maxdiff(first.root.membranes(), second.root.membranes()) == 0, "Adaptive state was not saved");
    for (int i = 0; i < 19; ++i) {
        auto a = live_tick(first, data, s, "A"), b = live_tick(second, data, restored, "A");
        require(a.speech == b.speech && s.extra == restored.extra, "Adaptive resumed speech/replay differs");
    }
    float resume_error = maxdiff(first.root.w.host(), second.root.w.host());
    resume_error = std::max(resume_error, maxdiff(first.root.m.host(), second.root.m.host()));
    resume_error = std::max(resume_error, maxdiff(first.root.v.host(), second.root.v.host()));
    resume_error = std::max(resume_error, maxdiff(first.root.membranes(), second.root.membranes()));
    require(resume_error < 3e-5 && s.meta == restored.meta && s.meta[22] == 184,
            "Adaptive resumed model/cursor differs");
    auto before = first.root.membranes();
    auto &replay = first.replay_view(4);
    std::vector<int> rx{65, 32, 107, 105}, ry{32, 107, 105, 110};
    replay.forward(rx, &ry);
    replay.backward();
    replay.update(int(++s.meta[7]), .001f);
    require(maxdiff(before, first.root.membranes()) == 0, "Replay overwrote adaptive live state");
    first.root.reset();
    for (float value : first.root.membranes())
        require(value == 0, "Adaptive reset left state behind");
    bool rejected = false;
    try {
        load(out / "resume.ckpt", old, restored, true);
    } catch (const std::exception &) {
        rejected = true;
    }
    require(rejected, "Adaptive checkpoint silently loaded as LIF");
    // These architectures have equal-sized layouts but different state semantics.
    Model other_secondary(Config{q.c, q.h, q.l, cell == 3 ? 2 : 3}, 1, 8);
    rejected = false;
    try {
        load(out / "resume.ckpt", other_secondary, restored, true);
    } catch (const std::exception &) {
        rejected = true;
    }
    require(rejected, "ALIF and trace checkpoint semantics were silently mixed");
    if (q.traced()) {
        rejected = false;
        try {
            step.forward({65}, nullptr, true, true);
        } catch (const std::exception &) {
            rejected = true;
        }
        require(rejected, "Trace emission was treated as ternary spikes");
    }
    std::ofstream result(out / "native.json");
    result << "{\"passed\":true,\"cell\":" << cell << ",\"chunk_max_error\":" << chunk_error
           << ",\"graph_max_error\":" << graph_error
           << ",\"disabled_adaptation_lif_max_error\":" << disabled_error
           << ",\"resume_max_error\":" << resume_error
           << ",\"equal_size_wrong_cell_rejected\":true,\"resumed_speech_identical\":true,\"replay_state_"
              "isolated\":true,\"common_initial_weights_"
              "identical\":true}\n";
    std::cout << "PASS " << (q.traced() ? "trace" : "adaptive")
              << ": oracle fixtures, stream/graph parity, LIF negative control, full "
                 "state/replay/speech resume. Max resume error "
              << resume_error << "\n";
}
