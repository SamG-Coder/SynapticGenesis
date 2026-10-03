// Device acceptance checks for the experimental policy; run only on an idle GPU.
// Independent full-model derivatives are checked afterwards by tests/oracle.py.
#pragma once
namespace membrane_learning_test {
void gradients(const fs::path &out, int cell, int batch) {
    fs::create_directories(out);
    Config q{32, 64, 2, cell};
    Model m(q, batch, 16);
    auto weights = initialize(q, m.a, 123);
    for (const auto &p : m.a.layers) {
        if (q.gated()) {
            for (int i = 0; i < q.h * q.c; ++i)
                weights[p.gate_w + i] = .03f * std::sin(float(i) * .19f);
            for (int j = 0; j < q.h; ++j)
                weights[p.gate_b + j] = .04f * std::cos(float(j) * .27f);
        }
    }
    activate_association_fixture(q, m.a, weights);
    m.w.put(weights);
    std::vector<float> initial(size_t(q.l) * q.recurrent_per_layer(batch));
    for (int l = 0; l < q.l; ++l) {
        size_t at = l * q.recurrent_per_layer(batch);
        for (int j = 0; j < batch * q.h; ++j) {
            initial[at + j] = 4.7f * std::sin(float(j + l * q.h) * .39f);
            initial[at + batch * q.h + j] = .4f * std::sin(float(j) * .17f);
        }
        if (q.associative())
            for (int j = 0; j < batch * association::matrix; ++j)
                initial[at + 2 * batch * q.h + j] = .1f * std::sin(float(j) * .13f);
    }
    m.membranes(initial);
    std::vector<int> x(m.N), y(m.N);
    std::vector<float> emphasis(m.N);
    for (int i = 0; i < m.N; ++i) {
        x[i] = (i * 37 + 91) % 256;
        y[i] = (i * 19 + 13) % 256;
        emphasis[i] = i % 5 == 0 ? 3.f : 1.f;
    }
    m.forward(x, &y, true);
    float loss = m.reweight_targets(emphasis);
    dump(out / "weights.f32", weights);
    dump(out / "initial_state.f32", initial);
    dump(out / "final_state.f32", m.membranes());
    dump(out / "logits.f32", m.logits.host());
    dump(out / "inputs.i32", x);
    dump(out / "targets.i32", y);
    dump(out / "target_weights.f32", emphasis);
    size_t outside = 0, unreachable = 0;
    for (const auto &layer : m.cache)
        for (float u : layer.u.host()) {
            outside += std::abs(u) > 1.5f;
            unreachable += std::abs(u) >= 2.f;
        }
    require(outside && unreachable, "Fixture must exercise nonzero direct gradients beyond surrogate support");
    m.backward();
    auto disabled = m.g.host();
    m.backward(0, 0, 1.75f);
    require(disabled == m.g.host(), "Explicit zero membrane cost changed gradients");
    m.backward(0, .2f, 1.5f);
    auto regularized = m.g.host();
    require(maxdiff(disabled, regularized) > 1e-5f, "Fixture did not distinguish the membrane objective");
    dump(out / "gradients.f32", regularized);
    m.backward(1, .2f, 1.5f);
    dump(out / "gradients_regularized.f32", m.g.host());
    m.backward(0, .2f, 1.5f);
    std::vector<float> first(m.a.n), second(m.a.n);
    for (size_t i = 0; i < m.a.n; ++i) {
        first[i] = .003f * std::sin(float(i) * .071f);
        second[i] = .002f + .001f * std::abs(std::cos(float(i) * .031f));
    }
    m.m.put(first); m.v.put(second);
    dump(out / "initial_m.f32", first);
    dump(out / "initial_v.f32", second);
    m.update(7, .001f, 0, 0);
    dump(out / "updated.f32", m.w.host());
    std::ofstream conf(out / "fixture.json", std::ios::binary);
    conf << std::setprecision(10) << "{\"channels\":32,\"hidden\":64,\"layers\":2,\"cell\":" << cell
         << ",\"batch\":" << batch << ",\"context\":16,\"loss\":" << loss << ",\"parameters\":" << m.a.n
         << ",\"adam_step\":7,\"membrane\":{\"cost\":0.2,\"band\":1.5},\"outside_band\":" << outside
         << ",\"outside_surrogate_support\":" << unreachable << "}\n";
    conf.close();
    require(bool(conf), "Cannot write gradient fixture");
}
void restart(const fs::path &out, int cell, bool si) {
    fs::create_directories(out);
    // Two synthetic documents produce full chunks, one-byte tails, stage
    // changes, stage replay and graph speech. They are test inputs only.
    std::vector<unsigned char> a{'a','b','c','d','e','f'}, b = a;
    b.insert(b.end(), {30,'g','h','i','j','k','l','m','n','o','p'});
    dump(out / "first.dat", a); dump(out / "second.dat", b);
    {
        std::ofstream f(out / "curriculum.sg", std::ios::binary);
        f << "SGCURRICULUM2\n4 \"first.dat\" 1 new\n12 \"second.dat\" 0.5 new\n";
    }
    LiveCurriculum curriculum(out / "curriculum.sg");
    LiveCorpus data(out / "first.dat"), resumed_data(out / "first.dat");
    Args args(0, nullptr);
    args.values = {{"--membrane-cost", ".02"}, {"--membrane-band", "1.5"},
                   {"--replay", "stage"}, {"--replay-every", "2"}, {"--replay-capacity", "8"},
                   {"--graph", "1"}, {"--speak-every", "3"}, {"--tokens", "4"}, {"--lr", ".001"}};
    if (si)
        args.values["--consolidation"] = "si";
    Config q{32,64,2,cell};
    LiveEngine full(q, 4, false), split(q, 4, false);
    State uninterrupted, checkpointed;
    uninterrupted.meta[10] = uninterrupted.meta[11] = 123;
    configure_live(uninterrupted, args, data, "a");
    curriculum.initialize(uninterrupted);
    full.root.w.put(initialize(q, full.root.a, 123));
    if (si)
        full.root.synapses->initialize(full.root.w, .1f, .001f);
    save(out / "initial.ckpt", full.root, uninterrupted);
    load(out / "initial.ckpt", split.root, checkpointed, true);
    auto tick = [&](LiveEngine &engine, LiveCorpus &corpus, State &s) {
        curriculum.advance(engine, corpus, s);
        return live_tick(engine, corpus, s, "a");
    };
    for (int i = 0; i < 5; ++i) {
        auto left = tick(full, data, uninterrupted), right = tick(split, resumed_data, checkpointed);
        require(left.speech == right.speech && left.loss == right.loss && left.replay_loss == right.replay_loss,
                "Identical regularized streams diverged");
    }
    save(out / "split.ckpt", split.root, checkpointed);
    LiveEngine resumed(q, 4, false);
    State restored;
    load(out / "split.ckpt", resumed.root, restored, true);
    resumed_data = curriculum.corpus(restored);
    for (int i = 5; i < 12; ++i) {
        auto left = tick(full, data, uninterrupted), right = tick(resumed, resumed_data, restored);
        require(left.speech == right.speech && left.loss == right.loss && left.replay_loss == right.replay_loss &&
                    left.gradient_norm == right.gradient_norm, "Restart changed regularized learning or speech");
    }
    save(out / "uninterrupted.ckpt", full.root, uninterrupted);
    save(out / "resumed.ckpt", resumed.root, restored);
    require(teachers::file_hash(out / "uninterrupted.ckpt") == teachers::file_hash(out / "resumed.ckpt"),
            "Resumed full checkpoint differs");
    auto loaded = read_checkpoint(out / "resumed.ckpt");
    require(loaded.state.meta[17] == 7 && live_version(loaded.state) == 5 &&
                loaded.state.extra[15] == 1 && loaded.state.extra[6] > 0 &&
                loaded.state.meta[30] == 16 && has_synaptic_history(loaded.state) == si,
            "Restart fixture did not exercise the intended paths");
}
void run(const fs::path &out) {
    require(!out.empty() && !fs::exists(out), "GPU checks need a fresh output directory");
    fs::create_directories(out);
    for (int cell = 3; cell <= 6; ++cell) {
        for (int batch : {1,2})
            gradients(out / ("cell-" + std::to_string(cell) + "-batch-" + std::to_string(batch)), cell, batch);
        for (bool si : {false,true})
            restart(out / ("restart-" + std::to_string(cell) + (si ? "-si" : "-plain")), cell, si);
    }
    std::ofstream f(out / "native-result.json", std::ios::binary);
    f << "{\"passed\":true,\"gradient_fixtures\":8,\"exact_restart_cases\":8,"
         "\"independent_oracle_still_required\":true,\"old_binary_disabled_parity_still_required\":true}\n";
    f.close();
    require(bool(f), "Cannot write GPU result");
}
} // namespace membrane_learning_test
