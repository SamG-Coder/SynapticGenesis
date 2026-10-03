// Controlled delayed-cue experiment. These synthetic examples train disposable
// benchmark models only, never the foundation-language model.
#pragma once
__global__ void query_only_grad(float *g, int N, int T) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N * 256)
        g[i] = (i / 256) % T == T - 1 ? g[i] * T : 0.f;
}
void cue_batch(int B, int T, uint64_t &rng, std::vector<int> &x, std::vector<int> &y) {
    x.resize(size_t(B) * T);
    y.resize(x.size());
    // Balance each batch, randomly permute label-to-row assignment, and sample
    // distractors independently of the cue. The network cannot see row indices.
    std::vector<int> labels(B);
    for (int b = 0; b < B; ++b)
        labels[b] = 'A' + b % 2;
    for (int b = B - 1; b > 0; --b)
        std::swap(labels[b], labels[size_t(rnd(rng) % (b + 1))]);
    for (int b = 0; b < B; ++b) {
        x[b * T] = labels[b];
        for (int t = 1; t < T - 1; ++t)
            x[b * T + t] = '0' + int(rnd(rng) % 4);
        x[b * T + T - 1] = '?';
        std::fill_n(y.data() + b * T, T, labels[b]);
    }
}
float query_loss(Model &m, const std::vector<int> &x, const std::vector<int> &y) {
    m.forward(x, &y);
    query_only_grad<<<(m.N * 256 + 255) / 256, 256>>>(m.dlogits.p, m.N, m.T);
    auto losses = m.losses.host();
    double sum = 0;
    for (int b = 0; b < m.B; ++b)
        sum += losses[b * m.T + m.T - 1];
    return float(sum / m.B);
}
struct CueResult {
    double loss = 0, accuracy = 0, no_adaptation_accuracy = 0, no_state_accuracy = 0;
};
CueResult cue_evaluate(Model &owner, int batches) {
    Model prefix(owner, owner.T - 1, ModelViewState::independent);
    Model query(prefix, 1, ModelViewState::shared);
    CueResult result;
    uint64_t rng = 8721349;
    std::vector<int> x, y, px, qx(owner.B, '?'), qy(owner.B);
    auto accuracy = [&](const std::vector<float> &logits) {
        int correct = 0;
        for (int b = 0; b < owner.B; ++b) {
            auto first = logits.begin() + b * 256;
            if (std::max_element(first, first + 256) - first == qy[b])
                ++correct;
        }
        return double(correct) / owner.B;
    };
    for (int i = 0; i < batches; ++i) {
        cue_batch(owner.B, owner.T, rng, x, y);
        px.clear();
        for (int b = 0; b < owner.B; ++b) {
            px.insert(px.end(), x.begin() + b * owner.T, x.begin() + (b + 1) * owner.T - 1);
            qy[b] = y[b * owner.T];
        }
        prefix.reset();
        prefix.forward(px, nullptr, true);
        auto state = prefix.membranes();
        result.loss += query.forward(qx, &qy, true);
        result.accuracy += accuracy(query.logits.host());
        if (owner.q.secondary()) {
            query.membranes(state);
            for (auto &c : query.cache)
                c.adapt_state.zero();
            query.forward(qx, nullptr, true);
            result.no_adaptation_accuracy += accuracy(query.logits.host());
        }
        query.reset();
        query.forward(qx, nullptr, true);
        result.no_state_accuracy += accuracy(query.logits.host());
    }
    result.loss /= batches;
    result.accuracy /= batches;
    result.no_adaptation_accuracy /= batches;
    result.no_state_accuracy /= batches;
    return result;
}
void memory_bench(const Args &args) {
    args.allow(
        {"out", "cell", "delay", "steps", "seed", "batch", "channels", "hidden", "layers", "lr", "fast"});
    fs::path out = args.get("out", "runs/memory-bench");
    if (fs::exists(out / "result.json"))
        throw std::runtime_error("Memory experiment exists; choose a new output directory");
    fs::create_directories(out);
    int delay = args.num("delay", 128), steps = args.num("steps", 2000), B = args.num("batch", 32);
    int seed = args.num("seed", 1337);
    require(delay >= 1 && delay <= 2046 && steps >= 1 && steps <= 100000 && B >= 2 && B <= 256 &&
                B % 2 == 0 && seed > 0,
            "Invalid memory benchmark limits");
    Config q{args.num("channels", 64), args.num("hidden", 128), args.num("layers", 2), cell_version(args)};
    Model m(q, B, delay + 2);
    m.w.put(initialize(q, m.a, uint64_t(seed)));
    bool fast = !args.get("fast").empty();
    m.fast(fast);
    uint64_t rng = uint64_t(seed) + 90001;
    float lr = args.real("lr", .001f);
    require(lr > 0 && lr <= .1f, "Invalid memory learning rate");
    auto before = cue_evaluate(m, 32);
    std::ofstream metrics(out / "metrics.jsonl");
    std::vector<int> x, y;
    auto started = std::chrono::steady_clock::now();
    double loss_sum = 0;
    for (int step = 1; step <= steps; ++step) {
        cue_batch(B, delay + 2, rng, x, y);
        loss_sum += query_loss(m, x, y);
        m.backward();
        m.update(step, lr, 0, 1);
        if (step % 250 == 0 || step == steps) {
            int count = step % 250 ? step % 250 : 250;
            metrics << "{\"step\":" << step << ",\"query_loss\":" << loss_sum / count
                    << ",\"spike_rate\":" << m.rate() << "}\n";
            metrics.flush();
            std::cout << q.name() << " delay=" << delay << " step=" << step
                      << " query_loss=" << loss_sum / count << "\n";
            loss_sum = 0;
        }
    }
    ck(cudaDeviceSynchronize());
    double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    auto after = cue_evaluate(m, 64);
    require(std::abs(after.no_state_accuracy - .5) < 1e-10,
            "Erasing all cue state did not recover chance; task leaks labels");
    State s;
    s.meta[7] = steps;
    s.meta[10] = rng;
    s.meta[11] = seed;
    s.hp[0] = lr;
    save(out / "synthetic-only.ckpt", m, s);
    std::ofstream f(out / "result.json");
    f << std::setprecision(10) << "{\"cell\":\"" << q.name() << "\",\"seed\":" << seed
      << ",\"delay\":" << delay << ",\"steps\":" << steps << ",\"batch\":" << B << ",\"parameters\":" << m.a.n
      << ",\"initial_loss\":" << before.loss << ",\"final_loss\":" << after.loss
      << ",\"accuracy\":" << after.accuracy << ",\"adaptation_erased_accuracy\":"
      << (q.adaptive() ? std::to_string(after.no_adaptation_accuracy) : "null")
      << ",\"trace_erased_accuracy\":" << (q.traced() ? std::to_string(after.no_adaptation_accuracy) : "null")
      << ",\"all_state_erased_accuracy\":" << after.no_state_accuracy
      << ",\"evaluation_sequences\":" << 64 * B << ",\"training_seconds\":" << seconds
      << ",\"synthetic_only\":true}\n";
    std::cout << "memory result accuracy=" << after.accuracy
              << " secondary_state_erased=" << after.no_adaptation_accuracy
              << " all_state_erased=" << after.no_state_accuracy << " seconds=" << seconds << "\n";
}
