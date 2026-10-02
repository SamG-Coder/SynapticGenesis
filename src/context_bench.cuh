// Evaluate identical target windows with realistic preceding state and state
// ablations. No parameters or optimizer state are changed by this command.
#pragma once
void context_bench(const Args &args) {
    args.allow({"checkpoint", "data", "prefix", "context", "batch", "batches", "output"});
    State s = header(args.get("checkpoint"));
    Config q = checkpoint_config(s);
    int prefix = args.num("prefix", 512), T = args.num("context", 128), B = args.num("batch", 16),
        batches = args.num("batches", 64);
    require(prefix >= 1 && prefix <= 2048 && T >= 1 && T <= 2048 && B >= 1 && B <= 256 && batches >= 1 &&
                batches <= 10000,
            "Invalid context benchmark limits");
    Data data(args.get("data"), prefix + T);
    Model history(q, B, prefix), target(q, B, T);
    load(args.get("checkpoint"), history, s);
    target.share_runtime(history);
    std::vector<int> full_x, full_y, px, x, y;
    uint64_t rng = 712367;
    double warm_loss = 0, reset_loss = 0, no_adaptation_loss = 0, adaptation_sum = 0;
    for (int batch = 0; batch < batches; ++batch) {
        data.batch(B, rng, full_x, full_y);
        px.clear();
        x.clear();
        y.clear();
        for (int b = 0; b < B; ++b) {
            auto at = full_x.begin() + b * (prefix + T);
            px.insert(px.end(), at, at + prefix);
            x.insert(x.end(), at + prefix, at + prefix + T);
            auto yt = full_y.begin() + b * (prefix + T) + prefix;
            y.insert(y.end(), yt, yt + T);
        }
        history.reset();
        history.forward(px, nullptr, true);
        auto state = history.membranes();
        warm_loss += target.forward(x, &y, true);
        if (q.secondary()) {
            target.membranes(state);
            for (auto &c : target.cache) {
                auto a = c.adapt_state.host();
                adaptation_sum += std::accumulate(a.begin(), a.end(), 0.0) / (double(q.l) * a.size());
                c.adapt_state.zero();
            }
            no_adaptation_loss += target.forward(x, &y, true);
        }
        reset_loss += target.forward(x, &y, false);
    }
    std::string output = args.get("output", "reports/context-bench.json");
    if (!fs::path(output).parent_path().empty())
        fs::create_directories(fs::path(output).parent_path());
    std::ofstream f(output);
    f << std::setprecision(10) << "{\"cell\":\"" << q.name() << "\",\"prefix_bytes\":" << prefix
      << ",\"context_bytes\":" << T << ",\"evaluated_target_bytes\":" << uint64_t(B) * T * batches
      << ",\"corpus_hash\":\"" << data.hash << "\",\"warm_context_loss\":" << warm_loss / batches
      << ",\"all_state_reset_loss\":" << reset_loss / batches << ",\"adaptation_reset_loss\":"
      << (q.adaptive() ? std::to_string(no_adaptation_loss / batches) : "null")
      << ",\"trace_reset_loss\":" << (q.traced() ? std::to_string(no_adaptation_loss / batches) : "null")
      << ",\"mean_adaptation_at_boundary\":"
      << (q.adaptive() ? std::to_string(adaptation_sum / batches) : "null")
      << ",\"parameters_unchanged\":true}\n";
    std::cout << q.name() << " prefix=" << prefix << " warm_loss=" << warm_loss / batches
              << " reset_loss=" << reset_loss / batches
              << " adaptation_reset=" << no_adaptation_loss / batches << "\n";
}
