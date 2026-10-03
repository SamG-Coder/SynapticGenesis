// Bounded capacity measurements through the production learner and speaker.
#define main synapticgenesis_cli_entry
#include "../src/spike_lm.cu"
#undef main

namespace capacity_probe {
double seconds(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}
uint64_t arrays(const LiveEngine &engine) {
    uint64_t bytes = explicit_model_bytes(engine.root);
    const uint64_t shared_parameters = 12 * engine.root.a.n;
    const uint64_t shared_recurrence = 4 * engine.root.q.recurrent_per_layer() * engine.root.q.l;
    bytes += explicit_model_bytes(engine.speaker) - shared_parameters - shared_recurrence;
    for (const auto &view : engine.tails)
        bytes += explicit_model_bytes(*view.second) - shared_parameters - shared_recurrence;
    for (const auto &view : engine.replay_views)
        bytes += explicit_model_bytes(*view.second) - shared_parameters;
    return bytes;
}
void run(const Args &args) {
    args.allow({"data", "out", "channels", "hidden", "layers", "cell", "chunk", "seed", "lr",
                "warmup", "updates", "rounds", "speak-every", "tokens", "top-k", "temperature",
                "prompt", "replay", "replay-every", "replay-capacity", "replay-seed", "graph", "fast",
                "decode-bytes", "save"});
    fs::path out = args.get("out");
    require(!out.empty() && !fs::exists(out), "Capacity probe requires a fresh output directory");
    Config q{args.num("channels", 256), args.num("hidden", 512), args.num("layers", 4), cell_version(args)};
    int chunk = args.num("chunk", 128), warmup = args.num("warmup", 256), updates = args.num("updates", 512);
    int rounds = args.num("rounds", 3), seed = args.num("seed", 1337), decode = args.num("decode-bytes", 1024);
    require(chunk >= 1 && chunk <= 4096 && warmup >= 0 && warmup <= 4096 && updates >= 1 &&
                updates <= 8192 && rounds >= 1 && rounds <= 10 && seed > 0 && decode >= 1 && decode <= 10000,
            "Invalid capacity probe bounds");
    require(args.get("replay", "none") != "stage", "Capacity probe uses a single admitted source edition");
    const std::string prompt = args.get("prompt", "The bird ");
    require(!prompt.empty() && prompt.size() <= 65536, "Invalid prompt");
    LiveCorpus data(args.get("data"));
    State state;
    state.meta[10] = state.meta[11] = seed;
    configure_live(state, args, data, prompt);
    state.meta[16] = !args.get("fast").empty();
    ck(cudaFree(nullptr));
    size_t free_before = 0, total = 0, free_now = 0;
    ck(cudaMemGetInfo(&free_before, &total));
    LiveEngine engine(q, chunk, state.meta[16] != 0);
    engine.root.w.put(initialize(q, engine.root.a, uint64_t(seed)));
    data.validate(state);
    ReplayMemory memory(state);
    memory.validate(data, chunk);
    fs::create_directories(out);
    std::ofstream report(out / "rounds.jsonl", std::ios::binary);
    std::ofstream speech(out / "speech.bin", std::ios::binary);
    require(bool(report) && bool(speech), "Cannot create capacity output");
    double loss_sum = 0;
    uint64_t learned_pairs = 0;
    auto tick = [&]() {
        auto result = live_tick(engine, data, state, prompt);
        ck(cudaDeviceSynchronize());
        require(std::isfinite(result.loss) && std::isfinite(result.gradient_norm) &&
                    std::isfinite(result.replay_loss), "Nonfinite capacity learning result");
        speech.write(result.speech.data(), std::streamsize(result.speech.size()));
        loss_sum += result.loss * result.observed;
        learned_pairs += result.observed;
    };
    auto started = std::chrono::steady_clock::now();
    for (int i = 0; i < warmup; ++i)
        tick();
    double warmup_seconds = seconds(started);
    size_t minimum_boundary_free = free_before;
    for (int round = 0; round < rounds; ++round) {
        uint64_t observed = state.meta[22], replayed = memory.pairs(), generated = state.meta[30];
        loss_sum = 0;
        learned_pairs = 0;
        LiveLatency ticks;
        started = std::chrono::steady_clock::now();
        for (int i = 0; i < updates; ++i) {
            auto tick_start = std::chrono::steady_clock::now();
            tick();
            ticks.add(seconds(tick_start) * 1000);
        }
        double elapsed = seconds(started);
        ck(cudaMemGetInfo(&free_now, &total));
        minimum_boundary_free = std::min(minimum_boundary_free, free_now);
        memory.validate(data, chunk);
        data.validate(state);
        report << std::setprecision(17) << "{\"round\":" << round << ",\"seconds\":" << elapsed
               << ",\"source_pairs\":" << state.meta[22] - observed << ",\"replay_pairs\":" << memory.pairs() - replayed
               << ",\"generated_bytes\":" << state.meta[30] - generated << ",\"loss\":" << loss_sum / learned_pairs
               << ",\"tick_p95_ms\":" << ticks.percentile(.95) << ",\"explicit_arrays_bytes\":" << arrays(engine)
               << ",\"boundary_free_bytes\":" << free_now << ",\"tail_views\":" << engine.tails.size()
               << ",\"replay_views\":" << engine.replay_views.size() << "}\n";
        report.flush();
        std::cout << "round=" << round << " parameters=" << engine.root.a.n << " new_bytes_per_second="
                  << (state.meta[22] - observed) / elapsed << " arrays_mib=" << arrays(engine) / 1048576.0 << '\n';
    }
    // Preserve an optional fixture before standalone decode changes recurrence.
    if (!args.get("save").empty()) {
        save(out / "latest.ckpt", engine.root, state);
        read_checkpoint(out / "latest.ckpt");
    }
    report.close();
    speech.close();
    require(bool(report) && bool(speech), "Capacity output flush failed");
    std::ofstream decoding(out / "decode.jsonl", std::ios::binary);
    require(bool(decoding), "Cannot create decode output");
    auto recurrent = engine.root.membranes();
    auto sampling_rng = state.meta[23];
    state.meta[27] = decode;
    std::string prior;
    for (int round = -1; round < rounds; ++round) {
        engine.root.membranes(recurrent);
        state.meta[23] = sampling_rng;
        started = std::chrono::steady_clock::now();
        auto generated = engine.speak(prompt, state);
        ck(cudaDeviceSynchronize());
        double elapsed = seconds(started);
        if (round >= 0) {
            require(generated == prior, "Repeated decode changed bytes at fixed weights/state/RNG");
            decoding << std::setprecision(17) << "{\"round\":" << round << ",\"generated_bytes\":" << decode
                     << ",\"prompt_bytes\":" << prompt.size() << ",\"seconds\":" << elapsed << "}\n";
        }
        prior = generated;
    }
    decoding.close();
    require(bool(decoding), "Decode output flush failed");
    std::ofstream summary(out / "result.json", std::ios::binary);
    summary << std::setprecision(17) << "{\"complete\":true,\"parameters\":" << engine.root.a.n
            << ",\"channels\":" << q.c << ",\"neurons_per_layer\":" << q.h << ",\"layers\":" << q.l
            << ",\"spiking_neurons\":" << q.h * q.l << ",\"recurrent_floats\":" << q.recurrent_per_layer() * q.l
            << ",\"chunk\":" << chunk << ",\"warmup_updates\":" << warmup << ",\"warmup_seconds\":" << warmup_seconds
            << ",\"measured_updates_per_round\":" << updates << ",\"rounds\":" << rounds
            << ",\"explicit_arrays_bytes\":" << arrays(engine) << ",\"baseline_free_bytes\":" << free_before
            << ",\"minimum_round_boundary_free_bytes\":" << minimum_boundary_free
            << ",\"source_hash\":\"" << data.hash << "\",\"generated_text_targets\":false,\"decode_repeated_bytes_exact\":true}\n";
    summary.close();
    require(bool(summary), "Capacity summary flush failed");
}
} // namespace capacity_probe
int main(int argc, char **argv) {
    try {
        std::cout.setf(std::ios::unitbuf);
        require(argc >= 2 && std::string(argv[1]) == "run", "Expected capacity probe run --options");
        capacity_probe::run(Args(argc, argv));
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
