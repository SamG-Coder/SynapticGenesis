// Early live-learning measurements with the unchanged production model/update.
#define main synapticgenesis_cli_entry
#include "../src/spike_lm.cu"
#undef main
#include "learning_scale_observer.cuh"

namespace early_learning {
void run(const Args &args) {
    args.allow({"out", "curriculum", "channels", "hidden", "layers", "seed", "chunk", "lr", "updates",
                "points", "core-scale", "replay", "replay-every", "replay-capacity", "replay-seed",
                "graph", "fast", "speak-every", "tokens", "top-k", "temperature", "prompt"});
    fs::path out = args.get("out");
    require(!out.empty() && !fs::exists(out), "Early-learning probe requires a fresh output directory");
    Config q{args.num("channels", 256), args.num("hidden", 1024), args.num("layers", 8), 6};
    int chunk = args.num("chunk", 128), seed = args.num("seed", 1337), updates = args.num("updates", 8176);
    require(seed > 0 && updates > 0 && updates <= 100000 && chunk >= 1 && chunk <= 4096, "Invalid probe bounds");
    auto selected = learning_scale::points(args.get("points", "1,32,128,512,2048,8176"), updates);
    std::string prompt = args.get("prompt", "The bird ");
    require(!prompt.empty() && prompt.size() <= 65536, "Invalid prompt");
    LiveCurriculum curriculum(args.get("curriculum"));
    require(uint64_t(updates) <= curriculum.stages.front().end_update,
            "Early-learning probe only covers the first curriculum stage");
    LiveCorpus data(curriculum.stages.front().corpus);
    State state;
    state.meta[10] = state.meta[11] = seed;
    configure_live(state, args, data, prompt);
    state.hp[0] = args.real("lr", .0003f);
    curriculum.initialize(state);
    curriculum.validate(state);
    curriculum.apply_feedback(data, 0);
    state.meta[16] = !args.get("fast").empty();
    LiveEngine engine(q, chunk, state.meta[16] != 0);
    engine.root.w.put(initialize(q, engine.root.a, uint64_t(seed)));
    data.validate(state);
    ReplayMemory memory(state);
    memory.validate(data, chunk);
    fs::create_directories(out);
    save(out / "initial.ckpt", engine.root, state);
    learning_scale::Observer observer(engine.root, out, selected);
    learning_scale::Moments norms;
    uint64_t clipped = 0;
    double loss_sum = 0;
    std::ofstream transcript(out / "speech.bin", std::ios::binary);
    require(bool(transcript), "Cannot create speech output");
    auto started = std::chrono::steady_clock::now();
    while (state.meta[24] < uint64_t(updates)) {
        require(!curriculum.advance(engine, data, state), "Unexpected curriculum transition");
        auto result = live_tick(engine, data, state, prompt, &observer);
        ck(cudaDeviceSynchronize());
        require(std::isfinite(result.loss) && std::isfinite(result.gradient_norm) && std::isfinite(result.replay_loss),
                "Nonfinite early-learning result");
        norms.add(result.gradient_norm);
        clipped += state.hp[2] > 0 && result.gradient_norm > state.hp[2];
        loss_sum += result.loss * result.observed;
        transcript.write(result.speech.data(), std::streamsize(result.speech.size()));
        observer.finish(result, state);
        if (state.meta[24] % 2048 == 0)
            std::cout << "source_update=" << state.meta[24] << " parameters=" << engine.root.a.n << '\n';
    }
    double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    require(observer.records == selected.size(), "Missing source observations");
    transcript.close();
    require(bool(transcript), "Speech output failed");
    data.validate(state);
    memory.validate(data, chunk);
    curriculum.validate(state);
    save(out / "latest.ckpt", engine.root, state);
    read_checkpoint(out / "latest.ckpt");
    std::ofstream report(out / "result.json", std::ios::binary);
    report << std::setprecision(17) << "{\"complete\":true,\"parameters\":" << engine.root.a.n
           << ",\"channels\":" << q.c << ",\"hidden\":" << q.h << ",\"layers\":" << q.l
           << ",\"seed\":" << seed << ",\"source_updates\":" << state.meta[24]
           << ",\"source_pairs\":" << state.meta[22] << ",\"global_updates\":" << state.meta[7]
           << ",\"replay_updates\":" << memory.updates() << ",\"replay_pairs\":" << memory.pairs()
           << ",\"generated_bytes\":" << state.meta[30] << ",\"core_scale\":" << memory.core_scale()
           << ",\"learning_rate\":" << state.hp[0] << ",\"source_loss\":" << loss_sum / state.meta[22]
           << ",\"source_gradient_norm_before_clip\":";
    norms.json(report);
    report << ",\"source_updates_clipped\":" << clipped << ",\"observations\":" << observer.records
           << ",\"instrumented_learning_seconds\":" << elapsed << ",\"observer_callback_seconds\":" << observer.seconds
           << ",\"shared_production_live_tick\":true,\"generated_text_targets\":false,\"reserved_tests_scored\":false}\n";
    report.close();
    require(bool(report), "Early-learning report failed");
    std::cout << "Early-learning probe complete at " << updates << " source observations\n";
}
} // namespace early_learning

int main(int argc, char **argv) {
    try {
        std::cout.setf(std::ios::unitbuf);
        require(argc >= 2, "Expected run or host-test");
        if (std::string(argv[1]) == "host-test") {
            require(argc == 2, "host-test takes no options");
            learning_scale::host_test();
        } else {
            require(std::string(argv[1]) == "run", "Expected run or host-test");
            early_learning::run(Args(argc, argv));
        }
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
