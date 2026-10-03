// Measure source-update interference without changing the production trajectory.
#define main synapticgenesis_cli_entry
#include "../src/spike_lm.cu"
#undef main
#include "replay_priority_scoring.cuh"

namespace replay_priority {
void run(const Args &args) {
    args.allow({"checkpoint", "curriculum", "out", "updates", "every", "per-group", "seed",
                "prompt", "audit-state", "reverse-candidates", "snapshots", "graph-scoring"});
    fs::path source = args.get("checkpoint"), schedule_path = args.get("curriculum"), out = args.get("out");
    require(!out.empty() && !fs::exists(out), "Replay priority probe needs a fresh output directory");
    State state = read_checkpoint(source).state;
    require(state.meta[17] == 5 && state.meta[5] == 1 && !has_synaptic_history(state),
            "Probe requires grouped live replay without SI or frozen teachers");
    int updates = args.num("updates", 0), every = args.num("every", 32), per_group = args.num("per-group", 8);
    int seed = args.num("seed", 1337);
    int audit = args.num("audit-state", 0), reverse = args.num("reverse-candidates", 0),
        snapshots = args.num("snapshots", 0), graph = args.num("graph-scoring", 0);
    require((audit == 0 || audit == 1) && (reverse == 0 || reverse == 1) &&
                (snapshots == 0 || snapshots == 1) && (graph == 0 || graph == 1),
            "Diagnostic switches require 0 or 1");
    require(updates > 0 && uint64_t(updates) > state.meta[24] &&
                uint64_t(updates) - state.meta[24] <= 4096 &&
                every >= 0 && every <= 4096 && per_group > 0 && per_group <= 64 && seed > 0,
            "Invalid bounded probe limits");
    LiveCurriculum curriculum(schedule_path);
    curriculum.validate(state);
    require(uint64_t(updates) <= curriculum.stages.back().end_update, "Probe exceeds original curriculum");
    std::string prompt = args.get("prompt", "The bird ");
    require(!prompt.empty() && prompt.size() <= 65536 && state.meta[29] == hash_bytes(prompt.data(), prompt.size()),
            "Probe requires original speech prompt");
    const uint64_t source_identity = file_identity(source), schedule_identity = file_identity(schedule_path);
    auto config = checkpoint_config(state);
    LiveEngine engine(config, int(state.meta[6]), state.meta[16] != 0);
    load(source, engine.root, state, true);
    auto data = curriculum.corpus(state);
    data.validate(state);
    ReplayMemory(state).validate(data, engine.root.T);
    require(ReplayMemory(state).count() > 0, "Probe needs observed replay memories");
    auto replay_every = ReplayMemory(state).every();
    uint64_t replay_updates = uint64_t(updates) / replay_every - state.meta[24] / replay_every;
    require(uint64_t(updates) - state.meta[24] + replay_updates + state.meta[7] <= 1000000000ull,
            "Probe exceeds optimizer update limit");
    const uint64_t begin = state.meta[24], original_updates = state.meta[7], original_pairs = state.meta[22],
                   original_speech = state.meta[30];
    fs::create_directories(out);
    Observer observer(out, uint64_t(every), uint64_t(per_group), uint64_t(seed), audit, reverse, snapshots, graph);
    std::ofstream speech(out / "speech.txt", std::ios::binary);
    require(bool(speech), "Cannot create probe speech log");
    LiveLatency ticks, speaking_ticks, scoring_ticks;
    double max_tick_ms = 0;
    auto start = std::chrono::steady_clock::now();
    while (state.meta[24] < uint64_t(updates)) {
        bool measuring = every && state.meta[24] % uint64_t(every) == 0;
        auto tick_start = std::chrono::steady_clock::now();
        curriculum.advance(engine, data, state);
        auto result = live_tick(engine, data, state, prompt, every ? &observer : nullptr);
        ck(cudaDeviceSynchronize());
        double elapsed = seconds(tick_start);
        ticks.add(elapsed * 1000);
        max_tick_ms = std::max(max_tick_ms, elapsed * 1000);
        if (measuring)
            scoring_ticks.add(elapsed * 1000);
        if (!result.speech.empty()) {
            speaking_ticks.add(elapsed * 1000);
            speech << result.speech << '\n';
        }
    }
    double elapsed = seconds(start);
    observer.finish();
    speech.close();
    require(bool(speech), "Probe speech flush failed");
    curriculum.validate(state);
    ReplayMemory(state).validate(data, engine.root.T);
    save(out / "latest.ckpt", engine.root, state);
    read_checkpoint(out / "latest.ckpt");
    require(file_identity(source) == source_identity && file_identity(schedule_path) == schedule_identity,
            "Probe input checkpoint or schedule changed");
    LiveCurriculum reloaded(schedule_path);
    require(reloaded.hash == curriculum.hash, "Probe source edition changed");
    std::ofstream report(out / "result.json");
    report << std::setprecision(17) << "{\"complete\":true,\"parameters\":" << engine.root.a.n
           << ",\"start_observation\":" << begin << ",\"end_observation\":" << state.meta[24]
           << ",\"optimizer_updates\":" << state.meta[7] - original_updates
           << ",\"source_pairs\":" << state.meta[22] - original_pairs
           << ",\"generated_bytes\":" << state.meta[30] - original_speech
           << ",\"observe_every\":" << every << ",\"per_group\":" << per_group << ",\"candidate_seed\":" << seed
           << ",\"measurements\":" << observer.observations << ",\"score_forward_calls\":" << observer.forward_calls
           << ",\"graph_scoring\":" << (graph ? "true" : "false")
           << ",\"scored_pairs\":" << observer.scored_pairs << ",\"live_seconds\":" << elapsed
           << ",\"scoring_seconds\":" << observer.scoring_seconds << ",\"scorer_setup_seconds\":" << observer.setup_seconds
           << ",\"state_audit_seconds\":" << observer.audit_seconds << ",\"snapshot_seconds\":" << observer.snapshot_seconds
           << ",\"scorer_views\":" << observer.view_count
           << ",\"scorer_owned_array_bytes\":" << observer.owned_array_bytes
           << ",\"observed_free_memory_drop_bytes\":" << observer.resident_view_allocation_bytes
           << ",\"max_view_creation_free_memory_drop_bytes\":" << observer.max_transient_allocation_bytes
           << ",\"tick_p50_ms\":" << ticks.percentile(.5) << ",\"tick_p95_ms\":" << ticks.percentile(.95)
           << ",\"speech_tick_p95_ms\":" << speaking_ticks.percentile(.95)
           << ",\"scoring_tick_p95_ms\":" << scoring_ticks.percentile(.95)
           << ",\"max_tick_ms\":" << max_tick_ms
           << ",\"learning_tf32\":" << (state.meta[16] ? "true" : "false")
           << ",\"score_math\":\"strict FP32, reset-state source windows\",\"inputs_unchanged\":true"
           << ",\"selects_replay\":false,\"generated_text_targets\":false}\n";
    report.close();
    require(bool(report), "Probe result flush failed");
    std::cout << "Replay measurements=" << observer.observations << " forwards=" << observer.forward_calls
              << " live_seconds=" << elapsed << '\n';
}
} // namespace replay_priority
int main(int argc, char **argv) {
    try {
        require(argc >= 2 && std::string(argv[1]) == "run", "Expected replay priority probe run --options");
        replay_priority::run(Args(argc, argv));
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
