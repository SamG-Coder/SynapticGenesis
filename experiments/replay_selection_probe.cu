// Disposable live replay experiment; model math and live update order stay shared.
#define main synapticgenesis_cli_entry
#include "../src/spike_lm.cu"
#undef main
#include "replay_two_choice.cuh"
#include "replay_selection_identity.cuh"

namespace replay_priority {
void run(const Args &args) {
    args.allow({"checkpoint", "resume", "curriculum", "out", "updates", "seconds", "mode", "seed", "prompt", "audit-state"});
    fs::path base = args.get("checkpoint"), resume = args.get("resume"), schedule = args.get("curriculum"), out = args.get("out");
    require(!out.empty() && !fs::exists(out), "Replay experiment needs a fresh output directory");
    auto ancestor = read_checkpoint(base).state;
    require(ancestor.meta[17] == 5 && ancestor.meta[5] == 1 && !has_synaptic_history(ancestor),
            "Replay experiment requires grouped live memory without SI or teachers");
    LiveCurriculum curriculum(schedule);
    curriculum.validate(ancestor);
    std::string mode_name = args.get("mode", "none"), prompt = args.get("prompt", "The bird ");
    require(mode_name == "none" || mode_name == "uniform" || mode_name == "interference", "Invalid replay experiment mode");
    int mode = mode_name == "none" ? 0 : (mode_name == "uniform" ? 1 : 2);
    int seed = args.num("seed", 42), updates = args.num("updates", 0), audit = args.num("audit-state", 0);
    float budget = args.real("seconds", 0);
    require(seed > 0 && updates > 0 && budget >= 0 && budget <= 600 && (audit == 0 || audit == 1), "Invalid replay experiment limits");
    require(!prompt.empty() && prompt.size() <= 65536 && ancestor.meta[29] == hash_bytes(prompt.data(), prompt.size()),
            "Replay experiment requires original prompt");
    auto id = identity(base, curriculum, ancestor, mode, uint64_t(seed));
    fs::path source = resume.empty() ? base : resume;
    if (!resume.empty())
        validate_identity(resume, id);
    State state = read_checkpoint(source).state;
    curriculum.validate(state);
    require(uint64_t(updates) > state.meta[24] && uint64_t(updates) - ancestor.meta[24] <= 32768 &&
                state.meta[24] >= ancestor.meta[24] && uint64_t(updates) <= curriculum.stages.back().end_update,
            "Replay experiment exceeds admitted update range");
    require(state.meta[17] == 5 && state.meta[5] == 1 && !has_synaptic_history(state), "Invalid resumed replay model");
    auto source_id = file_identity(source), schedule_id = file_identity(schedule);
    LiveEngine engine(checkpoint_config(state), int(state.meta[6]), state.meta[16] != 0);
    load(source, engine.root, state, true);
    auto data = curriculum.corpus(state);
    data.validate(state);
    ReplayMemory memory(state);
    memory.validate(data, engine.root.T);
    require(memory.count() > 0, "Replay experiment needs observed memory");
    uint64_t extra = uint64_t(updates) / memory.every() - state.meta[24] / memory.every();
    require(uint64_t(updates) - state.meta[24] + extra + state.meta[7] <= 1000000000ull, "Replay optimizer limit");
    const auto begin = state.meta[24], source_pairs = state.meta[22], replay_pairs = memory.pairs();
    fs::create_directories(out);
    TwoChoice selector(out, uint64_t(seed), mode == 2, audit != 0);
    std::ofstream speech(out / "speech.txt", std::ios::binary);
    require(bool(speech), "Cannot create replay speech log");
    LiveLatency ticks;
    double maximum = 0;
    auto start = std::chrono::steady_clock::now();
    do {
        auto tick_start = std::chrono::steady_clock::now();
        curriculum.advance(engine, data, state);
        auto result = live_tick(engine, data, state, prompt, mode ? &selector : nullptr);
        ck(cudaDeviceSynchronize());
        double elapsed = seconds(tick_start) * 1000;
        ticks.add(elapsed);
        maximum = std::max(maximum, elapsed);
        if (!result.speech.empty())
            speech << result.speech << '\n';
    } while (state.meta[24] < uint64_t(updates) && (!budget || seconds(start) < budget));
    double elapsed = seconds(start);
    selector.finish();
    speech.close();
    require(bool(speech), "Replay speech flush failed");
    curriculum.validate(state);
    memory.validate(data, engine.root.T);
    save(out / "latest.ckpt", engine.root, state);
    read_checkpoint(out / "latest.ckpt");
    save_identity(out / "latest.ckpt", id);
    validate_identity(out / "latest.ckpt", id);
    require(file_identity(source) == source_id && file_identity(base) == id[2] && file_identity(schedule) == schedule_id,
            "Replay experiment input changed");
    LiveCurriculum reloaded(schedule);
    require(reloaded.hash == curriculum.hash, "Replay source content changed");
    std::ofstream report(out / "result.json");
    report << std::setprecision(17) << "{\"complete\":true,\"mode\":\"" << mode_name << "\",\"start\":" << begin
           << ",\"end\":" << state.meta[24] << ",\"requested_end\":" << updates << ",\"seconds_budget\":" << budget
           << ",\"live_seconds\":" << elapsed << ",\"source_pairs\":" << state.meta[22] - source_pairs
           << ",\"replay_pairs\":" << memory.pairs() - replay_pairs << ",\"comparisons\":" << selector.comparisons
           << ",\"overrides\":" << selector.overrides << ",\"singleton_pools\":" << selector.singleton_pools
           << ",\"score_forward_calls\":" << selector.forward_calls << ",\"scored_pairs\":" << selector.scored_pairs
           << ",\"scoring_seconds\":" << selector.scoring_seconds << ",\"setup_seconds\":" << selector.setup_seconds
           << ",\"owned_scoring_bytes\":" << selector.owned_array_bytes << ",\"views\":" << selector.view_count
           << ",\"tick_p95_ms\":" << ticks.percentile(.95) << ",\"max_tick_ms\":" << maximum
           << ",\"inputs_unchanged\":true,\"generated_text_targets\":false}\n";
    report.close();
    require(bool(report), "Replay result flush failed");
    std::cout << mode_name << " end=" << state.meta[24] << " overrides=" << selector.overrides << " seconds=" << elapsed << '\n';
}
} // namespace replay_priority
int main(int argc, char **argv) {
    try {
        require(argc >= 2 && std::string(argv[1]) == "run", "Expected replay selection probe run --options");
        replay_priority::run(Args(argc, argv));
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
