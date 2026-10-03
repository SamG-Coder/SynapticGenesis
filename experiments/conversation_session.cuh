// Sequential questions and live updates on one resident mutable model.
#pragma once
#include "conversation_protocol.hpp"

namespace conversation {
inline double elapsed(std::chrono::steady_clock::time_point begin) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
}
class Session {
    LiveEngine engine;
    State state;
    const LiveCurriculum &curriculum;
    LiveCorpus data;
    Model answer;
    std::unique_ptr<GraphDecoder> decoder;
    fs::path out;
    std::string live_prompt;
    std::ofstream events, commands, speech;
    int tokens, top;
    float temperature;
    uint64_t seed, questions = 0, answers_generated = 0;

    void flush() {
        events.flush(); commands.flush(); speech.flush();
        require(bool(events) && bool(commands) && bool(speech), "Conversation journal write failed");
    }

  public:
    Session(const fs::path &checkpoint, const State &initial, const LiveCurriculum &selected,
            const LiveCurriculum *previous, const fs::path &directory, const std::string &saved_prompt,
            int generated, int top_k, float temp, uint64_t sampling_seed)
        : engine(checkpoint_config(initial), int(initial.meta[6]), initial.meta[16] != 0),
          curriculum(selected), data(selected.stages.front().corpus),
          answer(engine.root, 1, ModelViewState::independent), out(directory), live_prompt(saved_prompt),
          tokens(generated), top(top_k), temperature(temp), seed(sampling_seed) {
        load(checkpoint, engine.root, state, true);
        require(live_version(state) == 5 && state.meta[5] == 1 && !has_synaptic_history(state),
                "Resident conversation currently requires ordinary grouped replay without teachers or SI");
        if (previous)
            curriculum.extend(state, *previous);
        data = curriculum.corpus(state);
        data.validate(state);
        ReplayMemory(state).validate(data, engine.root.T);
        curriculum.validate(state);
        curriculum.rate(state, state.hp[7]);
        require(state.meta[29] == hash_bytes(live_prompt.data(), live_prompt.size()),
                "Preserve the checkpoint's original live speech prompt");
        require(answer.w.p == engine.root.w.p && answer.m.p == engine.root.m.p &&
                    answer.v.p == engine.root.v.p && answer.g.p == engine.root.g.p &&
                    answer.decay.p == engine.root.decay.p, "Question view must share the learned workspace");
        answer.fast(false); // Match the preserved sample command's strict FP32 inference.
        fs::create_directories(out);
        events.open(out / "events.jsonl", std::ios::binary);
        commands.open(out / "commands.sgconversation", std::ios::binary);
        speech.open(out / "speech.bin", std::ios::binary);
        commands << "SGCONVERSATION1\n";
        events << std::setprecision(17) << "{\"event\":\"start\",\"parameters\":" << engine.root.a.n
               << ",\"online_updates\":" << state.meta[24] << ",\"global_updates\":" << state.meta[7]
               << ",\"single_parameter_workspace\":true,\"independent_question_recurrence\":true"
               << ",\"question_math\":\"strict FP32\",\"question_scratch_bytes\":"
               << explicit_model_bytes(answer) - 20 * engine.root.a.n;
        state.membrane.report(events);
        events << "}\n";
        flush();
        std::cout << "ready parameters=" << engine.root.a.n << " online_updates=" << state.meta[24]
                  << " shared_weights=yes\n";
    }
    void ask(const Command &command) {
        const auto start = std::chrono::steady_clock::now();
        const bool captured = !decoder;
        if (!decoder)
            decoder = std::make_unique<GraphDecoder>(answer);
        const double capture_seconds = captured ? elapsed(start) : 0;
        answer.reset();
        decoder->begin(); // Finish previous updates and the independent reset.
        const std::string prefix = prompt(command.question);
        auto timing = std::chrono::steady_clock::now();
        std::vector<float> logits;
        for (unsigned char byte : prefix)
            logits = decoder->step(byte);
        const double prefill_seconds = elapsed(timing);
        uint64_t rng = seed;
        std::string generated;
        timing = std::chrono::steady_clock::now();
        for (int index = 0; index < tokens; ++index) {
            const int byte = draw_byte(logits, top, temperature, rng);
            generated.push_back(char(byte));
            logits = decoder->step(byte);
        }
        const double decode_seconds = elapsed(timing);
        const fs::path filename = "answer-" + command.label + ".bin";
        std::ofstream raw(out / filename, std::ios::binary);
        raw << prefix << generated;
        raw.close();
        require(bool(raw), "Cannot save raw conversation answer");
        ++questions;
        answers_generated += uint64_t(tokens);
        events << "{\"event\":\"answer\",\"id\":" << probes::json(command.label)
               << ",\"question\":" << probes::json(command.question) << ",\"answer\":" << probes::json(generated)
               << ",\"raw_file\":" << probes::json(filename.string())
               << ",\"online_updates\":" << state.meta[24] << ",\"global_updates\":" << state.meta[7]
               << ",\"generated_bytes\":" << tokens << ",\"captured_now\":" << (captured ? "true" : "false")
               << ",\"capture_seconds\":" << capture_seconds << ",\"prefill_seconds\":" << prefill_seconds
               << ",\"decode_seconds\":" << decode_seconds
               << ",\"resident_decode_bytes_per_second\":" << tokens / decode_seconds << "}\n";
        std::cout << "answer " << command.label << ": ";
        for (unsigned char c : generated) {
            if (c == '\n' || c == '\t' || (c >= 32 && c < 127))
                std::cout << char(c);
            else
                std::cout << '<' << unsigned(c) << '>';
        }
        std::cout << "\n";
    }
    void learn(uint64_t endpoint) {
        require(endpoint > state.meta[24] && endpoint <= curriculum.stages.back().end_update,
                "Learning endpoint must advance within the admitted curriculum");
        const uint64_t every = ReplayMemory(state).every();
        const uint64_t replay_steps = every ? endpoint / every - state.meta[24] / every : 0;
        require(endpoint - state.meta[24] + replay_steps + state.meta[7] <= 1000000000ull,
                "Conversation would exceed the optimizer step limit");
        const uint64_t start_updates = state.meta[24], start_pairs = state.meta[22];
        const auto start = std::chrono::steady_clock::now();
        for (; state.meta[24] < endpoint;) {
            curriculum.advance(engine, data, state, &events);
            const auto result = live_tick(engine, data, state, live_prompt);
            ck(cudaDeviceSynchronize());
            require(std::isfinite(result.loss) && std::isfinite(result.gradient_norm) &&
                        std::isfinite(result.replay_loss), "Nonfinite resident learning result");
            speech.write(result.speech.data(), std::streamsize(result.speech.size()));
            events << "{\"event\":\"update\",\"online_updates\":" << state.meta[24]
                   << ",\"global_updates\":" << state.meta[7] << ",\"observed_pairs\":" << state.meta[22]
                   << ",\"loss\":" << result.loss << ",\"gradient_norm\":" << result.gradient_norm
                   << ",\"replay_pairs\":" << ReplayMemory(state).pairs() << "}\n";
        }
        const double seconds = elapsed(start);
        events << "{\"event\":\"learned\",\"source_updates\":" << state.meta[24] - start_updates
               << ",\"source_pairs\":" << state.meta[22] - start_pairs << ",\"seconds\":" << seconds
               << ",\"source_pairs_per_second\":" << (state.meta[22] - start_pairs) / seconds
               << ",\"online_updates\":" << state.meta[24] << "}\n";
        std::cout << "learned online_updates=" << state.meta[24] << " source_pairs=" << state.meta[22] - start_pairs << '\n';
    }
    void snapshot(const fs::path &filename) {
        save(out / filename, engine.root, state);
        events << "{\"event\":\"saved\",\"file\":" << probes::json(filename.string())
               << ",\"online_updates\":" << state.meta[24] << ",\"global_updates\":" << state.meta[7] << "}\n";
    }
    void execute(const Command &command) {
        commands << command.source << '\n';
        commands.flush(); // Record the intended operation before mutating state.
        require(bool(commands), "Cannot journal conversation command");
        if (command.kind == "ask")
            ask(command);
        else if (command.kind == "learn")
            learn(command.endpoint);
        else if (command.kind == "save")
            snapshot("checkpoint-" + command.label + ".ckpt");
        flush();
    }
    void finish() {
        snapshot("final.ckpt");
        events << "{\"event\":\"complete\",\"questions\":" << questions
               << ",\"question_generated_bytes\":" << answers_generated
               << ",\"online_updates\":" << state.meta[24] << ",\"global_updates\":" << state.meta[7]
               << ",\"query_outputs_are_training_targets\":false}\n";
        flush();
    }
};
} // namespace conversation
