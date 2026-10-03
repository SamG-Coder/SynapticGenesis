// A resident session over the existing production model and learning tick.
#define main synapticgenesis_cli_entry
#include "../src/spike_lm.cu"
#undef main
#include "conversation_session.cuh"

namespace conversation {
void run(const Args &args) {
    args.allow({"resume", "curriculum", "extend-curriculum", "out", "script", "prompt",
                "answer-bytes", "answer-top-k", "answer-temperature", "answer-seed"});
    const fs::path checkpoint = args.get("resume"), out = args.get("out");
    const std::string script = args.get("script"), saved_prompt = args.get("prompt", "The bird ");
    require(!checkpoint.empty() && !out.empty() && !fs::exists(out) && !script.empty(),
            "Supply --resume, --curriculum, --script and a fresh --out directory");
    const State initial = header(checkpoint);
    require(initial.meta[5] == 1 && live_version(initial) == 5, "Expected a grouped-replay live checkpoint");
    require(!saved_prompt.empty() && initial.meta[29] == hash_bytes(saved_prompt.data(), saved_prompt.size()),
            "Preserve the saved live --prompt");
    int count = args.num("answer-bytes", 96), top = args.num("answer-top-k", 1);
    float temperature = args.real("answer-temperature", 1);
    int seed = args.num("answer-seed", 42);
    require(count >= 1 && count <= 10000 && top >= 1 && top <= 256 && seed > 0 &&
                std::isfinite(temperature) && temperature > 0, "Invalid answer generation settings");
    auto curriculum = std::make_unique<LiveCurriculum>(args.get("curriculum"));
    std::unique_ptr<LiveCurriculum> previous;
    if (!args.get("extend-curriculum").empty()) {
        previous = std::move(curriculum);
        curriculum = std::make_unique<LiveCurriculum>(args.get("extend-curriculum"));
        curriculum->validate_extension(*previous);
    }
    std::vector<Command> prepared;
    if (script == "-")
        read_magic(std::cin);
    else {
        require(fs::file_size(script) <= 1024 * 1024, "Conversation script exceeds one MiB");
        std::ifstream input(script, std::ios::binary);
        require(bool(input), "Cannot open conversation script");
        prepared = read(input, initial.meta[24], curriculum->stages.back().end_update);
    }
    Session session(checkpoint, initial, *curriculum, previous.get(), out, saved_prompt,
                    count, top, temperature, uint64_t(seed));
    if (script != "-") {
        for (const auto &command : prepared)
            session.execute(command);
    } else {
        Sequence sequence(initial.meta[24], curriculum->stages.back().end_update);
        std::string source;
        size_t received_bytes = 0;
        while (line(std::cin, source)) {
            received_bytes += source.size() + 1;
            require(received_bytes <= 1024 * 1024, "Interactive command input exceeds one MiB");
            Command command;
            try {
                command = parse(source);
                sequence.accept(command);
            } catch (const std::exception &error) {
                std::cout << "rejected: " << error.what() << '\n';
                continue; // A mistyped command does not discard the resident learner.
            }
            if (command.kind.empty())
                continue;
            session.execute(command);
            if (sequence.finished)
                break;
        }
        require(!std::cin.bad(), "Conversation input failed");
    }
    session.finish();
}
} // namespace conversation
int main(int argc, char **argv) {
    try {
        std::cout.setf(std::ios::unitbuf);
        require(argc >= 2 && std::string(argv[1]) == "run", "Expected resident conversation run --options");
        conversation::run(Args(argc, argv));
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
