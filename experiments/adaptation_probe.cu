// Controlled learning on disposable copies through the production Model path.
// This is a diagnostic, not a replacement live-training implementation.
#define main synapticgenesis_cli_entry
#include "../src/spike_lm.cu"
#undef main
#include "adaptation_io.cuh"

namespace adaptation {
struct Evaluation {
    double loss = 0;
    uint64_t pairs = 0;
    std::vector<double> document_loss;
    std::vector<uint64_t> document_pairs;
};
void targets(const std::string &text, size_t offset, size_t count, std::vector<int> &x,
             std::vector<int> &y) {
    x.resize(count);
    y.resize(count);
    for (size_t i = 0; i < count; ++i) {
        x[i] = static_cast<unsigned char>(text[offset + i]);
        y[i] = static_cast<unsigned char>(text[offset + i + 1]);
    }
}
Evaluation evaluate_source(Model &root, const Source &source, uint64_t &peak_bytes) {
    Evaluation result;
    std::vector<int> x, y;
    for (const auto &document : source.documents) {
        double sum = 0;
        uint64_t pairs = 0;
        for (size_t offset = 0; offset + 1 < document.size(); offset += 128) {
            int count = int(std::min(size_t(128), document.size() - offset - 1));
            targets(document, offset, size_t(count), x, y);
            std::unique_ptr<Model> tail;
            Model *model = &root;
            if (count != 128) {
                tail = std::make_unique<Model>(root.q, 1, count);
                // Construction briefly owns its own parameters before sharing.
                peak_bytes = std::max(peak_bytes, explicit_model_bytes(root) + explicit_model_bytes(*tail));
                tail->share_parameters(root);
                model = tail.get();
            }
            model->reset();
            model->forward(x, &y);
            auto values = model->losses.host();
            for (float value : values) {
                require(std::isfinite(value) && value >= 0, "Nonfinite/negative adaptation evaluation loss");
                sum += value;
            }
            pairs += count;
        }
        result.document_loss.push_back(sum / double(pairs));
        result.document_pairs.push_back(pairs);
        result.loss += sum;
        result.pairs += pairs;
    }
    result.loss /= double(result.pairs);
    return result;
}
void print_evaluation(std::ostream &out, const Evaluation &value) {
    out << "{\"loss\":" << value.loss << ",\"pairs\":" << value.pairs << ",\"documents\":[";
    for (size_t i = 0; i < value.document_loss.size(); ++i) {
        if (i)
            out << ',';
        out << "{\"index\":" << i << ",\"loss\":" << value.document_loss[i]
            << ",\"pairs\":" << value.document_pairs[i] << '}';
    }
    out << "]}";
}
void run(const Args &args) {
    args.allow({"train", "validation", "schedule", "checkpoint", "resume", "out", "mode",
                "seed", "lr", "steps", "endpoints", "channels", "hidden", "layers", "cell"});
    fs::path out = args.get("out");
    require(!out.empty() && !fs::exists(out), "Adaptation needs a fresh output directory");
    const Source train(args.get("train")), validation(args.get("validation"));
    const Schedule schedule(args.get("schedule"), train);
    std::string mode = args.get("mode"), original = args.get("checkpoint");
    require(mode == "fresh" || mode == "carry" || mode == "reset", "Expected fresh/carry/reset mode");
    require((mode == "fresh") == original.empty(), "Only experienced starts require an original checkpoint");
    int seed = args.num("seed", 1337), steps = args.num("steps", 4096);
    float lr = args.real("lr", .000075f);
    require(seed > 0 && steps >= 0 && uint64_t(steps) <= schedule.windows.size() && lr > 0 && lr <= .01f,
            "Invalid adaptation seed, steps or learning rate");
    auto endpoint_text = args.get("endpoints", "0,64,256,1024,4096");
    std::replace(endpoint_text.begin(), endpoint_text.end(), ',', ' ');
    std::istringstream endpoint_input(endpoint_text);
    std::vector<int> endpoints;
    int point;
    while (endpoint_input >> point) {
        require(point >= 0 && (endpoints.empty() || point > endpoints.back()), "Invalid adaptation endpoints");
        endpoints.push_back(point);
    }
    require(endpoint_input.eof() && !endpoints.empty(), "Invalid endpoint list");
    Config q{args.num("channels", 256), args.num("hidden", 512), args.num("layers", 4),
             args.get("cell").empty() ? 6 : cell_version(args)};
    StoredCheckpoint ancestor;
    if (!original.empty()) {
        ancestor = read_checkpoint(original);
        auto loaded = checkpoint_config(ancestor.state);
        for (auto key : {"channels", "hidden", "layers", "cell"})
            require(args.get(key).empty(), "Experienced starts derive their architecture from the checkpoint");
        q = loaded;
    }
    require(q.cell == 5 || q.cell == 6, "Adaptation diagnostic supports selective and associative cells");
    Model model(q, 1, 128);
    if (mode == "fresh")
        model.w.put(initialize(q, model.a, uint64_t(seed)));
    else {
        model.w.put(ancestor.weights);
        if (mode == "carry") {
            model.m.put(ancestor.first);
            model.v.put(ancestor.second);
        }
    }
    Identity id{};
    id[0] = magic;
    id[1] = 1;
    id[3] = schedule.hash;
    id[4] = mode == "fresh" ? 0 : (mode == "carry" ? 1 : 2);
    id[5] = original.empty() ? 0 : identity(original);
    id[6] = uint64_t(seed);
    id[7] = mode == "carry" ? ancestor.state.meta[7] : 0;
    id[8] = tensor_hash(model.w);
    id[9] = tensor_hash(model.v, tensor_hash(model.m));
    id[10] = float_word(lr);
    id[12] = train.hash;
    id[13] = schedule.windows.size();
    id[14] = validation.hash;
    State state;
    state.meta[7] = id[7];
    state.meta[11] = uint64_t(seed);
    state.meta[12] = train.hash;
    state.meta[13] = validation.hash;
    state.meta[19] = magic; // Explicitly label disposable diagnostic checkpoints.
    state.hp[0] = lr;
    state.hp[1] = .01f;
    state.hp[2] = 1.f;
    if (!args.get("resume").empty()) {
        id = restore_identity(args.get("resume"), id);
        load(args.get("resume"), model, state);
        require(state.meta[17] == 0 && state.meta[19] == magic && state.meta[16] == 0 &&
                    state.meta[5] == 1 && state.meta[6] == 128 && state.meta[7] == id[7] + id[2] &&
                    state.meta[11] == id[6] && state.meta[12] == id[12] && state.meta[13] == id[14] &&
                    float_word(state.hp[0]) == id[10] && state.hp[1] == .01f && state.hp[2] == 1.f,
                "Adaptation checkpoint policy mismatch");
    }
    const uint64_t begin = id[2];
    require(uint64_t(steps) >= begin && id[7] + uint64_t(steps) <= 1000000000ull,
            "Adaptation step limit precedes resume or exceeds optimizer bounds");
    fs::create_directories(out);
    std::ofstream log(out / "updates.jsonl");
    require(bool(log), "Cannot create adaptation journal");
    log << std::setprecision(17);
    uint64_t peak_bytes = explicit_model_bytes(model);
    double update_seconds = 0, evaluation_seconds = 0;
    std::vector<int> x, y;
    auto assess = [&](uint64_t step) {
        auto start = std::chrono::steady_clock::now();
        uint64_t before = learning_hash(model);
        auto train_result = evaluate_source(model, train, peak_bytes);
        auto validation_result = evaluate_source(model, validation, peak_bytes);
        require(learning_hash(model) == before, "Adaptation evaluation changed weights or optimizer");
        evaluation_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        model.reset();
        auto file = out / ("checkpoint-" + std::to_string(step) + ".ckpt");
        save(file, model, state);
        read_checkpoint(file); // Validate finite values and the authoritative payload.
        id[2] = step;
        save_identity(file, id);
        std::ofstream report(out / ("evaluation-" + std::to_string(step) + ".json"));
        report << std::setprecision(17) << "{\"updates\":" << step << ",\"global_updates\":" << state.meta[7]
               << ",\"presented_pairs\":" << step * 128 << ",\"weights_optimizer_unchanged\":true,\"train\":";
        print_evaluation(report, train_result);
        report << ",\"validation\":";
        print_evaluation(report, validation_result);
        report << "}\n";
        report.close();
        require(bool(report), "Adaptation evaluation flush failed");
        std::cout << "Probe updates=" << step << " train=" << train_result.loss
                  << " development=" << validation_result.loss << std::endl;
    };
    assess(begin);
    for (uint64_t step = begin; step < uint64_t(steps); ++step) {
        auto window = schedule.windows[size_t(step)];
        targets(train.documents[window.document], window.offset, 128, x, y);
        auto start = std::chrono::steady_clock::now();
        model.reset();
        float loss = model.forward(x, &y);
        require(std::isfinite(loss) && loss >= 0, "Invalid adaptation training loss");
        model.backward();
        float norm = model.update(int(++state.meta[7]), lr, .01f, 1.f, 1.f);
        ck(cudaDeviceSynchronize());
        double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        update_seconds += seconds;
        log << "{\"update\":" << step + 1 << ",\"document\":" << window.document
            << ",\"offset\":" << window.offset << ",\"loss_before_update\":" << loss
            << ",\"gradient_norm\":" << norm << ",\"seconds\":" << seconds << "}\n";
        if (step + 1 == uint64_t(steps) ||
            std::binary_search(endpoints.begin(), endpoints.end(), int(step + 1)))
            assess(step + 1);
    }
    log.close();
    require(bool(log), "Adaptation update journal flush failed");
    require((original.empty() || identity(original) == id[5]) && identity(args.get("train")) == train.hash &&
                identity(args.get("validation")) == validation.hash && identity(args.get("schedule")) == schedule.hash,
            "Adaptation input changed during execution");
    std::ofstream report(out / "result.json");
    report << std::setprecision(17) << "{\"completed\":true,\"mode\":" << probes::json(mode)
           << ",\"seed\":" << seed << ",\"cell\":" << q.cell << ",\"channels\":" << q.c
           << ",\"hidden\":" << q.h << ",\"layers\":" << q.l << ",\"parameters\":" << model.a.n
           << ",\"initial_weights_hash\":\"" << id[8] << "\",\"initial_moments_hash\":\"" << id[9]
           << "\",\"initial_global_updates\":" << id[7] << ",\"global_updates\":" << state.meta[7]
           << ",\"start_updates\":" << begin << ",\"end_updates\":" << steps
           << ",\"session_presented_pairs\":" << (uint64_t(steps) - begin) * 128
           << ",\"learning_rate\":" << lr << ",\"update_seconds\":" << update_seconds
           << ",\"evaluation_seconds\":" << evaluation_seconds << ",\"peak_explicit_model_bytes\":" << peak_bytes
           << ",\"inputs_unchanged\":true,\"reset_state_each_window\":true,\"generated_targets\":false}\n";
    report.close();
    require(bool(report), "Adaptation result flush failed");
}
} // namespace adaptation
int main(int argc, char **argv) {
    try {
        require(argc >= 2 && std::string(argv[1]) == "run", "Expected adaptation probe run --options");
        adaptation::run(Args(argc, argv));
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
