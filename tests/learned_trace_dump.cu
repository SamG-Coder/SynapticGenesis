// Diagnostic access to the existing native forward computation. No duplicate
// neuron equations and no training; build separately from the study executable.
#define main synapticgenesis_cli_entry
#include "../src/spike_lm.cu"
#undef main

int main(int argc, char **argv) {
    try {
        if (argc != 4)
            throw std::runtime_error("Expected checkpoint, full input-text file, fresh output directory");
        const fs::path checkpoint = argv[1], input = argv[2], out = argv[3];
        require(!fs::exists(out), "Diagnostic output already exists");
        std::ifstream file(input, std::ios::binary);
        require(bool(file), "Cannot read diagnostic input");
        std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        require(text.size() >= 2 && text.size() <= 4097, "Invalid diagnostic input size");
        State state = header(checkpoint);
        probes::Scorer scorer(checkpoint_config(state));
        require(scorer.root.q.cell == 5, "This diagnostic requires the selective cell");
        load(checkpoint, scorer.root, state);
        std::vector<int> x, y;
        for (size_t i = 0; i + 1 < text.size(); ++i) {
            x.push_back(static_cast<unsigned char>(text[i]));
            y.push_back(static_cast<unsigned char>(text[i + 1]));
        }
        auto &model = scorer.view(int(x.size()));
        model.forward(x, &y);
        fs::create_directories(out);
        dump(out / "logits.f32", model.logits.host());
        dump(out / "losses.f32", model.losses.host());
        for (int i = 0; i < model.q.l; ++i) {
            const std::string prefix = "layer-" + std::to_string(i) + "-";
            auto &cache = model.cache[i];
            dump(out / (prefix + "norm.f32"), cache.norm.host());
            dump(out / (prefix + "z.f32"), cache.z.host());
            dump(out / (prefix + "gate.f32"), cache.gate.host());
            dump(out / (prefix + "u.f32"), cache.u.host());
            dump(out / (prefix + "spikes.f32"), cache.s.host());
            dump(out / (prefix + "emission.f32"), cache.emission.host());
        }
        std::cout << "Read-only selective traces: " << model.T << " bytes, " << model.q.l << " layers\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
