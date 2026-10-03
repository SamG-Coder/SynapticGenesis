// Allocation mode must preserve the common forward path and reject learning.
#pragma once
#include "test_fixtures.cuh"
void frozen_model_test(const fs::path &out) {
    auto rejects_learning = [](auto action) {
        bool failed = false;
        try { action(); } catch (const std::runtime_error &error) {
            failed = std::string(error.what()).find("requires learning buffers") != std::string::npos;
        }
        require(failed, "Frozen model accepted a learning operation or failed for another reason");
    };
    std::ofstream report(out / "frozen-model.json");
    report << "{\"cases\":[";
    int cases = 0, graph_cases = 0;
    for (int cell = 1; cell <= 6; ++cell)
        for (auto shape : std::array<std::pair<int, int>, 3>{{{1, 1}, {1, 8}, {3, 8}}})
            for (bool fast : {false, true}) {
                Config q{16, 24, 2, cell};
                int batch = shape.first, context = shape.second;
                Model full(q, batch, context), frozen(q, batch, context, ModelBuffers::frozen_forward);
                auto weights = initialize(q, full.a, 4321);
                activate_association_fixture(q, full.a, weights);
                if (q.gated())
                    for (auto layer : full.a.layers)
                        for (int i = 0; i < q.c * q.h; ++i)
                            weights[layer.gate_w + i] = .04f * std::sin(float(i) * .17f);
                full.w.put(weights); frozen.w.put(weights);
                full.fast(fast); frozen.fast(fast);
                uint64_t full_bytes = explicit_model_bytes(full), frozen_bytes = explicit_model_bytes(frozen);
                require(full_bytes == model_working_bytes(q, batch, context) &&
                            frozen_bytes == model_working_bytes(q, batch, context, ModelBuffers::frozen_forward) &&
                            frozen_bytes < full_bytes, "Frozen allocation estimate or saving differs");
                require(!frozen.target && !frozen.g.n && !frozen.m.n && !frozen.v.n && !frozen.decay.n &&
                            !frozen.dlogits.n && !frozen.losses.n && !frozen.loss_weights.n && !frozen.dx.n &&
                            !frozen.dnorm.n && !frozen.ds.n && !frozen.dz.n && !frozen.dgate.n &&
                            !frozen.neuron_partials.n, "Frozen model allocated learning-only buffers");
                auto state = full.membranes();
                for (size_t i = 0; i < state.size(); ++i) state[i] = .08f * std::sin(float(i) * .13f);
                full.membranes(state); frozen.membranes(state);
                std::vector<int> input(full.N);
                std::iota(input.begin(), input.end(), 41);
                for (bool streaming : {true, true, false}) {
                    full.forward(input, nullptr, streaming);
                    frozen.forward(input, nullptr, streaming);
                    require(full.logits.host() == frozen.logits.host() &&
                                full.membranes() == frozen.membranes(), "Frozen forward arithmetic differs");
                    for (auto &byte : input) byte = (byte + 17) % 256;
                }
                distillation::Targets targets(full.N, 2.f);
                targets.from_logits(full.logits);
                auto probabilities = targets.probabilities();
                targets.from_logits(frozen.logits);
                require(probabilities == targets.probabilities(), "Frozen teacher probabilities differ");
                if (batch == 1 && context == 1) {
                    full.membranes(state); frozen.membranes(state);
                    GraphDecoder decoder(frozen);
                    require(frozen.membranes() == state, "Frozen graph capture changed recurrence");
                    for (int byte : {41, 99, 32, 111, 70, 93, 12}) {
                        full.forward({byte}, nullptr, true);
                        require(decoder.step(byte) == full.logits.host() &&
                                    full.membranes() == frozen.membranes(), "Frozen graph decoding differs");
                    }
                    ++graph_cases;
                }
                auto unchanged_state = frozen.membranes();
                rejects_learning([&]() { frozen.forward(input, &input, true); });
                rejects_learning([&]() { frozen.backward(); });
                rejects_learning([&]() { frozen.update(1, .001f); });
                rejects_learning([&]() { frozen.reweight_targets(std::vector<float>(full.N, 1.f)); });
                rejects_learning([&]() { targets.apply(frozen, .5f); });
                rejects_learning([&]() { targets.apply(frozen, 0.f); });
                rejects_learning([&]() { frozen.share_parameters(full); });
                rejects_learning([&]() { full.share_parameters(frozen); });
                State saved;
                auto path = out / "frozen-save-guard.ckpt";
                save(path, full, saved);
                auto identity = teachers::file_hash(path);
                rejects_learning([&]() { save(path, frozen, saved); });
                rejects_learning([&]() { load(path, frozen, saved); });
                require(teachers::file_hash(path) == identity && frozen.w.host() == weights &&
                            frozen.membranes() == unchanged_state, "Rejected operation changed frozen state or file");
                report << (cases++ ? "," : "") << "{\"cell\":" << cell << ",\"batch\":" << batch
                       << ",\"context\":" << context << ",\"tf32\":" << (fast ? "true" : "false")
                       << ",\"full_gpu_bytes\":" << full_bytes << ",\"frozen_gpu_bytes\":" << frozen_bytes << "}";
            }
    report << "],\"passed\":true,\"forward_and_state_exact\":true,\"teacher_probabilities_exact\":true,"
              "\"explicit_allocation_counts_exact\":true,\"learning_and_checkpoint_mutation_rejected\":true,"
              "\"graph_cases\":" << graph_cases << "}\n";
    std::cout << "PASS frozen model: " << cases << " allocation/forward cases, " << graph_cases
              << " graph cases; learning operations rejected\n";
}
