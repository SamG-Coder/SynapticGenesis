// Causal fast-state updates, independent streams, negative controls and live resume.
#pragma once
#include "adaptive_tests.cuh"
void associative_test(const Args &args) {
    adaptive_test(args, 6); // Shared neuron, gradient-fixture and live-state contract.
    fs::path out = args.get("out", "reports/associative-tests");
    constexpr int B = 2, T = 4, W = association::width;
    association::Cache cache(B, T, 8);
    std::vector<float> features(B * T * association::packed, 0.f);
    for (int b = 0; b < B; ++b)
        for (int t = 0; t < T; ++t) {
            auto f = features.data() + (b * T + t) * association::packed;
            f[t == 3 ? 1 : 0] = 1; // Query the older key on the second step.
            f[W + t % 2] = 1;
            f[2 * W] = float(b + 1) * (t == 0 ? .25f : (t == 1 ? .5f : -.75f));
            f[3 * W] = t == 3 ? .5f : 1.f;
            f[3 * W + 1] = t == 3 ? 0.f : 1.f;
        }
    cache.features.put(features);
    cache.initial.zero();
    association::forward<<<B, 256>>>(cache.features.p, cache.initial.p, cache.previous.p, cache.reads.p,
                                     cache.state.p, T, true);
    auto reads = cache.reads.host(), state = cache.state.host();
    float kernel_error = 0;
    for (int b = 0; b < B; ++b) {
        const float expected[] = {.25f, .25f, -.75f, .25f};
        for (int t = 0; t < T; ++t)
            for (int row = 0; row < W; ++row)
                kernel_error = std::max(
                    kernel_error, std::abs(reads[(b * T + t) * W + row] - (row ? 0 : (b + 1) * expected[t])));
        require(state[b * W * W] == -.375f * (b + 1) && state[b * W * W + 1] == .25f * (b + 1),
                "Delta overwrite, independent streams or decay failed");
    }
    require(kernel_error == 0, "Fast memory disagrees with literal key/value calculation");

    Config q{32, 64, 2, 6};
    Model model(q, 1, 8), zero_memory(q, 1, 8);
    auto weights = initialize(q, model.a, 123);
    activate_association_fixture(q, model.a, weights);
    model.w.put(weights);
    zero_memory.w.put(weights);
    std::vector<int> x{'T', 'h', 'e', ' ', 'b', 'i', 'r', 'd'}, y{'h', 'e', ' ', 'b', 'i', 'r', 'd', '.'};
    model.forward(x, &y, true);
    auto before = model.membranes();
    require(maxdiff(before, zero_memory.membranes()) > 0, "Observation did not change fast state");
    require(model.w.host() == weights, "Forward-only state update changed slow weights");
    auto changed = before;
    for (int l = 0; l < q.l; ++l)
        std::fill_n(changed.begin() + l * q.recurrent_per_layer() + 2 * q.h, association::matrix, 0.f);
    zero_memory.membranes(changed);
    model.forward(x, nullptr, true);
    zero_memory.forward(x, nullptr, true);
    float state_effect = maxdiff(model.logits.host(), zero_memory.logits.host());
    require(state_effect > 1e-6, "Stored associations had no effect on later predictions");

    auto saved = model.membranes();
    model.forward(x, &y, false);
    auto original = model.logits.host();
    require(saved == model.membranes(), "Non-streaming evaluation overwrote live associations");
    y.assign(8, 0);
    model.forward(x, &y, false);
    require(original == model.logits.host(), "Target labels leaked into forward memory");
    x[7] = '!';
    model.forward(x, nullptr, false);
    auto altered = model.logits.host();
    require(std::equal(original.begin(), original.begin() + 7 * 256, altered.begin()),
            "Future input changed an earlier prediction");
    require(maxdiff(original, altered) > 0, "Causality fixture did not change its final input response");

    // Zero-initialized readout must begin learning, then train its storage projections.
    model.w.put(initialize(q, model.a, 321));
    model.forward(x, &y, false);
    model.backward();
    auto gradient = model.g.host();
    float first_readout_gradient = 0;
    for (const auto &layer : model.a.layers)
        for (int i = 0; i < q.c * W; ++i)
            first_readout_gradient =
                std::max(first_readout_gradient, std::abs(gradient[layer.association_out + i]));
    require(first_readout_gradient > 1e-6, "Zero readout cannot begin learning");
    model.update(1, .001f, 0, 0);
    model.forward(x, &y, false);
    model.backward();
    gradient = model.g.host();
    float next_projection_gradient = 0;
    for (const auto &layer : model.a.layers)
        for (int i = 0; i < association::packed * q.h; ++i)
            next_projection_gradient =
                std::max(next_projection_gradient, std::abs(gradient[layer.association_w + i]));
    require(next_projection_gradient > 1e-8, "Fast-state projections cannot learn after readout activation");
    std::ofstream report(out / "association.json");
    report << "{\"passed\":true,\"literal_delta_error\":" << kernel_error
           << ",\"stored_state_logit_effect\":" << state_effect
           << ",\"initial_readout_gradient\":" << first_readout_gradient
           << ",\"subsequent_projection_gradient\":" << next_projection_gradient
           << ",\"causal\":true,\"targets_excluded\":true,\"slow_weights_unchanged_by_forward\":true,"
              "\"independent_streams\":true,\"nonstreaming_state_isolated\":true}\n";
    std::cout << "PASS associative: literal delta rule, causal forward plasticity, state ablation and "
                 "learning bootstrap\n";
}
