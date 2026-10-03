// Independent scalar references plus complete same-device training repeats.
#pragma once
void reduction_test(const Args &args) {
    args.allow({"out"});
    fs::path out = args.get("out", "reports/reduction-tests");
    fs::create_directories(out);
    double embedding_error = 0, input_error = 0, gain_error = 0;
    int kernel_cases = 0, repeat_cases = 0;
    for (int n : {1, 7, 128, 1025})
        for (int c : {8, 37, 256}) {
            Model storage(Config{c, 8, 1}, 1, n);
            std::vector<int> tokens(n);
            std::vector<float> x(size_t(n) * c), dy(x.size()), initial(x.size()), gain(c);
            for (int i = 0; i < n; ++i)
                tokens[i] = n == 1025 ? 7 : (i * 13) % 19;
            for (size_t i = 0; i < x.size(); ++i) {
                x[i] = .2f * std::sin(float(i % 137) * .17f);
                dy[i] = .003f * std::cos(float(i % 89) * .13f);
                initial[i] = .01f * std::sin(float(i % 31));
            }
            for (int j = 0; j < c; ++j)
                gain[j] = .7f + .01f * (j % 11);
            Buf values(x.size()), upstream(dy.size()), gains(c), dg(c), dx(x.size()), rs(n),
                normalized(x.size()), emb(size_t(256) * c);
            values.put(x);
            upstream.put(dy);
            gains.put(gain);
            ck(cudaMemcpy(storage.input, tokens.data(), n * sizeof(int), cudaMemcpyHostToDevice));
            rms_fwd<<<n, 256>>>(normalized.p, rs.p, values.p, gains.p, c);
            auto r = rs.host();
            std::vector<double> expected_gain(c, .125), expected_embedding(size_t(256) * c, 0);
            for (int row = 0; row < n; ++row)
                for (int j = 0; j < c; ++j) {
                    int i = row * c + j;
                    expected_gain[j] += double(dy[i]) * x[i] * r[row];
                    expected_embedding[tokens[row] * c + j] += dy[i];
                }
            std::vector<float> first_dx, first_dg, first_emb;
            for (int repeat = 0; repeat < 4; ++repeat) {
                dx.put(initial);
                dg.put(std::vector<float>(c, .125f));
                rms_bwd<<<n + (c + 31) / 32, 256>>>(dx.p, dg.p, upstream.p, values.p, gains.p, rs.p, n, c,
                                                    true);
                embedding_grad<<<256, 256>>>(emb.p, upstream.p, storage.input, n, c);
                auto got_dx = dx.host(), got_dg = dg.host(), got_emb = emb.host();
                if (!repeat) {
                    first_dx = got_dx;
                    first_dg = got_dg;
                    first_emb = got_emb;
                } else {
                    require(got_dx == first_dx && got_dg == first_dg && got_emb == first_emb,
                            "Fixed-order kernel repeat differs");
                }
                for (int row = 0; row < n; ++row) {
                    double dot = 0;
                    for (int j = 0; j < c; ++j)
                        dot += double(dy[row * c + j]) * gain[j] * x[row * c + j] / c;
                    for (int j = 0; j < c; ++j) {
                        int i = row * c + j;
                        double expected = initial[i] + double(dy[i]) * gain[j] * r[row] -
                                          double(x[i]) * r[row] * r[row] * r[row] * dot;
                        input_error = std::max(input_error, std::abs(expected - got_dx[i]));
                    }
                }
                for (int j = 0; j < c; ++j)
                    gain_error = std::max(gain_error, std::abs(expected_gain[j] - got_dg[j]));
                for (size_t i = 0; i < got_emb.size(); ++i)
                    embedding_error = std::max(embedding_error, std::abs(expected_embedding[i] - got_emb[i]));
            }
            ++kernel_cases;
        }
    require(std::max({embedding_error, input_error, gain_error}) < 2e-5,
            "Fixed-order reduction disagrees with double scalar reference");

    auto identical = [](const std::vector<float> &a, const std::vector<float> &b) {
        return a.size() == b.size() && !std::memcmp(a.data(), b.data(), a.size() * sizeof(float));
    };
    for (int cell : {1, 2, 3, 4, 5})
        for (int batch : {1, 3})
            for (bool fast : {false, true}) {
                Config q{40, 72, 2, cell};
                Model first(q, batch, 17), second(q, batch, 17);
                auto weights = initialize(q, first.a, 1337), state = first.membranes();
                if (q.gated())
                    for (auto layer : first.a.layers)
                        for (int i = 0; i < q.c * q.h; ++i)
                            weights[layer.gate_w + i] = .03f * std::sin(float(i) * .19f);
                for (size_t i = 0; i < state.size(); ++i)
                    state[i] = .3f * std::sin(float(i) * .11f);
                for (Model *model : {&first, &second}) {
                    model->w.put(weights);
                    model->membranes(state);
                    model->fast(fast);
                }
                for (int step = 1; step <= 24; ++step) {
                    std::vector<int> x(first.N), y(first.N);
                    for (int i = 0; i < first.N; ++i) {
                        x[i] = (i * 7 + step) % 9 + 97;
                        y[i] = (i * 7 + step + 1) % 9 + 97;
                    }
                    for (Model *model : {&first, &second}) {
                        model->forward(x, &y, true);
                        model->backward(.001f);
                        model->update(step, .0003f);
                    }
                    require(identical(first.g.host(), second.g.host()) &&
                                identical(first.w.host(), second.w.host()) &&
                                identical(first.m.host(), second.m.host()) &&
                                identical(first.v.host(), second.v.host()) &&
                                identical(first.membranes(), second.membranes()) &&
                                identical(first.logits.host(), second.logits.host()),
                            "Repeated complete training step differs");
                }
                ++repeat_cases;
            }
    std::ofstream result(out / "native.json");
    result << std::setprecision(12) << "{\"passed\":true,\"scalar_reference_cases\":" << kernel_cases
           << ",\"embedding_max_abs_error\":" << embedding_error
           << ",\"rms_input_max_abs_error\":" << input_error << ",\"rms_gain_max_abs_error\":" << gain_error
           << ",\"complete_repeat_cases\":" << repeat_cases
           << ",\"updates_per_repeat\":24,\"all_five_cells\":true,\"strict_fp32_and_tf32\":true,"
              "\"single_and_multiple_sequences\":true,\"training_arrays_bitwise_identical\":true}";
    std::cout << "PASS ordered reductions: scalar references and " << repeat_cases
              << " complete repeated training cases\n";
}
