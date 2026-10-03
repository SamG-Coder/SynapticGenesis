// Numerical fixture for explicitly weighted observed targets; all model math is native.
#pragma once
void feedback_test(const Args &args) {
    args.allow({"out"});
    fs::path base = args.get("out", "reports/feedback-tests");
    for (int cell : {1, 2, 3, 4, 5}) {
        fs::path out = base / (cell == 5   ? "selective"
                               : cell == 1 ? "lif"
                                           : (cell == 2 ? "alif" : (cell == 3 ? "trace" : "gated")));
        fs::create_directories(out);
        Config q{32, 64, 2, cell};
        Model model(q, 2, 16);
        auto w = initialize(q, model.a, 123);
        model.w.put(w);
        auto initial = model.membranes();
        for (size_t i = 0; i < initial.size(); ++i)
            initial[i] = .3f + .2f * std::sin(float(i) * .17f);
        model.membranes(initial);
        std::vector<int> x(32), y(32);
        std::vector<float> weights(32, 1.f);
        for (int i = 0; i < 32; ++i) {
            x[i] = (i * 37 + 91) % 256;
            y[i] = (i * 19 + 13) % 256;
            weights[i] = (i % 16 >= 12) ? 64.f : (i % 3 == 0 ? 0.f : 1.f);
        }
        model.forward(x, &y, true);
        float loss = model.reweight_targets(weights);
        dump(out / "weights.f32", w);
        dump(out / "inputs.i32", x);
        dump(out / "targets.i32", y);
        dump(out / "target_weights.f32", weights);
        dump(out / "initial_state.f32", initial);
        dump(out / "final_state.f32", model.membranes());
        dump(out / "logits.f32", model.logits.host());
        model.backward();
        dump(out / "gradients.f32", model.g.host());
        model.backward(1);
        dump(out / "gradients_regularized.f32", model.g.host());
        model.backward();
        model.update(1, .001f, 0, 0);
        dump(out / "updated.f32", model.w.host());
        std::ofstream report(out / "fixture.json");
        report << std::setprecision(10) << "{\"channels\":32,\"hidden\":64,\"layers\":2,\"cell\":" << cell
               << ",\"batch\":2,\"context\":16,\"loss\":" << loss << ",\"parameters\":" << model.a.n << '}';
        for (auto bad : {std::vector<float>(31, 1.f), std::vector<float>(32, 0.f),
                         std::vector<float>(32, -1.f), std::vector<float>(32, 1001.f),
                         std::vector<float>(32, std::numeric_limits<float>::quiet_NaN())}) {
            bool rejected = false;
            try {
                model.reweight_targets(bad);
            } catch (const std::exception &) {
                rejected = true;
            }
            require(rejected, "Invalid target emphasis accepted");
        }
        // Check the document offset against a literal independently selected
        // target position, including a window starting immediately before it.
        std::string document = "The key is in the box.\nAnswer: box.";
        dump(out / "lesson.dat", std::vector<char>(document.begin(), document.end()));
        LiveCorpus data(out / "lesson.dat");
        data.emphasize_answers(0, 1, 64);
        require(data.feedback.at(0).answer_start == 31, "Answer field byte boundary incorrect");
        Model tiny(q, 1, 2);
        tiny.w.put(w);
        std::vector<int> tx{' ', 'b'}, ty{'b', 'o'};
        float original = tiny.forward(tx, &ty);
        auto original_grad = tiny.dlogits.host();
        float weighted = data.emphasize(tiny, 0, 30, 2, original);
        require(std::abs(weighted - original) < 1e-6f && maxdiff(original_grad, tiny.dlogits.host()) < 1e-7f,
                "A fully emphasized window must retain its mean gradient");
        tiny.forward(tx, &ty);
        auto raw = tiny.dlogits.host();
        data.emphasize(tiny, 0, 29, 2, original);
        auto scaled = tiny.dlogits.host();
        for (int i = 0; i < 512; ++i)
            require(std::abs(scaled[i] - raw[i] * (i < 256 ? 2.f / 65 : 128.f / 65)) < 1e-7,
                    "First answer target was off by one");
    }
    std::cout << "PASS feedback: all cell oracle fixtures, target boundaries and invalid weights\n";
}
