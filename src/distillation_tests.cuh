// Synthetic numerical fixtures only; these models never become population ancestors.
#pragma once
#include "test_fixtures.cuh"

void distillation_test(const Args &args) {
    args.allow({"out"});
    fs::path base = args.get("out", "reports/distillation-tests");
    auto rejects = [](auto action) {
        bool failed = false;
        try {
            action();
        } catch (const std::exception &) {
            failed = true;
        }
        require(failed, "Invalid distillation input accepted");
    };
    for (int fixture = 1; fixture <= 7; ++fixture) {
        int cell = std::min(fixture, 6);
        bool warm_adam = fixture == 7;
        fs::path out = base / ("cell-" + std::to_string(cell) + (warm_adam ? "-warm-adam" : ""));
        fs::create_directories(out);
        Config q{32, 64, 2, cell};
        Model student(q, 2, 16), a({24, 40, 1, 1}, 2, 16), b({40, 72, 2, 3}, 2, 16);
        auto weights = initialize(q, student.a, 123);
        activate_association_fixture(q, student.a, weights);
        student.w.put(weights);
        for (auto *teacher : {&a, &b}) {
            auto w = initialize(teacher->q, teacher->a, teacher == &a ? 91 : 312);
            for (int j = 0; j < 256; ++j)
                w[teacher->a.bias + j] = 2.f * std::sin(float(j) * (teacher == &a ? .31f : .23f));
            teacher->w.put(w);
            teacher->reset();
        }
        auto initial = student.membranes();
        for (size_t i = 0; i < initial.size(); ++i)
            initial[i] = .3f + .2f * std::sin(float(i) * .17f);
        student.membranes(initial);
        std::vector<int> x(32), y(32);
        std::vector<float> emphasis(32, 1.f);
        for (int i = 0; i < 32; ++i) {
            x[i] = (i * 37 + 91) % 256;
            y[i] = (i * 19 + 13) % 256;
            emphasis[i] = i % 16 >= 12 ? 64.f : (i % 3 == 0 ? 0.f : 1.f);
        }
        a.forward(x);
        b.forward(x);
        auto a_weights = a.w.host(), b_weights = b.w.host();
        auto a_state = a.membranes(), b_state = b.membranes();
        auto a_logits = a.logits.host(), b_logits = b.logits.host();
        float observed = student.forward(x, &y, true);
        auto original_gradient = student.dlogits.host(), final_state = student.membranes();
        float temperature = cell <= 2 ? .5f : (cell <= 4 ? 2.f : 4.f);
        const float strength = .7f, mixture = .3f;
        distillation::Targets targets(32, temperature);
        // Zero strength must recover the actual-target objective exactly, without
        // requiring a prepared teacher or touching the learner's recurrence.
        auto disabled = targets.apply(student, 0);
        require(disabled.observed_loss == observed && disabled.total_loss == observed &&
                    disabled.teacher_penalty == 0 && student.dlogits.host() == original_gradient,
                "Zero-strength distillation changed the observed objective");
        float weighted_observed = student.reweight_targets(emphasis);
        auto weighted_gradient = student.dlogits.host();
        student.dlogits.put(original_gradient);
        auto weighted_disabled = targets.apply(student, 0, emphasis);
        require(weighted_disabled.total_loss == weighted_observed &&
                    weighted_disabled.teacher_penalty == 0 && student.dlogits.host() == weighted_gradient,
                "Zero-strength distillation changed the weighted observed objective");
        student.dlogits.put(original_gradient);
        rejects([&]() { targets.apply(student, strength); });
        targets.from_logits(a.logits, &b.logits, mixture);
        auto probability = targets.probabilities();
        double largest_mass_error = 0;
        for (int row = 0; row < 32; ++row) {
            double mass = 0;
            for (int j = 0; j < 256; ++j) {
                float value = probability[row * 256 + j];
                require(std::isfinite(value) && value >= 0 && value <= 1, "Invalid teacher probability");
                mass += value;
            }
            largest_mass_error = std::max(largest_mass_error, std::abs(mass - 1));
        }
        require(largest_mass_error < 3e-7, "Teacher probabilities are not normalized");
        bool weighted = cell % 2 == 0;
        auto result = targets.apply(student, strength, weighted ? emphasis : std::vector<float>{});
        require(result.teacher_penalty > 0, "Fixture must exercise the teacher objective");
        dump(out / "weights.f32", weights);
        dump(out / "inputs.i32", x);
        dump(out / "targets.i32", y);
        if (weighted)
            dump(out / "target_weights.f32", emphasis);
        dump(out / "teacher-a-logits.f32", a_logits);
        dump(out / "teacher-b-logits.f32", b_logits);
        dump(out / "teacher-probabilities.f32", probability);
        dump(out / "initial_state.f32", initial);
        dump(out / "final_state.f32", final_state);
        dump(out / "logits.f32", student.logits.host());
        dump(out / "dlogits.f32", student.dlogits.host());
        student.backward();
        dump(out / "gradients.f32", student.g.host());
        student.backward(1);
        dump(out / "gradients_regularized.f32", student.g.host());
        student.backward();
        if (warm_adam) {
            std::vector<float> first(student.a.n), second(student.a.n);
            for (size_t i = 0; i < first.size(); ++i) {
                first[i] = .001f * std::sin(float(i) * .13f);
                second[i] = .0001f * (2.f + std::cos(float(i) * .17f));
            }
            student.m.put(first);
            student.v.put(second);
            dump(out / "initial_m.f32", first);
            dump(out / "initial_v.f32", second);
        }
        int adam_step = warm_adam ? 7 : 1;
        student.update(adam_step, .001f, 0, 0);
        dump(out / "updated.f32", student.w.host());
        require(student.membranes() == final_state && a.membranes() == a_state && b.membranes() == b_state &&
                    a.w.host() == a_weights && b.w.host() == b_weights && a.logits.host() == a_logits &&
                    b.logits.host() == b_logits &&
                    a.m.host() == std::vector<float>(a.a.n, 0.f) &&
                    a.v.host() == std::vector<float>(a.a.n, 0.f) &&
                    b.m.host() == std::vector<float>(b.a.n, 0.f) &&
                    b.v.host() == std::vector<float>(b.a.n, 0.f),
                "Student update mutated frozen teachers or recurrence");
        std::ofstream report(out / "fixture.json");
        report << std::setprecision(10) << "{\"channels\":32,\"hidden\":64,\"layers\":2,\"cell\":" << cell
               << ",\"batch\":2,\"context\":16,\"loss\":" << result.total_loss
               << ",\"parameters\":" << student.a.n << ",\"adam_step\":" << adam_step
               << ",\"distillation\":{\"temperature\":" << temperature
               << ",\"strength\":" << strength << ",\"mixture\":" << mixture
               << ",\"observed_loss\":" << result.observed_loss << ",\"penalty\":" << result.teacher_penalty
               << ",\"gpu_bytes\":" << targets.gpu_bytes() << ",\"probability_mass_error\":"
               << largest_mass_error << "},\"zero_strength_exact\":true,\"teachers_unchanged\":true}";
    }
    // Endpoint mixtures, stable large logits, self teacher, and invalid inputs.
    Model tiny({8, 8, 1, 1}, 1, 1);
    tiny.w.put(initialize(tiny.q, tiny.a, 77));
    std::vector<int> x{7}, y{11};
    tiny.forward(x, &y);
    auto hard_gradient = tiny.dlogits.host();
    distillation::Targets self(1, 2.f);
    self.from_logits(tiny.logits);
    auto same = self.apply(tiny, 1.f);
    require(std::abs(same.teacher_penalty) < 1e-6 && maxdiff(hard_gradient, tiny.dlogits.host()) < 1e-7,
            "Self teacher changed its own predictions");
    Buf a(256), b(256), bad_shape(255);
    std::vector<float> av(256, -1000.f), bv(256, -1000.f);
    av[3] = 1000;
    bv[211] = 1000;
    a.put(av);
    b.put(bv);
    distillation::Targets extreme(1, .25f);
    for (float mix : {0.f, .25f, 1.f}) {
        extreme.from_logits(a, &b, mix);
        auto p = extreme.probabilities();
        require(p[3] == mix && p[211] == 1 - mix && std::accumulate(p.begin(), p.end(), 0.f) == 1,
                "Extreme-logit mixture or mixture endpoint incorrect");
        tiny.forward(x, &y);
        auto result = extreme.apply(tiny, .7f);
        require(std::isfinite(result.total_loss), "Zero-probability targets produced a nonfinite loss");
    }
    rejects([&]() { distillation::Targets invalid(0, 1); });
    distillation::Targets wrong_student(2, 1);
    rejects([&]() { wrong_student.apply(tiny, 0); });
    for (float tau : {0.f, .24f, 17.f, std::numeric_limits<float>::quiet_NaN()})
        rejects([&]() { distillation::Targets invalid(1, tau); });
    rejects([&]() { extreme.from_logits(bad_shape); });
    rejects([&]() { extreme.from_logits(a, &bad_shape); });
    for (float mix : {-.1f, 1.1f, std::numeric_limits<float>::quiet_NaN()})
        rejects([&]() { extreme.from_logits(a, &b, mix); });
    for (float value : {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity()}) {
        av[9] = value;
        a.put(av);
        rejects([&]() { extreme.from_logits(a, &b); });
        rejects([&]() { extreme.apply(tiny, 1); });
    }
    self.from_logits(tiny.logits);
    auto before = tiny.dlogits.host();
    for (float strength : {-1.f, 101.f, std::numeric_limits<float>::quiet_NaN()})
        rejects([&]() { self.apply(tiny, strength); });
    rejects([&]() { self.apply(tiny, 1, {0}); });
    rejects([&]() { self.apply(tiny, 1, {-1}); });
    rejects([&]() { self.apply(tiny, 1, {1, 1}); });
    require(tiny.dlogits.host() == before, "Rejected settings changed the learner gradient");
    std::ofstream summary(base / "result.json");
    summary << "{\"passed\":true,\"cell_types\":6,\"fixtures\":7,\"zero_strength_exact\":true,"
               "\"frozen_teachers_unchanged\":true,\"unequal_teacher_dimensions\":true,"
               "\"extreme_logits_and_zero_probabilities\":true,\"mixture_endpoints\":true,"
               "\"self_teacher\":true,\"invalid_inputs_rejected\":true,"
               "\"live_policy_exercised\":false,\"learning_quality_claim\":false}";
    std::cout << "PASS distillation: six gradient fixtures, teacher isolation and objective controls\n";
}
