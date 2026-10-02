#pragma once
void evolution_test(const Args &args) {
    args.allow({"out"});
    fs::path out = args.get("out", "reports/evolution-tests");
    fs::create_directories(out);
    double growth_error = 0, crossover_error = 0, new_weight_change = 0;
    for (int cell : {1, 2}) {
        Config a{8, 16, 2, cell}, b{8, 24, 2, cell}, c{8, 32, 2, cell};
        Layout aa(a), ba(b), ca(c);
        auto wa = initialize(a, aa, 19);
        auto wb = evolution::inherit(b, a, wa, a, wa, {false, false}, 31);
        Model base(a, 1, 8), widened(b, 1, 8);
        uint64_t actual_buffers = 2ull * base.N * sizeof(int);
        for (const Buf *buffer :
             {&base.w, &base.g, &base.m, &base.v, &base.decay, &base.finalnorm, &base.finalrs, &base.logits,
              &base.dlogits, &base.losses, &base.dx, &base.dy, &base.dnorm, &base.ds, &base.dz})
            actual_buffers += buffer->n * 4;
        for (const auto &buffer : base.x)
            actual_buffers += buffer.n * 4;
        for (const auto &cache : base.cache)
            for (const Buf *buffer :
                 {&cache.norm, &cache.rs, &cache.z, &cache.u, &cache.s, &cache.state, &cache.initial_state,
                  &cache.adapt, &cache.adapt_state, &cache.initial_adapt})
                actual_buffers += buffer->n * 4;
        require(actual_buffers == evolution::working_bytes(a, 1, 8),
                "GPU admission estimate differs from actual explicit buffers");
        base.w.put(wa);
        widened.w.put(wb);
        std::vector<int> x{'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'}, y{'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i'};
        for (int repeat = 0; repeat < 3; ++repeat) {
            base.forward(x, &y, true);
            widened.forward(x, &y, true);
            growth_error = std::max(growth_error, double(maxdiff(base.logits.host(), widened.logits.host())));
        }
        require(growth_error < 3e-5, "Width expansion changed inherited behavior");
        auto before = widened.w.host();
        widened.backward();
        widened.update(1, .001f, 0);
        auto after = widened.w.host();
        double cell_change = 0;
        for (int l = 0; l < a.l; ++l)
            for (int row = 0; row < b.c; ++row)
                for (int neuron = a.h; neuron < b.h; ++neuron) {
                    size_t index = ba.layers[l].wo + size_t(row) * b.h + neuron;
                    require(before[index] == 0, "New neuron output was not neutral at birth");
                    cell_change = std::max(cell_change, double(std::abs(after[index])));
                }
        require(cell_change > 1e-6, "New neurons cannot learn outgoing connections");
        new_weight_change = std::max(new_weight_change, cell_change);
        // Only donor block 1 changes, so the expected mixed model is this donor
        // itself. Its other block, embedding and readout still match parent A.
        const auto &block = ba.layers[1];
        for (size_t i = 0; i < size_t(b.c) * b.h; ++i)
            wb[block.wo + i] *= 1.2f;
        wb[block.bo] += .03f;
        auto wc = evolution::inherit(c, a, wa, b, wb, {false, true}, 55);
        Model expected(b, 1, 8), mixed(c, 1, 8);
        expected.w.put(wb);
        mixed.w.put(wc);
        for (int repeat = 0; repeat < 3; ++repeat) {
            expected.forward(x, &y, true);
            mixed.forward(x, &y, true);
            crossover_error =
                std::max(crossover_error, double(maxdiff(expected.logits.host(), mixed.logits.host())));
        }
        require(crossover_error < 3e-5, "Whole-block inheritance differs from expected donor behavior");
        require(maxdiff(base.logits.host(), mixed.logits.host()) > 1e-5, "Donor block had no effect");
        State state;
        state.meta[10] = state.meta[11] = 19;
        save(out / (cell == 1 ? "lif-child.ckpt" : "alif-child.ckpt"), mixed, state);
        Model loaded(c, 1, 8);
        State restored;
        load(out / (cell == 1 ? "lif-child.ckpt" : "alif-child.ckpt"), loaded, restored);
        require(restored.meta[7] == 0 && maxdiff(wc, loaded.w.host()) == 0,
                "Child birth checkpoint did not roundtrip");
        auto moments = loaded.m.host(), variances = loaded.v.host();
        require(std::all_of(moments.begin(), moments.end(), [](float x) { return x == 0; }) &&
                    std::all_of(variances.begin(), variances.end(), [](float x) { return x == 0; }),
                "Child inherited parental optimizer history");
    }
    Config small{8, 16, 2};
    uint64_t r0 = 42, r1 = 42, r2 = 42;
    auto unchanged = evolution::child_config(small, small, 0, 64, 100000, r0);
    auto grown = evolution::child_config(small, small, 1, 64, 100000, r1);
    auto capped = evolution::child_config(small, small, 1, 64, Layout(small).n, r2);
    require(unchanged.h == small.h && grown.h > small.h && capped.h == small.h,
            "Width mutation chance or parameter cap failed");
    r1 = 123;
    r2 = 123;
    for (int i = 0; i < 100; ++i)
        require(evolution::child_config(small, small, .3, 64, 100000, r1).h ==
                    evolution::child_config(small, small, .3, 64, 100000, r2).h,
                "Seeded mutation decisions differ");
    std::vector<evolution::Candidate> members(5);
    for (size_t i = 0; i < members.size(); ++i) {
        auto &m = members[i];
        m.member.id = "member-" + std::to_string(i);
        m.member.ceiling = 2;
        m.parameters = 100;
        m.state.meta[7] = 100;
        m.score = 1 + i * .1;
    }
    members[0].state.meta[7] = 0;  // Better score does not make an untrained newborn eligible.
    members[3].member.ceiling = 1; // Fails its lineage-specific improvement requirement.
    members[4].parameters = 200;   // Over budget.
    auto selected = evolution::elites(members, .5, 150);
    require(selected == std::vector<size_t>({1, 2}), "Reproduction gate admitted an ineligible model");
    for (auto &m : members) {
        m.state.meta[7] = 100;
        m.parameters = 100;
        m.member.ceiling = 2;
    }
    selected = evolution::elites(members, .4, 150);
    require(selected == std::vector<size_t>({0, 1}), "Elite ranking admitted a weaker model");
    members[0].alive = false;
    selected = evolution::elites(members, .4, 150);
    require(selected == std::vector<size_t>({1, 2}), "Dead model was allowed to reproduce");
    members[0].state.hp[0] = .0002f;
    members[1].state.hp[0] = .0008f;
    members[0].member.setting_mutation_chance = members[1].member.setting_mutation_chance = 0;
    evolution::Rules rules;
    rules.corpus = 42;
    rules.tick = 11;
    r0 = 88;
    auto newborn = evolution::child_state(members[0], members[1], rules, r0);
    require(std::abs(newborn.hp[0] - .0004f) < 1e-9 && newborn.meta[7] == 0 && newborn.extra.empty() &&
                newborn.synaptic.empty(),
            "Parental settings or fresh child history are incorrect");
    members[0].state.meta[17] = members[1].state.meta[17] = 4;
    members[0].state.hp[7] = .0002f;
    members[1].state.hp[7] = .0008f;
    members[0].state.hp[0] *= .25f;
    members[1].state.hp[0] *= .1f;
    members[0].state.hp[6] = .25f;
    members[1].state.hp[6] = .5f;
    r0 = 88;
    auto young = evolution::child_state(members[0], members[1], rules, r0);
    require(young.hp[0] == newborn.hp[0] && young.meta[17] == 0 && young.meta[24] == 0 && young.hp[7] == 0,
            "Newborn inherited its parents' developmental slowdown or age");
    members[0].state.hp[7] = 0;
    bool invalid_rate = false;
    try {
        evolution::child_state(members[0], members[1], rules, r0);
    } catch (const std::exception &) {
        invalid_rate = true;
    }
    require(invalid_rate, "Invalid parental base rate was accepted");
    evolution::write_rules(out / "population.sg", rules);
    auto restored_rules = evolution::read_rules(out / "population.sg");
    require(restored_rules.corpus == rules.corpus && restored_rules.size_cost == rules.size_cost &&
                restored_rules.tick == 11,
            "Population protocol roundtrip failed");
    evolution::Member member;
    member.id = "child";
    member.parent_a = "parent-a";
    member.parent_b = "parent-b";
    member.generation = 2;
    member.ceiling = 1.125;
    member.born_tick = 3;
    member.lifespan = 2;
    evolution::write_member(out / "member.sg", member);
    auto restored_member = evolution::read_member(out / "member.sg");
    require(restored_member.generation == 2 && restored_member.ceiling == 1.125 &&
                restored_member.parent_a == "parent-a" && restored_member.born_tick == 3 &&
                restored_member.lifespan == 2,
            "Lineage record roundtrip failed");
    require(evolution::alive(restored_member, 4) && !evolution::alive(restored_member, 5) &&
                !evolution::alive(restored_member, 6),
            "Old-age boundary is incorrect");
    evolution::Food ample, scarce;
    ample.capacity = scarce.capacity = 1000;
    ample.used = 100;
    scarce.used = 900;
    require(ample.fits(900) && !ample.fits(901) && !scarce.fits(101),
            "Resource admission exceeded available credits");
    require(scarce.elite_fraction(.5) < ample.elite_fraction(.5) &&
                scarce.improvement(.005) > ample.improvement(.005),
            "Scarcity did not tighten selection");
    scarce.capacity = 0;
    require(!scarce.fits(1) && scarce.pressure() == 1, "Empty food budget permitted birth");
    std::ofstream f(out / "native.json");
    f << std::setprecision(10)
      << "{\"passed\":true,\"both_cells\":true,\"width_growth_max_error\":" << growth_error
      << ",\"block_inheritance_max_error\":" << crossover_error
      << ",\"new_neuron_output_weight_change\":" << new_weight_change
      << ",\"mutation_caps_and_seed_checked\":true,\"newborn_gate_checked\":true,"
         "\"improvement_gate_checked\":true,\"elite_ranking_checked\":true,"
         "\"parent_setting_inheritance_checked\":true,\"lineage_roundtrip_checked\":true,"
         "\"base_rate_inherited_without_developmental_slowdown\":true,"
         "\"old_age_and_dead_parent_gate_checked\":true,\"scarcity_and_admission_checked\":true,"
         "\"gpu_buffer_estimate_checked\":true}\n";
    std::cout << "PASS evolution: growth, inherited blocks, new-neuron learning, bounded seeded mutation, "
                 "fitness gates, settings and lineage. Growth max error "
              << growth_error << '\n';
}
