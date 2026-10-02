// Native population selection, inherited learning settings and bounded width mutation.
#pragma once
#include <sstream>

namespace evolution {
struct Rules {
    uint64_t corpus = 0;
    int batch = 8, context = 128, batches = 32;
    double size_cost = .02;
    void validate() const {
        require(batch >= 1 && batch <= 256 && context >= 1 && context <= 2048 && batches >= 1 &&
                    batches <= 10000 && std::isfinite(size_cost) && size_cost >= 0 && size_cost <= 10,
                "Invalid population evaluation protocol");
    }
    double score(double loss, size_t parameters) const {
        return loss + size_cost * double(parameters) / 1000000.;
    }
};
struct Member {
    std::string id, parent_a, parent_b;
    uint64_t generation = 0, hash_a = 0, hash_b = 0;
    double ceiling = 3., growth_chance = .1, setting_mutation_chance = .2;
};
bool valid_id(const std::string &id) {
    return !id.empty() && id.size() <= 64 && std::all_of(id.begin(), id.end(), [](char c) {
        return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '-' ||
               c == '_';
    });
}
void validate(const Member &m) {
    require(valid_id(m.id) && (m.parent_a.empty() || valid_id(m.parent_a)) &&
                (m.parent_b.empty() || valid_id(m.parent_b)) && m.generation < 1000000 &&
                std::isfinite(m.ceiling) && m.ceiling > 0 && m.ceiling < 100 &&
                std::isfinite(m.growth_chance) && m.growth_chance >= 0 && m.growth_chance <= 1 &&
                std::isfinite(m.setting_mutation_chance) && m.setting_mutation_chance >= 0 &&
                m.setting_mutation_chance <= 1,
            "Invalid population member");
    require(m.generation ? (!m.parent_a.empty() && !m.parent_b.empty() && m.parent_a != m.parent_b)
                         : (m.parent_a.empty() && m.parent_b.empty()),
            "Invalid member lineage");
}
Rules read_rules(const fs::path &path) {
    std::ifstream f(path);
    std::string magic, extra;
    Rules r;
    f >> magic >> r.corpus >> r.batch >> r.context >> r.batches >> r.size_cost;
    require(bool(f) && magic == "SGPOP1" && !(f >> extra), "Invalid population protocol file");
    r.validate();
    return r;
}
void write_rules(const fs::path &path, const Rules &r) {
    r.validate();
    std::ofstream f(path);
    f << std::setprecision(17) << "SGPOP1\n"
      << r.corpus << ' ' << r.batch << ' ' << r.context << ' ' << r.batches << ' ' << r.size_cost << '\n';
    f.close();
    require(bool(f), "Cannot write population protocol");
}
Member read_member(const fs::path &path) {
    std::ifstream f(path);
    Member m;
    std::string magic, extra;
    f >> magic >> std::quoted(m.id) >> m.generation >> m.ceiling >> m.growth_chance >>
        m.setting_mutation_chance >> std::quoted(m.parent_a) >> std::quoted(m.parent_b) >> m.hash_a >>
        m.hash_b;
    require(bool(f) && magic == "SGMEMBER1" && !(f >> extra), "Invalid member file");
    validate(m);
    return m;
}
void write_member(const fs::path &path, const Member &m) {
    validate(m);
    std::ofstream f(path);
    f << std::setprecision(17) << "SGMEMBER1\n"
      << std::quoted(m.id) << ' ' << m.generation << '\n'
      << m.ceiling << ' ' << m.growth_chance << ' ' << m.setting_mutation_chance << '\n'
      << std::quoted(m.parent_a) << ' ' << std::quoted(m.parent_b) << '\n'
      << m.hash_a << ' ' << m.hash_b << '\n';
    f.close();
    require(bool(f), "Cannot write population member");
}
struct Candidate {
    Member member;
    fs::path checkpoint;
    Config config;
    State state;
    size_t parameters = 0;
    double loss = 0, score = 0;
};
bool eligible(const Candidate &c, size_t limit) {
    return c.state.meta[7] > 0 && c.parameters <= limit && std::isfinite(c.score) &&
           c.score < c.member.ceiling;
}
std::vector<size_t> elites(const std::vector<Candidate> &members, double fraction, size_t limit) {
    require(std::isfinite(fraction) && fraction > 0 && fraction <= 1, "Invalid elite fraction");
    std::vector<size_t> result;
    for (size_t i = 0; i < members.size(); ++i)
        if (eligible(members[i], limit))
            result.push_back(i);
    std::sort(result.begin(), result.end(), [&](size_t a, size_t b) {
        return members[a].score == members[b].score ? members[a].member.id < members[b].member.id
                                                    : members[a].score < members[b].score;
    });
    size_t count = std::max(size_t(2), size_t(std::ceil(result.size() * fraction)));
    result.resize(std::min(count, result.size()));
    return result;
}
bool compatible(Config a, Config b) {
    return a.c == b.c && a.l == b.l && a.cell == b.cell;
}
Config child_config(Config a, Config b, double chance, int max_hidden, size_t max_parameters, uint64_t &rng) {
    require(compatible(a, b), "Parents need matching channels, layers and cell type");
    require(chance >= 0 && chance <= 1 && std::isfinite(chance) && max_hidden >= 8 && max_hidden <= 8192 &&
                rng,
            "Invalid width-mutation settings");
    a.h = std::max(a.h, b.h);
    require(a.h <= max_hidden && Layout(a).n <= max_parameters, "Inherited model exceeds size budget");
    bool attempt = uniform(rng) < chance;
    Config grown = a;
    int increase = std::max(8, ((a.h + 3) / 4 + 7) / 8 * 8);
    grown.h = std::min(max_hidden, a.h + increase);
    if (attempt && Layout(grown).n <= max_parameters)
        a = grown;
    return a;
}
// Whole residual blocks keep each neuron's input, leak/adaptation and output
// together. Shared channel bases can still drift between parents: crossover is
// experimental and every child must requalify. Padding alone preserves output.
std::vector<float> inherit(Config child, Config a, const std::vector<float> &wa, Config b,
                           const std::vector<float> &wb, const std::vector<bool> &donor, uint64_t seed) {
    require(compatible(a, b) && compatible(a, child) && child.h >= std::max(a.h, b.h) &&
                donor.size() == size_t(child.l),
            "Incompatible inheritance dimensions");
    Layout ca(child), aa(a), ba(b);
    require(wa.size() == aa.n && wb.size() == ba.n, "Parent weight count mismatch");
    auto result = initialize(child, ca, seed);
    auto copy = [&](size_t dest, const std::vector<float> &source, size_t at, size_t count) {
        std::copy_n(source.begin() + at, count, result.begin() + dest);
    };
    copy(ca.emb, wa, aa.emb, size_t(256) * a.c);
    copy(ca.final_gain, wa, aa.final_gain, a.c);
    copy(ca.head, wa, aa.head, size_t(256) * a.c);
    copy(ca.bias, wa, aa.bias, 256);
    for (int l = 0; l < child.l; ++l) {
        const auto &q = donor[l] ? b : a;
        const auto &source = donor[l] ? wb : wa;
        const auto &from = donor[l] ? ba.layers[l] : aa.layers[l];
        const auto &to = ca.layers[l];
        copy(to.gain, source, from.gain, q.c);
        copy(to.wi, source, from.wi, size_t(q.h) * q.c);
        copy(to.bi, source, from.bi, q.h);
        copy(to.bo, source, from.bo, q.c);
        copy(to.leak, source, from.leak, q.h);
        if (child.adaptive()) {
            copy(to.adapt_leak, source, from.adapt_leak, q.h);
            copy(to.adapt_scale, source, from.adapt_scale, q.h);
        }
        std::fill_n(result.begin() + to.wo, size_t(child.c) * child.h, 0.f);
        for (int c = 0; c < child.c; ++c)
            copy(to.wo + size_t(c) * child.h, source, from.wo + size_t(c) * q.h, q.h);
    }
    return result;
}
State child_state(const Candidate &a, const Candidate &b, const Rules &rules, uint64_t &rng) {
    State s;
    require(a.state.hp[0] > 0 && b.state.hp[0] > 0 && a.state.hp[0] <= .1f && b.state.hp[0] <= .1f,
            "Invalid parental learning rate");
    s.hp[0] = float(std::sqrt(double(a.state.hp[0]) * b.state.hp[0]));
    double mutation = .5 * (a.member.setting_mutation_chance + b.member.setting_mutation_chance);
    if (uniform(rng) < mutation)
        s.hp[0] *= float(std::exp((2 * uniform(rng) - 1) * std::log(2.)));
    s.hp[0] = std::clamp(s.hp[0], .000001f, .01f);
    for (int key : {1, 2, 4})
        s.hp[key] = .5f * (a.state.hp[key] + b.state.hp[key]);
    s.meta[5] = rules.batch;
    s.meta[6] = rules.context;
    s.meta[8] = 1000;
    s.meta[9] = 50;
    s.meta[10] = s.meta[11] = rnd(rng);
    s.meta[16] = a.state.meta[16];
    return s;
}
void add(const Args &args) {
    args.allow({"population", "checkpoint", "id", "data", "max-score", "growth-chance",
                "setting-mutation-chance", "batch", "context", "batches", "size-cost"});
    fs::path root = args.get("population", "runs/population");
    Member m;
    m.id = args.get("id");
    m.ceiling = args.real("max-score", 3.f);
    m.growth_chance = args.real("growth-chance", .1f);
    m.setting_mutation_chance = args.real("setting-mutation-chance", .2f);
    validate(m);
    require(!fs::exists(root / m.id), "Member ID already exists");
    Rules rules;
    bool existing = fs::exists(root / "population.sg");
    if (existing) {
        rules = read_rules(root / "population.sg");
        for (auto key : {"batch", "context", "batches", "size-cost"})
            require(args.get(key).empty(), "Existing population preserves evaluation protocol");
    } else {
        rules.batch = args.num("batch", 8);
        rules.context = args.num("context", 128);
        rules.batches = args.num("batches", 32);
        rules.size_cost = args.real("size-cost", .02f);
        rules.validate();
    }
    Data data(args.get("data"), rules.context);
    require(!existing || data.hash == rules.corpus, "Population evaluation corpus changed");
    rules.corpus = data.hash;
    State s = header(args.get("checkpoint"));
    Model model(checkpoint_config(s), rules.batch, rules.context);
    load(args.get("checkpoint"), model, s);
    require(s.meta[12] != rules.corpus, "Population evaluation corpus is also the training corpus");
    // A founder registration copies the supplied checkpoint exactly. Registration
    // does not establish eligibility: evolve remeasures and enforces its gate.
    fs::create_directories(root / m.id);
    fs::copy_file(args.get("checkpoint"), root / m.id / "latest.ckpt");
    write_member(root / m.id / "member.sg", m);
    if (!existing)
        write_rules(root / "population.sg", rules);
    std::cout << "Registered founder " << m.id << "; reproduction requires score < " << m.ceiling
              << " and at least one training update.\n";
}
void run(const Args &args) {
    args.allow({"population", "data", "round", "children", "seed", "elite-fraction", "max-hidden",
                "max-parameters", "min-improvement", "crossover"});
    fs::path root = args.get("population", "runs/population");
    std::string round = args.get("round");
    require(valid_id(round), "Round must be a short alphanumeric ID");
    int children = args.num("children", 2), max_hidden = args.num("max-hidden", 2048);
    int max_parameters = args.num("max-parameters", 4000000), seed = args.num("seed", 1337);
    double fraction = args.real("elite-fraction", .5f), improve = args.real("min-improvement", .005f),
           crossover = args.real("crossover", .25f);
    require(children >= 1 && children <= 128 && seed > 0 && max_parameters > 0 && improve > 0 &&
                improve < 1 && crossover >= 0 && crossover <= 1 && max_hidden >= 8 && max_hidden <= 8192,
            "Invalid evolution limits");
    require(!fs::exists(root / (round + ".json")), "Round already exists");
    for (int i = 0; i < children; ++i)
        require(!fs::exists(root / (round + "-child-" + std::to_string(i))), "Child ID already exists");
    auto rules = read_rules(root / "population.sg");
    Data data(args.get("data"), rules.context);
    require(data.hash == rules.corpus, "Population evaluation corpus changed");
    std::vector<fs::path> paths;
    for (const auto &entry : fs::directory_iterator(root))
        if (entry.is_directory() && fs::exists(entry.path() / "member.sg"))
            paths.push_back(entry.path());
    std::sort(paths.begin(), paths.end());
    std::vector<Candidate> members;
    for (const auto &path : paths) {
        Candidate c;
        c.member = read_member(path / "member.sg");
        require(path.filename().string() == c.member.id, "Member ID does not match its directory");
        c.checkpoint = path / "latest.ckpt";
        c.state = header(c.checkpoint);
        c.config = checkpoint_config(c.state);
        c.parameters = Layout(c.config).n;
        // Do not allocate a model larger than this invocation's resource budget.
        if (c.parameters <= size_t(max_parameters) && c.config.h <= max_hidden) {
            Model model(c.config, rules.batch, rules.context);
            load(c.checkpoint, model, c.state);
            require(c.state.meta[12] != rules.corpus,
                    "Member was trained on the population evaluation corpus");
            c.loss = evaluate(model, data, rules.batches);
            c.score = rules.score(c.loss, c.parameters);
        } else
            c.loss = c.score = 1e30;
        std::cout << c.member.id << " generation=" << c.member.generation << " loss=" << c.loss
                  << " score=" << c.score << " gate=" << c.member.ceiling
                  << " eligible=" << eligible(c, size_t(max_parameters)) << '\n';
        members.push_back(std::move(c));
    }
    auto pool = elites(members, fraction, size_t(max_parameters));
    std::vector<std::pair<size_t, size_t>> pairs;
    for (size_t i = 0; i < pool.size(); ++i)
        for (size_t j = i + 1; j < pool.size(); ++j)
            if (compatible(members[pool[i]].config, members[pool[j]].config))
                pairs.emplace_back(pool[i], pool[j]);
    require(!pairs.empty(), "Fewer than two compatible elite models qualify to reproduce");
    uint64_t rng = uint64_t(seed);
    std::ostringstream report;
    report << std::setprecision(10) << "{\"round\":" << std::quoted(round) << ",\"seed\":" << seed
           << ",\"evaluation_corpus_hash\":\"" << rules.corpus << "\",\"batch\":" << rules.batch
           << ",\"context\":" << rules.context << ",\"batches\":" << rules.batches
           << ",\"size_cost_per_million_parameters\":" << rules.size_cost
           << ",\"elite_fraction\":" << fraction << ",\"min_improvement\":" << improve
           << ",\"max_hidden\":" << max_hidden << ",\"max_parameters\":" << max_parameters
           << ",\"crossover\":" << crossover << ",\"members\":[";
    for (size_t i = 0; i < members.size(); ++i) {
        const auto &m = members[i];
        report << (i ? "," : "") << "{\"id\":" << std::quoted(m.member.id)
               << ",\"generation\":" << m.member.generation << ",\"step\":" << m.state.meta[7]
               << ",\"loss\":" << m.loss << ",\"score\":" << m.score << ",\"ceiling\":" << m.member.ceiling
               << ",\"parameters\":" << m.parameters
               << ",\"eligible\":" << (eligible(m, size_t(max_parameters)) ? "true" : "false")
               << ",\"elite\":" << (std::find(pool.begin(), pool.end(), i) != pool.end() ? "true" : "false")
               << '}';
    }
    report << "],\"children\":[";
    for (int i = 0; i < children; ++i) {
        auto pair = pairs[size_t(rnd(rng) % pairs.size())];
        const auto &a = members[pair.first]; // Elites are sorted: a has the better score.
        const auto &b = members[pair.second];
        Member child;
        child.id = round + "-child-" + std::to_string(i);
        child.parent_a = a.member.id;
        child.parent_b = b.member.id;
        child.generation = 1 + std::max(a.member.generation, b.member.generation);
        child.ceiling = std::min(a.score, b.score) - improve;
        child.growth_chance = .5 * (a.member.growth_chance + b.member.growth_chance);
        child.setting_mutation_chance =
            .5 * (a.member.setting_mutation_chance + b.member.setting_mutation_chance);
        child.hash_a = a.state.meta[15];
        child.hash_b = b.state.meta[15];
        validate(child);
        Config q =
            child_config(a.config, b.config, child.growth_chance, max_hidden, size_t(max_parameters), rng);
        std::vector<bool> donor(size_t(q.l));
        for (int l = 0; l < q.l; ++l)
            donor[l] = uniform(rng) < crossover;
        if (crossover > 0 && std::none_of(donor.begin(), donor.end(), [](bool x) { return x; }))
            donor[size_t(rnd(rng) % donor.size())] = true;
        State sa, sb;
        Model ma(a.config, 1, 1), mb(b.config, 1, 1);
        load(a.checkpoint, ma, sa);
        load(b.checkpoint, mb, sb);
        auto weights = inherit(q, a.config, ma.w.host(), b.config, mb.w.host(), donor, rnd(rng));
        auto state = child_state(a, b, rules, rng);
        Model model(q, rules.batch, rules.context);
        model.w.put(weights); // Fresh optimizer and recurrent state at birth.
        auto directory = root / child.id;
        fs::create_directories(directory);
        save(directory / "initial.ckpt", model, state);
        save(directory / "latest.ckpt", model, state);
        write_member(directory / "member.sg", child);
        double birth_loss = evaluate(model, data, rules.batches);
        report << (i ? "," : "") << "{\"id\":" << std::quoted(child.id)
               << ",\"generation\":" << child.generation << ",\"parent_a\":" << std::quoted(child.parent_a)
               << ",\"parent_b\":" << std::quoted(child.parent_b) << ",\"parent_a_payload_hash\":\""
               << child.hash_a << "\",\"parent_b_payload_hash\":\"" << child.hash_b << "\",\"hidden\":" << q.h
               << ",\"parameters\":" << model.a.n
               << ",\"grew\":" << (q.h > std::max(a.config.h, b.config.h) ? "true" : "false")
               << ",\"growth_chance\":" << child.growth_chance
               << ",\"setting_mutation_chance\":" << child.setting_mutation_chance
               << ",\"learning_rate\":" << state.hp[0] << ",\"activity_cost\":" << state.hp[4]
               << ",\"birth_loss\":" << birth_loss << ",\"required_score_below\":" << child.ceiling
               << ",\"eligible_at_birth\":false,\"donor_blocks\":[";
        for (size_t l = 0; l < donor.size(); ++l)
            report << (l ? "," : "") << (donor[l] ? "true" : "false");
        report << "]}";
        std::cout << "Born " << child.id << " generation=" << child.generation << " hidden=" << q.h
                  << " parameters=" << model.a.n << " lr=" << state.hp[0] << "; train and beat score "
                  << child.ceiling << " before reproduction.\n";
    }
    report << "]}\n";
    std::ofstream f(root / (round + ".json"));
    f << report.str();
    f.close();
    require(bool(f), "Cannot write evolution report");
}
} // namespace evolution
