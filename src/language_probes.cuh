// Read-only paired language probes. Answers change when facts in context change;
// the query and candidate strings stay identical within each pair.
#pragma once
#include "live_views.cuh"
namespace probes {
std::string json(const std::string &value) {
    std::ostringstream out;
    out << '"';
    for (unsigned char c : value) {
        if (c == '"' || c == '\\')
            out << '\\' << char(c);
        else if (c < 32 || c >= 127)
            out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c) << std::dec;
        else
            out << char(c);
    }
    out << '"';
    return out.str();
}
struct Item {
    std::string id, pair, skill, context, query;
    std::array<std::string, 2> choices;
    int correct = 0;
};
struct Suite {
    std::vector<Item> items;
    std::map<std::string, std::vector<size_t>> pairs;
    uint64_t hash = 0;
    std::string format;
    size_t group_size = 2;
    explicit Suite(const fs::path &path) {
        require(fs::is_regular_file(path) && fs::file_size(path) <= 64 * 1024 * 1024,
                "Missing or oversized probe file");
        std::ifstream file(path, std::ios::binary);
        std::string bytes((std::istreambuf_iterator<char>(file)), {});
        hash = hash_bytes(bytes.data(), bytes.size());
        std::istringstream in(bytes);
        std::string magic;
        size_t count = 0;
        in >> magic >> count;
        format = magic;
        group_size = magic == "SGPROBE2" ? 4 : 2;
        require(bool(in) && (magic == "SGPROBE1" || magic == "SGPROBE2") && count >= group_size &&
                    count <= 10000 && count % group_size == 0,
                "Invalid probe header");
        std::map<std::string, bool> ids;
        for (size_t i = 0; i < count; ++i) {
            Item item;
            in >> std::quoted(item.id) >> std::quoted(item.pair) >> std::quoted(item.skill) >> item.correct >>
                std::quoted(item.context) >> std::quoted(item.query) >> std::quoted(item.choices[0]) >>
                std::quoted(item.choices[1]);
            require(bool(in) && !item.id.empty() && item.id.size() <= 128 && !ids.count(item.id) &&
                        !item.pair.empty() && item.pair.size() <= 128 && !item.skill.empty() &&
                        item.skill.size() <= 128 && item.correct >= 0 && item.correct < 2 &&
                        !item.context.empty() && item.context.size() <= 2048 && !item.query.empty() &&
                        item.query.size() <= 512 && !item.choices[0].empty() && !item.choices[1].empty() &&
                        item.choices[0].size() <= 128 && item.choices[1].size() <= 128 &&
                        item.choices[0] != item.choices[1],
                    "Invalid or duplicate probe item");
            ids[item.id] = true;
            pairs[item.pair].push_back(items.size());
            items.push_back(std::move(item));
        }
        std::string extra;
        require(!(in >> extra), "Unexpected trailing probe content");
        for (const auto &entry : pairs) {
            require(entry.second.size() == group_size, "Incorrect probe group size");
            const auto &a = items[entry.second[0]], &b = items[entry.second[1]];
            if (group_size == 4) {
                std::map<std::string, std::map<std::string, int>> grid;
                for (auto index : entry.second) {
                    const auto &item = items[index];
                    require(item.choices == a.choices && item.skill == a.skill &&
                                !grid[item.context].count(item.query),
                            "Invalid or duplicate binding context/query");
                    grid[item.context][item.query] = item.correct;
                }
                require(grid.size() == 2, "Binding group requires two contexts");
                const auto &first = grid.begin()->second, &second = std::next(grid.begin())->second;
                require(first.size() == 2 && second.size() == 2 &&
                            first.begin()->second != std::next(first.begin())->second,
                        "Binding group requires two opposite queries per context");
                for (const auto &query : first)
                    require(second.count(query.first) && second.at(query.first) != query.second,
                            "Binding group must reverse each query across contexts");
                continue;
            }
            require(a.context != b.context && a.query == b.query && a.choices == b.choices &&
                        a.correct != b.correct && a.skill == b.skill,
                    "Probe pairs must reverse answers using different context and identical queries/choices");
        }
    }
};
struct Scorer {
    Model root;
    LiveViews views;
    explicit Scorer(Config q) : root(q, 1, 1) {}
    Model &view(int n) {
        if (n == 1)
            return root;
        return views.get(root, n, true, false);
    }
    double score(const std::string &prompt, const std::string &answer) {
        // Prompt losses never contribute to the score. The first answer byte is
        // predicted from the final prompt byte, before seeing any answer bytes.
        std::string text = prompt + answer;
        std::vector<int> x(text.begin(), text.end() - 1), y(text.begin() + 1, text.end());
        for (auto &v : x)
            v = static_cast<unsigned char>(v);
        for (auto &v : y)
            v = static_cast<unsigned char>(v);
        auto &m = view(int(x.size()));
        m.forward(x, &y); // Independent zero-state candidate, strict FP32.
        auto loss = m.losses.host();
        return std::accumulate(loss.begin() + prompt.size() - 1, loss.end(), 0.0);
    }
    std::string greedy(const std::string &prompt, size_t length) {
        root.reset();
        std::vector<int> x;
        for (unsigned char c : prompt)
            x.push_back(c);
        auto &prefix = view(int(x.size()));
        prefix.forward(x, nullptr, true);
        auto all = prefix.logits.host();
        std::vector<float> logits(all.end() - 256, all.end());
        std::string result;
        for (size_t i = 0; i < length; ++i) {
            int byte = int(std::max_element(logits.begin(), logits.end()) - logits.begin());
            result.push_back(char(byte));
            if (i + 1 != length) {
                root.forward({byte}, nullptr, true);
                logits = root.logits.host();
            }
        }
        return result;
    }
};
int winner(const std::array<double, 2> &scores) {
    if (std::abs(scores[0] - scores[1]) <= 1e-8)
        return -1; // A numerical tie is not a successful answer.
    return scores[0] < scores[1] ? 0 : 1;
}
void run(const Args &args) {
    args.allow({"checkpoint", "probes", "output"});
    fs::path output = args.get("output", "reports/language-probes.json");
    require(!fs::exists(output), "Probe output exists; choose a new output file");
    Suite suite(args.get("probes"));
    State state = header(args.get("checkpoint"));
    Scorer scorer(checkpoint_config(state));
    load(args.get("checkpoint"), scorer.root, state);
    auto before_w = scorer.root.w.host(), before_m = scorer.root.m.host(), before_v = scorer.root.v.host();
    std::ostringstream rows;
    rows << std::setprecision(12);
    std::vector<bool> correct, erased, exact;
    std::map<std::string, std::array<size_t, 4>> skills;
    double nll = 0;
    size_t targets = 0;
    auto start = std::chrono::steady_clock::now();
    for (const auto &item : suite.items) {
        std::array<double, 2> full, no_context;
        for (size_t j = 0; j < 2; ++j) {
            full[j] = scorer.score(item.context + item.query, item.choices[j]);
            no_context[j] = scorer.score(item.query, item.choices[j]);
        }
        // Generation budget is independent of the gold label. Unequal choices
        // are allowed for scoring, but exact generation requires this full budget.
        auto answer = scorer.greedy(item.context + item.query,
                                    std::max(item.choices[0].size(), item.choices[1].size()));
        correct.push_back(winner(full) == item.correct);
        erased.push_back(winner(no_context) == item.correct);
        exact.push_back(answer == item.choices[item.correct]);
        auto &skill = skills[item.skill];
        ++skill[0];
        skill[1] += correct.back();
        skill[2] += erased.back();
        skill[3] += exact.back();
        nll += full[item.correct];
        targets += item.choices[item.correct].size();
        if (correct.size() > 1)
            rows << ',';
        rows << "{\"id\":" << json(item.id) << ",\"pair\":" << json(item.pair)
             << ",\"skill\":" << json(item.skill) << ",\"gold\":" << item.correct << ",\"candidate_nll\":["
             << full[0] << ',' << full[1] << "],\"context_erased_nll\":[" << no_context[0] << ','
             << no_context[1] << "],\"prediction\":" << winner(full)
             << ",\"context_erased_prediction\":" << winner(no_context) << ",\"greedy\":" << json(answer)
             << '}';
    }
    ck(cudaDeviceSynchronize());
    double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    size_t pair_correct = 0, pair_erased = 0, pair_exact = 0;
    for (const auto &entry : suite.pairs) {
        auto all = [&](const std::vector<bool> &values) {
            return std::all_of(entry.second.begin(), entry.second.end(), [&](size_t i) { return values[i]; });
        };
        pair_correct += all(correct);
        pair_erased += all(erased);
        pair_exact += all(exact);
    }
    require(pair_erased == 0, "Context-erased paired probe leaked a label or mutable state");
    require(before_w == scorer.root.w.host() && before_m == scorer.root.m.host() &&
                before_v == scorer.root.v.host(),
            "Read-only probes modified model parameters or optimizer history");
    if (!output.parent_path().empty())
        fs::create_directories(output.parent_path());
    std::ofstream report(output, std::ios::binary);
    auto fraction = [](const std::vector<bool> &values) {
        return double(std::count(values.begin(), values.end(), true)) / values.size();
    };
    report << std::setprecision(12) << "{\"format\":" << json(suite.format) << ",\"suite_hash\":\""
           << suite.hash << "\",\"checkpoint_payload_hash\":\"" << state.meta[15]
           << "\",\"items\":" << suite.items.size() << ",\"groups\":" << suite.pairs.size()
           << ",\"group_size\":" << suite.group_size
           << ",\"joint_accuracy\":" << double(pair_correct) / suite.pairs.size()
           << ",\"context_erased_joint_accuracy\":0,\"greedy_exact_joint_accuracy\":"
           << double(pair_exact) / suite.pairs.size();
    if (suite.group_size == 2)
        report << ",\"pairs\":" << suite.pairs.size()
               << ",\"paired_accuracy\":" << double(pair_correct) / suite.pairs.size()
               << ",\"context_erased_paired_accuracy\":0,\"greedy_exact_paired_accuracy\":"
               << double(pair_exact) / suite.pairs.size();
    report << ",\"accuracy\":" << fraction(correct) << ",\"context_erased_accuracy\":" << fraction(erased)
           << ",\"greedy_exact_accuracy\":" << fraction(exact)
           << ",\"answer_loss_nats_per_byte\":" << nll / targets << ",\"answer_bytes\":" << targets
           << ",\"elapsed_seconds\":" << seconds
           << ",\"parameters_and_optimizer_unchanged\":true,\"strict_fp32\":true,\"skills\":{";
    size_t index = 0;
    for (const auto &entry : skills) {
        const auto &v = entry.second;
        if (index++)
            report << ',';
        report << json(entry.first) << ":{\"items\":" << v[0] << ",\"accuracy\":" << double(v[1]) / v[0]
               << ",\"context_erased_accuracy\":" << double(v[2]) / v[0]
               << ",\"greedy_exact_accuracy\":" << double(v[3]) / v[0] << '}';
    }
    report << "},\"results\":[" << rows.str() << "]}\n";
    report.close();
    require(bool(report), "Cannot write probe report");
    std::cout << "language probes accuracy=" << fraction(correct)
              << " all-in-group=" << double(pair_correct) / suite.pairs.size()
              << " greedy-exact=" << fraction(exact) << " erased=" << fraction(erased)
              << " seconds=" << seconds << '\n';
}
} // namespace probes
