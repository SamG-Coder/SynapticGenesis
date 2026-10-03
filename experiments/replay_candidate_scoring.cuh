// Read-only scoring around a real live source update, using production forward.
#pragma once
#include "replay_score_graph.cuh"
namespace replay_priority {
double seconds(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}
uint64_t tensor_identity(const Buf &buffer, uint64_t hash = 14695981039346656037ull) {
    auto values = buffer.host();
    return hash_bytes(values.data(), values.size() * sizeof(float), hash);
}
uint64_t learned_identity(const Model &model) {
    auto memory = model.membranes();
    return hash_bytes(memory.data(), memory.size() * sizeof(float),
                      tensor_identity(model.v, tensor_identity(model.m, tensor_identity(model.w))));
}
uint64_t file_identity(const fs::path &path) {
    std::ifstream input(path, std::ios::binary);
    require(bool(input), "Cannot read probe identity input");
    std::string value((std::istreambuf_iterator<char>(input)), {});
    require(value.size() == fs::file_size(path), "Incomplete probe identity read");
    return hash_bytes(value.data(), value.size());
}
uint64_t choose(uint64_t &rng, uint64_t bound) {
    require(bound > 0, "Empty candidate range");
    uint64_t minimum = (~bound + 1) % bound, value;
    do {
        value = rnd(rng);
    } while (value < minimum);
    return value % bound;
}
struct Score {
    double mean = 0, weighted = 0, answer = 0;
    size_t answers = 0;
};
struct Candidate {
    uint64_t slot, group;
    Episode episode;
    Score before;
};
class CandidateScorer {
    bool audit_, graph_;
    // Scoring owns recurrence, activations and strict-FP32 handles. It never
    // changes a learning replay view's cuBLAS mode or live neuron state.
    std::map<int, std::unique_ptr<Model>> scorers_;
    // Graphs must be released before their model buffers.
    std::map<int, std::unique_ptr<ScoreGraph>> graphs_;

  public:
    uint64_t forward_calls = 0, scored_pairs = 0;
    uint64_t resident_view_allocation_bytes = 0, max_transient_allocation_bytes = 0;
    uint64_t owned_array_bytes = 0, view_count = 0;
    double setup_seconds = 0, scoring_seconds = 0, audit_seconds = 0;
    CandidateScorer(bool audit, bool graph) : audit_(audit), graph_(graph) {}
    std::vector<Score> score(LiveEngine &engine, const LiveCorpus &data, const std::vector<Candidate> &pool_) {
        auto setup_start = std::chrono::steady_clock::now();
        for (const auto &candidate : pool_) {
            int n = int(candidate.episode.length);
            if (!scorers_.count(n)) {
                size_t free_before, total;
                ck(cudaMemGetInfo(&free_before, &total));
                auto model = std::make_unique<Model>(engine.root.q, 1, n);
                size_t free_during;
                ck(cudaMemGetInfo(&free_during, &total));
                max_transient_allocation_bytes = std::max(max_transient_allocation_bytes,
                    uint64_t(free_before > free_during ? free_before - free_during : 0));
                model->share_parameters(engine.root);
                owned_array_bytes += explicit_model_bytes(*model) -
                    (model->w.n + model->m.n + model->v.n) * sizeof(float);
                ++view_count;
                if (graph_)
                    graphs_.emplace(n, std::make_unique<ScoreGraph>(*model));
                scorers_.emplace(n, std::move(model));
                size_t free_after;
                ck(cudaMemGetInfo(&free_after, &total));
                resident_view_allocation_bytes += free_before > free_after ? free_before - free_after : 0;
            }
        }
        ck(cudaDeviceSynchronize());
        setup_seconds += seconds(setup_start);
        auto audit_start = std::chrono::steady_clock::now();
        uint64_t identity = audit_ ? learned_identity(engine.root) : 0;
        audit_seconds += seconds(audit_start);
        auto start = std::chrono::steady_clock::now();
        std::vector<Score> result;
        for (const auto &candidate : pool_) {
            auto e = candidate.episode;
            ReplayMemory::validate_episode(e, data, engine.root.T);
            size_t at = data.docs[size_t(e.document)].first + size_t(e.offset), n = size_t(e.length);
            std::vector<int> x(data.bytes.begin() + at, data.bytes.begin() + at + n);
            std::vector<int> y(data.bytes.begin() + at + 1, data.bytes.begin() + at + n + 1);
            auto &model = *scorers_.at(int(n));
            std::vector<float> values;
            if (graph_)
                values = graphs_.at(int(n))->score(x, y);
            else {
                model.forward(x, &y); // Same forward, reset-state replay window.
                values = model.losses.host();
            }
            auto weights = data.target_weights(size_t(e.document), size_t(e.offset), n);
            Score value;
            double mass = 0;
            for (size_t i = 0; i < n; ++i) {
                require(std::isfinite(values[i]) && values[i] >= 0, "Invalid replay candidate loss");
                double weight = weights.empty() ? 1 : weights[i];
                value.mean += values[i];
                value.weighted += weight * values[i];
                mass += weight;
                if (weight > 1) {
                    value.answer += values[i];
                    ++value.answers;
                }
            }
            value.mean /= n;
            value.weighted /= mass;
            if (value.answers)
                value.answer /= value.answers;
            result.push_back(value);
            ++forward_calls;
            scored_pairs += n;
        }
        ck(cudaDeviceSynchronize());
        scoring_seconds += seconds(start);
        audit_start = std::chrono::steady_clock::now();
        if (audit_)
            require(identity == learned_identity(engine.root), "Candidate scoring changed learning/live state");
        audit_seconds += seconds(audit_start);
        return result;
    }
};
} // namespace replay_priority
