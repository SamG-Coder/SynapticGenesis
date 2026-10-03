// Actual-update diagnostic journal; candidate scoring is shared with experiments.
#pragma once
#include "replay_candidate_scoring.cuh"
namespace replay_priority {
class Observer : public LiveSourceObserver, public CandidateScorer {
    fs::path out_;
    std::ofstream log_;
    uint64_t every_, per_group_, seed_;
    bool reverse_, snapshots_, selected_ = false;
    std::vector<Candidate> pool_;
    void snapshot(Model &model, const State &state, const std::string &name) {
        State copy;
        copy.meta[7] = state.meta[7];
        copy.meta[11] = state.meta[11];
        copy.meta[19] = 0x3152505347424full; // Disposable scoring snapshot.
        copy.hp[0] = state.hp[0];
        save(out_ / name, model, copy); // Ordinary weights/moments, no live state.
    }
    static void print(std::ostream &out, const Score &value) {
        out << "{\"mean\":" << value.mean << ",\"weighted\":" << value.weighted
            << ",\"answer_targets\":" << value.answers << ",\"answer\":";
        if (value.answers)
            out << value.answer;
        else
            out << "null";
        out << '}';
    }

  public:
    uint64_t observations = 0;
    double snapshot_seconds = 0;
    Observer(const fs::path &out, uint64_t every, uint64_t per_group, uint64_t seed,
             bool audit, bool reverse, bool snapshots, bool graph = false)
        : CandidateScorer(audit, graph), out_(out), log_(out / "scores.jsonl"), every_(every), per_group_(per_group), seed_(seed),
          reverse_(reverse), snapshots_(snapshots) {
        require(bool(log_), "Cannot create replay score journal");
        log_ << std::setprecision(17);
    }
    void before_source(LiveEngine &engine, const LiveCorpus &data, const State &s, const Episode &) override {
        selected_ = every_ && s.meta[24] % every_ == 0;
        if (!selected_)
            return;
        StageReplayView memory(s);
        pool_.clear();
        uint64_t rng = seed_ ^ ((s.meta[24] + 1) * 0x9e3779b97f4a7c15ull);
        if (!rng)
            rng = 1;
        uint64_t first = 0;
        for (size_t group = 0; group < memory.groups(); ++group) {
            uint64_t count = memory.value(group, 2);
            std::vector<uint64_t> remaining;
            for (uint64_t i = 0; i < count; ++i)
                remaining.push_back(first + i);
            for (uint64_t i = 0; i < std::min(count, per_group_); ++i) {
                size_t at = size_t(choose(rng, remaining.size()));
                uint64_t slot = remaining[at];
                pool_.push_back({slot, group, memory.at(slot), {}});
                remaining[at] = remaining.back();
                remaining.pop_back();
            }
            first += count;
        }
        require(!pool_.empty(), "Replay candidate pool is empty");
        if (reverse_)
            std::reverse(pool_.begin(), pool_.end());
        if (snapshots_ && observations == 0) {
            auto start = std::chrono::steady_clock::now();
            snapshot(engine.root, s, "score-before.ckpt");
            snapshot_seconds += seconds(start);
        }
        auto scores = score(engine, data, pool_);
        for (size_t i = 0; i < pool_.size(); ++i)
            pool_[i].before = scores[i];
    }
    void after_source(LiveEngine &engine, const LiveCorpus &data, const State &s, const Episode &current) override {
        if (!selected_)
            return;
        auto scores = score(engine, data, pool_);
        if (snapshots_ && observations == 0) {
            auto start = std::chrono::steady_clock::now();
            snapshot(engine.root, s, "score-after.ckpt");
            snapshot_seconds += seconds(start);
        }
        log_ << "{\"source_observation\":" << s.meta[24] + 1 << ",\"global_update\":" << s.meta[7]
             << ",\"source\":{\"document\":" << current.document << ",\"offset\":" << current.offset
             << ",\"length\":" << current.length << "},\"candidates\":[";
        for (size_t i = 0; i < pool_.size(); ++i) {
            const auto &candidate = pool_[i];
            if (i)
                log_ << ',';
            log_ << "{\"slot\":" << candidate.slot << ",\"group\":" << candidate.group + 1
                 << ",\"document\":" << candidate.episode.document << ",\"offset\":" << candidate.episode.offset
                 << ",\"length\":" << candidate.episode.length << ",\"before\":";
            print(log_, candidate.before);
            log_ << ",\"after\":";
            print(log_, scores[i]);
            log_ << '}';
        }
        log_ << "]}\n";
        require(bool(log_), "Replay score journal write failed");
        ++observations;
    }
    void finish() {
        log_.close();
        require(bool(log_), "Replay score journal flush failed");
    }
};
} // namespace replay_priority
