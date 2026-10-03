// Experimental replay selection; stage choice and all real RNG draws stay native.
#pragma once
#include "replay_candidate_scoring.cuh"
namespace replay_priority {
bool same_episode(const Episode &a, const Episode &b) {
    return a.document == b.document && a.offset == b.offset && a.length == b.length;
}
class TwoChoice : public LiveSourceObserver, public CandidateScorer {
    uint64_t seed_;
    bool prioritize_, ready_ = false;
    std::ofstream log_;
    std::vector<Candidate> pool_;
    Episode selected_{}, uniform_{};
    uint64_t group_ = 0;

  public:
    uint64_t comparisons = 0, overrides = 0, singleton_pools = 0;
    TwoChoice(const fs::path &out, uint64_t seed, bool prioritize, bool audit)
        : CandidateScorer(audit, true), seed_(seed), prioritize_(prioritize), log_(out / "selection.jsonl") {
        require(bool(log_), "Cannot create replay selection journal");
        log_ << std::setprecision(17);
    }
    void before_source(LiveEngine &engine, const LiveCorpus &data, const State &s, const Episode &current) override {
        ready_ = false;
        if ((s.meta[24] + 1) % s.extra[2] || StageReplayView(s).count() == 0)
            return;
        // Preview on a host-state copy. The live loop still performs the real
        // uniform choice and RNG advancement at its original point in time.
        State preview = s;
        uniform_ = ReplayMemory(preview).choose(current);
        StageReplayView memory(s);
        group_ = memory.group_for(uniform_.document);
        uint64_t start = 0;
        for (size_t g = 0; g < group_; ++g)
            start += memory.value(g, 2);
        uint64_t first = 0;
        std::vector<uint64_t> alternatives;
        for (uint64_t i = start; i < start + memory.value(group_, 2); ++i) {
            auto e = memory.at(i);
            if (same_episode(e, uniform_))
                first = i;
            else if (e.length == uniform_.length)
                alternatives.push_back(i);
        }
        pool_ = {{first, group_, uniform_, {}}};
        if (!alternatives.empty()) {
            uint64_t rng = seed_ ^ ((s.meta[24] + 1) * 0x9e3779b97f4a7c15ull);
            if (!rng)
                rng = 1;
            uint64_t slot = alternatives[size_t(choose(rng, alternatives.size()))];
            pool_.push_back({slot, group_, memory.at(slot), {}});
        } else
            ++singleton_pools;
        auto before = score(engine, data, pool_);
        for (size_t i = 0; i < pool_.size(); ++i)
            pool_[i].before = before[i];
        ready_ = true;
    }
    void after_source(LiveEngine &engine, const LiveCorpus &data, const State &s, const Episode &) override {
        if (!ready_)
            return;
        auto after = score(engine, data, pool_);
        double first_delta = after[0].weighted - pool_[0].before.weighted;
        size_t chosen = 0;
        // Only replace the uniform choice with a more damaged positive-delta
        // alternative. Ties and pools with no positive damage keep uniform.
        if (prioritize_ && pool_.size() == 2 &&
            after[1].weighted - pool_[1].before.weighted > std::max(0.0, first_delta))
            chosen = 1;
        selected_ = pool_[chosen].episode;
        log_ << "{\"observation\":" << s.meta[24] + 1 << ",\"group\":" << group_ + 1
             << ",\"chosen\":" << chosen << ",\"candidates\":[";
        for (size_t i = 0; i < pool_.size(); ++i) {
            if (i)
                log_ << ',';
            auto e = pool_[i].episode;
            log_ << "{\"slot\":" << pool_[i].slot << ",\"document\":" << e.document
                 << ",\"offset\":" << e.offset << ",\"length\":" << e.length
                 << ",\"before\":" << pool_[i].before.weighted << ",\"after\":" << after[i].weighted << '}';
        }
        log_ << "]}\n";
        require(bool(log_), "Replay selection journal write failed");
        ++comparisons;
        overrides += chosen != 0;
    }
    Episode choose_replay(const LiveCorpus &, const State &, const Episode &uniform) override {
        require(ready_ && same_episode(uniform, uniform_), "Uniform replay differs from preview");
        ready_ = false;
        return selected_;
    }
    void finish() {
        log_.close();
        require(bool(log_), "Replay selection journal flush failed");
    }
};
} // namespace replay_priority
