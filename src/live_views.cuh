// Sequential live execution and scoring reuse one gradient/optimizer workspace.
// Cached shapes own activation scratch and, when requested, independent state.
#pragma once
class LiveViews {
    std::map<int, std::unique_ptr<Model>> views_;
    std::map<int, uint64_t> used_;
    uint64_t clock_ = 0;

  public:
    static constexpr size_t capacity = 4;
    size_t size() const { return views_.size(); }
    auto begin() const { return views_.begin(); }
    auto end() const { return views_.end(); }
    Model &get(Model &root, int length, bool stream, bool fast) {
        auto found = views_.find(length);
        if (found == views_.end()) {
            if (views_.size() == capacity) {
                // No graph references these views. Complete any submitted work
                // before releasing the least recently used scratch allocation.
                ck(cudaDeviceSynchronize());
                auto oldest = std::min_element(used_.begin(), used_.end(),
                    [](const auto &a, const auto &b) { return a.second < b.second; });
                views_.erase(oldest->first);
                used_.erase(oldest);
            }
            auto view = std::make_unique<Model>(root, length,
                stream ? ModelViewState::shared : ModelViewState::independent);
            view->fast(fast);
            found = views_.emplace(length, std::move(view)).first;
        }
        used_[length] = ++clock_;
        return *found->second;
    }
    static void share(Model &view, Model &root, bool stream) {
        if (stream)
            view.share_runtime(root);
        else
            view.share_parameters(root);
        // Backward and Adam complete before another live view runs. Gradients
        // are scratch, and the decay mask is constant for this common layout.
        view.g.share(root.g);
        view.decay.share(root.decay);
    }
};
