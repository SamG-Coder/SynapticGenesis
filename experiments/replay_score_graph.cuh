// Captured reset-state scoring through the production forward and classifier.
#pragma once
namespace replay_priority {
class ScoreGraph {
    Model &model_;
    cudaStream_t stream_ = nullptr;
    cudaGraph_t graph_ = nullptr;
    cudaGraphExec_t executable_ = nullptr;
    int *tokens_ = nullptr;
    float *losses_ = nullptr;

    void forward() {
        auto &m = model_;
        ck(cudaMemcpyAsync(m.input, tokens_, m.N * sizeof(int), cudaMemcpyHostToDevice, stream_));
        ck(cudaMemcpyAsync(m.target, tokens_ + m.N, m.N * sizeof(int), cudaMemcpyHostToDevice, stream_));
        m.forward_device(false, false, stream_);
        classifier<<<m.N, 256, 0, stream_>>>(m.dlogits.p, m.losses.p, m.logits.p, m.target, m.N);
        ck(cudaMemcpyAsync(losses_, m.losses.p, m.N * sizeof(float), cudaMemcpyDeviceToHost, stream_));
    }
    void release() noexcept {
        if (stream_)
            cudaStreamSynchronize(stream_);
        if (executable_)
            cudaGraphExecDestroy(executable_);
        if (graph_)
            cudaGraphDestroy(graph_);
        if (tokens_)
            cudaFreeHost(tokens_);
        if (losses_)
            cudaFreeHost(losses_);
        if (stream_)
            cudaStreamDestroy(stream_);
    }

  public:
    explicit ScoreGraph(Model &model) : model_(model) {
        model.require_learning("Captured target scoring");
        try {
            ck(cudaDeviceSynchronize());
            ck(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
            ck(cudaMallocHost(&tokens_, 2 * model.N * sizeof(int)));
            ck(cudaMallocHost(&losses_, model.N * sizeof(float)));
            std::fill_n(tokens_, 2 * model.N, 0);
            // Warm the existing kernels/cuBLAS before capture. This model owns
            // recurrence, and every scored window starts from reset state.
            forward();
            ck(cudaStreamSynchronize(stream_));
            ck(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal));
            forward();
            ck(cudaStreamEndCapture(stream_, &graph_));
            ck(cudaGraphInstantiateWithFlags(&executable_, graph_, 0));
            cb(cublasSetStream(model.blas, nullptr));
        } catch (...) {
            if (stream_) {
                cudaStreamCaptureStatus status;
                if (cudaStreamIsCapturing(stream_, &status) == cudaSuccess && status != cudaStreamCaptureStatusNone) {
                    cudaGraph_t abandoned = nullptr;
                    cudaStreamEndCapture(stream_, &abandoned);
                    if (abandoned)
                        cudaGraphDestroy(abandoned);
                }
            }
            cublasSetStream(model.blas, nullptr);
            release();
            throw;
        }
    }
    ScoreGraph(const ScoreGraph &) = delete;
    ScoreGraph &operator=(const ScoreGraph &) = delete;
    ~ScoreGraph() { release(); }
    std::vector<float> score(const std::vector<int> &input, const std::vector<int> &target) {
        require(input.size() == size_t(model_.N) && target.size() == input.size(), "Graph score shape mismatch");
        for (size_t i = 0; i < input.size(); ++i) {
            require(input[i] >= 0 && input[i] < 256 && target[i] >= 0 && target[i] < 256,
                    "Graph score token outside byte vocabulary");
            tokens_[i] = input[i];
            tokens_[model_.N + i] = target[i];
        }
        // Caller synchronizes the live writer once before a complete scoring
        // pass. Each launch completes before another source update can run.
        ck(cudaGraphLaunch(executable_, stream_));
        ck(cudaStreamSynchronize(stream_));
        return std::vector<float>(losses_, losses_ + model_.N);
    }
};
} // namespace replay_priority
