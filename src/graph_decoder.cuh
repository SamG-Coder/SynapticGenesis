// Captured one-byte inference on the same mutable weights and membrane state.
// Call begin() after another execution view updates those allocations. step()
// completes before returning; training may then resume on the default stream.
#pragma once
struct GraphDecoder {
    Model &model;
    cudaStream_t stream = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    int *input = nullptr;
    float *output = nullptr;
    GraphDecoder(Model &m, bool additions = false) : model(m) {
        if (m.B != 1 || m.T != 1)
            throw std::runtime_error("Graph decoder requires B=T=1");
        try {
            begin();
            auto state = m.membranes();
            ck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
            ck(cudaMallocHost(&input, sizeof(int)));
            ck(cudaMallocHost(&output, 256 * sizeof(float)));
            *input = 0;
            ck(cudaMemcpyAsync(m.input, input, sizeof(int), cudaMemcpyHostToDevice, stream));
            // Warm cuBLAS outside capture. Restore the warm-up's state changes.
            m.forward_device(true, additions, stream);
            ck(cudaStreamSynchronize(stream));
            m.membranes(state);
            ck(cudaDeviceSynchronize());
            ck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
            ck(cudaMemcpyAsync(m.input, input, sizeof(int), cudaMemcpyHostToDevice, stream));
            m.forward_device(true, additions, stream);
            ck(cudaMemcpyAsync(output, m.logits.p, 256 * sizeof(float), cudaMemcpyDeviceToHost, stream));
            ck(cudaStreamEndCapture(stream, &graph));
            ck(cudaGraphInstantiateWithFlags(&executable, graph, 0));
            cb(cublasSetStream(m.blas, nullptr));
        } catch (...) {
            // End invalidated capture before releasing allocations.
            if (stream) {
                cudaStreamCaptureStatus status;
                if (cudaStreamIsCapturing(stream, &status) == cudaSuccess &&
                    status != cudaStreamCaptureStatusNone) {
                    cudaGraph_t abandoned = nullptr;
                    cudaStreamEndCapture(stream, &abandoned);
                    if (abandoned)
                        cudaGraphDestroy(abandoned);
                }
            }
            cublasSetStream(m.blas, nullptr);
            release();
            throw;
        }
    }
    GraphDecoder(const GraphDecoder &) = delete;
    GraphDecoder &operator=(const GraphDecoder &) = delete;
    void release() noexcept {
        if (stream)
            cudaStreamSynchronize(stream);
        if (executable)
            cudaGraphExecDestroy(executable);
        if (graph)
            cudaGraphDestroy(graph);
        if (input)
            cudaFreeHost(input);
        if (output)
            cudaFreeHost(output);
        if (stream)
            cudaStreamDestroy(stream);
    }
    ~GraphDecoder() {
        release();
    }
    void begin() {
        ck(cudaDeviceSynchronize());
    }
    std::vector<float> step(int byte) {
        if (byte < 0 || byte > 255)
            throw std::runtime_error("Graph input outside [0,255]");
        *input = byte;
        ck(cudaGraphLaunch(executable, stream));
        ck(cudaStreamSynchronize(stream));
        return std::vector<float>(output, output + 256);
    }
};
