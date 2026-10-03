// Isolate shared-layout cost without changing arithmetic or the model's state.
#pragma once
struct AssociationKernelGraph {
    cudaStream_t stream = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    cudaEvent_t start = nullptr, finish = nullptr;
    AssociationKernelGraph(association::Cache &cache, float *output, int B, int T, int iterations,
                           int variant, bool forward = false) {
        ck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        ck(cudaEventCreate(&start));
        ck(cudaEventCreate(&finish));
        ck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        for (int i = 0; i < iterations; ++i) {
            if (forward && variant == 0)
                association::forward<false><<<B, 256, 0, stream>>>(
                    cache.features.p, cache.initial.p, cache.previous.p, cache.reads.p, output, T, true);
            else if (forward)
                association::forward<true><<<B, 256, 0, stream>>>(
                    cache.features.p, cache.initial.p, cache.previous.p, cache.reads.p, output, T, true);
            else if (variant == 0)
                association::backward<32, false><<<B, 256, 0, stream>>>(
                    cache.features.p, cache.inverse.p, cache.previous.p, cache.dread.p, output, T);
            else if (variant == 1)
                association::backward<33, false><<<B, 256, 0, stream>>>(
                    cache.features.p, cache.inverse.p, cache.previous.p, cache.dread.p, output, T);
            else if (variant == 2)
                association::backward<32, true><<<B, 256, 0, stream>>>(
                    cache.features.p, cache.inverse.p, cache.previous.p, cache.dread.p, output, T);
            else
                association::backward<33, true><<<B, 256, 0, stream>>>(
                    cache.features.p, cache.inverse.p, cache.previous.p, cache.dread.p, output, T);
        }
        ck(cudaStreamEndCapture(stream, &graph));
        ck(cudaGraphInstantiateWithFlags(&executable, graph, 0));
    }
    AssociationKernelGraph(const AssociationKernelGraph &) = delete;
    AssociationKernelGraph &operator=(const AssociationKernelGraph &) = delete;
    ~AssociationKernelGraph() {
        if (stream)
            cudaStreamSynchronize(stream);
        if (executable)
            cudaGraphExecDestroy(executable);
        if (graph)
            cudaGraphDestroy(graph);
        if (start)
            cudaEventDestroy(start);
        if (finish)
            cudaEventDestroy(finish);
        if (stream)
            cudaStreamDestroy(stream);
    }
    double measure(int iterations) {
        ck(cudaEventRecord(start, stream));
        ck(cudaGraphLaunch(executable, stream));
        ck(cudaEventRecord(finish, stream));
        ck(cudaEventSynchronize(finish));
        float milliseconds = 0;
        ck(cudaEventElapsedTime(&milliseconds, start, finish));
        return milliseconds * 1000. / iterations;
    }
};

void association_layout_bench(const Args &args) {
    args.allow({"out", "rounds", "iterations"});
    int rounds = args.num("rounds", 7), iterations = args.num("iterations", 100);
    require(rounds >= 3 && rounds <= 31 && iterations >= 1 && iterations <= 1000,
            "Invalid association layout benchmark bounds");
    fs::path out = args.get("out", "reports/association-layout");
    fs::create_directories(out);
    std::ofstream report(out / "benchmark.json");
    report << std::setprecision(12) << "{\"rounds\":" << rounds << ",\"iterations\":" << iterations
           << ",\"timing\":\"CUDA events around captured graphs of recurrence kernels; excludes "
              "projection, normalization, Adam and host transfer.\",\"cases\":[";
    int cases = 0;
    for (int B : {1, 4})
        for (int T : {1, 17, 74, 128, 512}) {
            association::Cache cache(B, T, 8);
            std::vector<Buf> outputs;
            for (int variant = 0; variant < 4; ++variant)
                outputs.emplace_back(cache.draw.n);
            std::vector<float> raw(cache.raw.n), incoming(cache.initial.n), upstream(cache.dread.n);
            for (size_t i = 0; i < raw.size(); ++i)
                raw[i] = .7f * std::sin(float(i % 113) * .19f);
            for (int n = 0; n < B * T; ++n) {
                if (n % 7 == 0)
                    for (int j = 0; j < 2 * association::width; ++j)
                        raw[n * association::packed + j] *= 1e-6f; // Normalization epsilon matters.
                if (n % 5 == 0) {
                    raw[n * association::packed + 96] = 40.f;
                    raw[n * association::packed + 97] = -40.f;
                }
            }
            for (size_t i = 0; i < incoming.size(); ++i)
                incoming[i] = .1f * std::cos(float(i % 73) * .13f);
            for (size_t i = 0; i < upstream.size(); ++i)
                upstream[i] = .03f * std::sin(float(i % 89) * .17f);
            cache.raw.put(raw);
            cache.initial.put(incoming);
            cache.dread.put(upstream);
            association::prepare<<<B * T, 32>>>(cache.raw.p, cache.features.p, cache.inverse.p, B * T);
            ck(cudaDeviceSynchronize());
            std::vector<std::unique_ptr<AssociationKernelGraph>> graphs;
            for (int variant = 0; variant < 6; ++variant) {
                graphs.push_back(std::make_unique<AssociationKernelGraph>(
                    cache, variant < 4 ? outputs[variant].p : cache.state.p, B, T, iterations,
                    variant < 4 ? variant : variant - 4, variant >= 4));
            }
            graphs[4]->measure(iterations);
            auto previous = cache.previous.host(), reads = cache.reads.host(), state = cache.state.host();
            graphs[5]->measure(iterations);
            auto identical = [](const std::vector<float> &a, const std::vector<float> &b) {
                return a.size() == b.size() && !std::memcmp(a.data(), b.data(), a.size() * sizeof(float));
            };
            require(identical(previous, cache.previous.host()) && identical(reads, cache.reads.host()) &&
                        identical(state, cache.state.host()),
                    "Register state changed forward history, reads or final state");
            // Warm each reverse implementation after identical forward inputs.
            for (int variant = 0; variant < 4; ++variant)
                graphs[variant]->measure(iterations);
            auto reference = outputs[0].host();
            require(std::all_of(reference.begin(), reference.end(), [](float x) { return std::isfinite(x); }),
                    "Backward control produced nonfinite values");
            for (int variant = 1; variant < 4; ++variant) {
                auto candidate = outputs[variant].host();
                require(!std::memcmp(reference.data(), candidate.data(), reference.size() * sizeof(float)),
                        "Shared layout changed a gradient");
            }
            std::array<std::vector<double>, 6> times;
            for (int round = 0; round < rounds; ++round)
                for (int pass = 0; pass < 6; ++pass) {
                    int mode = (round + pass) % 6;
                    times[mode].push_back(graphs[mode]->measure(iterations));
                }
            auto median = [](std::vector<double> values) {
                std::sort(values.begin(), values.end());
                return values.size() % 2 ? values[values.size() / 2]
                                         : (values[values.size() / 2 - 1] + values[values.size() / 2]) * .5;
            };
            report << (cases++ ? "," : "") << "{\"batch\":" << B << ",\"context\":" << T
                   << ",\"gradients_bitwise_identical\":true,\"forward_state_identical\":true";
            const char *names[] = {"packed",        "padded",         "cached",
                                   "cached_padded", "forward_shared", "forward_register"};
            std::cout << "Association B=" << B << " T=" << T;
            for (int mode = 0; mode < 6; ++mode) {
                double us = median(times[mode]);
                report << ",\"" << names[mode] << "_us\":" << us << ",\"" << names[mode] << "_rounds_us\":[";
                for (size_t i = 0; i < times[mode].size(); ++i)
                    report << (i ? "," : "") << times[mode][i];
                report << "]";
                std::cout << " " << names[mode] << "=" << us;
            }
            report << "}";
            std::cout << " us, gradients identical\n";
        }
    report << "],\"passed\":true,\"case_count\":" << cases
           << ",\"extra_shared_bytes_per_block\":{\"padded\":256,\"cached\":528,\"cached_padded\":784},"
              "\"extra_global_bytes\":0}\n";
}
