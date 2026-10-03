// Shared model execution and allocation ownership. One forward path serves
// live learners, generation views and frozen teachers.
#pragma once
enum class ModelBuffers { learning, frozen_forward };
struct Cache {
    Buf norm, rs, z, u, s, state, initial_state;
    Buf adapt, adapt_state, initial_adapt, emission, gate;
    std::unique_ptr<association::Cache> fast_memory;
    Cache(int N, int C, int H, int B, bool secondary, bool traced, bool gated, bool associative,
          bool learning = true)
        : norm(size_t(N) * C), rs(N), z(size_t(N) * H), u(size_t(N) * H), s(size_t(N) * H),
          state(size_t(B) * H), initial_state(size_t(B) * H), adapt(secondary ? size_t(N) * H : 0),
          adapt_state(secondary ? size_t(B) * H : 0), initial_adapt(secondary ? size_t(B) * H : 0),
          emission(traced ? size_t(N) * H : 0), gate(gated ? size_t(N) * H : 0) {
        state.zero();
        if (secondary)
            adapt_state.zero();
        if (associative)
            fast_memory = std::make_unique<association::Cache>(B, N / B, H, learning);
    }
};
struct Model {
    Config q;
    Layout a;
    int B, T, N;
    const bool learning_buffers;
    cublasHandle_t blas{};
    Buf w, g, m, v, decay, finalnorm, finalrs, logits, dlogits, losses, loss_weights, dx, dy, dnorm, ds, dz,
        dgate, neuron_partials;
    std::vector<Buf> x;
    std::vector<Cache> cache;
    // The object exists before views are created; later initialization/restoration
    // is visible through every shared view without replacing its identity.
    std::shared_ptr<SynapticMemory> synapses = std::make_shared<SynapticMemory>();
    int *input = nullptr;
    int *target = nullptr;
    Model(Config config, int batch, int time, ModelBuffers buffers = ModelBuffers::learning)
        : q(config), a(q), B(batch), T(time), N(batch * time),
          learning_buffers(buffers == ModelBuffers::learning), w(a.n), g(learning_buffers ? a.n : 0),
          m(learning_buffers ? a.n : 0), v(learning_buffers ? a.n : 0), decay(learning_buffers ? a.n : 0),
          finalnorm(size_t(N) * q.c), finalrs(N), logits(size_t(N) * 256),
          dlogits(learning_buffers ? size_t(N) * 256 : 0), losses(learning_buffers ? N : 0),
          loss_weights(learning_buffers ? N : 0), dx(learning_buffers ? size_t(N) * q.c : 0),
          dy(learning_buffers || q.associative() ? size_t(N) * q.c : 0),
          dnorm(learning_buffers ? size_t(N) * q.c : 0), ds(learning_buffers ? size_t(N) * q.h : 0),
          dz(learning_buffers ? size_t(N) * q.h : 0),
          dgate(learning_buffers && q.gated() ? size_t(N) * q.h : 0),
          neuron_partials(learning_buffers && B > 1 ? size_t(B) * q.h * (q.secondary() ? 3 : 1) : 0) {
        if (B < 1 || B > 256 || T < 1 || T > 4096)
            throw std::runtime_error("Invalid batch/context");
        cb(cublasCreate(&blas));
        cb(cublasSetMathMode(blas, CUBLAS_PEDANTIC_MATH));
        cb(cublasSetAtomicsMode(blas, CUBLAS_ATOMICS_NOT_ALLOWED));
        ck(cudaMalloc(&input, size_t(N) * sizeof(int)));
        if (learning_buffers)
            ck(cudaMalloc(&target, size_t(N) * sizeof(int)));
        for (int i = 0; i <= q.l; ++i)
            x.emplace_back(size_t(N) * q.c);
        for (int i = 0; i < q.l; ++i)
            cache.emplace_back(N, q.c, q.h, B, q.secondary(), q.traced(), q.gated(), q.associative(),
                               learning_buffers);
        if (learning_buffers) {
            std::vector<float> mask(a.n, 0);
            for (auto pair : a.decay)
                std::fill_n(mask.data() + pair.first, pair.second, 1.f);
            decay.put(mask);
            m.zero();
            v.zero();
        }
    }
    ~Model() {
        cudaFree(input);
        cudaFree(target);
        cublasDestroy(blas);
    }
    void fast(bool enabled) {
        cb(cublasSetMathMode(blas, enabled ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_PEDANTIC_MATH));
    }
    void require_learning(const char *operation) const {
        if (!learning_buffers)
            throw std::runtime_error(std::string(operation) + " requires learning buffers");
    }
    // Execution views own their scratch space, but use the same parameter,
    // optimizer and recurrent-state allocations. Shared ownership prevents a
    // view from dangling when another execution context is destroyed.
    void share_parameters(Model &owner) {
        require_learning("Parameter sharing");
        owner.require_learning("Parameter sharing");
        if (q.c != owner.q.c || q.h != owner.q.h || q.l != owner.q.l || q.cell != owner.q.cell)
            throw std::runtime_error("Shared parameter dimensions differ");
        w.share(owner.w);
        m.share(owner.m);
        v.share(owner.v);
        synapses = owner.synapses;
    }
    void share_runtime(Model &owner) {
        if (B != owner.B)
            throw std::runtime_error("Shared runtime batch differs");
        share_parameters(owner);
        for (int l = 0; l < q.l; ++l) {
            cache[l].state.share(owner.cache[l].state);
            if (q.secondary())
                cache[l].adapt_state.share(owner.cache[l].adapt_state);
            if (q.associative())
                cache[l].fast_memory->state.share(owner.cache[l].fast_memory->state);
        }
    }
    std::vector<float> membranes() const {
        std::vector<float> result;
        for (const auto &c : cache) {
            auto h = c.state.host();
            result.insert(result.end(), h.begin(), h.end());
            if (q.secondary()) {
                auto adaptation = c.adapt_state.host();
                result.insert(result.end(), adaptation.begin(), adaptation.end());
            }
            if (q.associative()) {
                auto memory = c.fast_memory->state.host();
                result.insert(result.end(), memory.begin(), memory.end());
            }
        }
        return result;
    }
    void membranes(const std::vector<float> &values) {
        size_t width = size_t(B) * q.h;
        size_t stride = q.recurrent_per_layer(B);
        if (values.size() != stride * q.l)
            throw std::runtime_error("Membrane state dimensions differ");
        for (int l = 0; l < q.l; ++l) {
            ck(cudaMemcpy(cache[l].state.p, values.data() + l * stride, width * 4, cudaMemcpyHostToDevice));
            if (q.secondary())
                ck(cudaMemcpy(cache[l].adapt_state.p, values.data() + l * stride + width, width * 4,
                              cudaMemcpyHostToDevice));
            if (q.associative())
                ck(cudaMemcpy(cache[l].fast_memory->state.p, values.data() + l * stride + 2 * width,
                              size_t(B) * association::matrix * 4, cudaMemcpyHostToDevice));
        }
    }
    void linear(float *out, const float *in, size_t weight, size_t bias, int I, int O,
                cudaStream_t stream = nullptr) {
        const float one = 1, zero = 0;
        cb(cublasSgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, O, N, I, &one, w.p + weight, I, in, I, &zero, out, O));
        bias_add<<<(N * O + 255) / 256, 256, 0, stream>>>(out, w.p + bias, N, O);
    }
    void linear_backward(float *din, const float *in, const float *dout, size_t weight, size_t bias, int I,
                         int O) {
        require_learning("Linear backward");
        const float one = 1, zero = 0;
        cb(cublasSgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N, I, N, O, &one, w.p + weight, I, dout, O, &zero, din,
                       I));
        cb(cublasSgemm(blas, CUBLAS_OP_N, CUBLAS_OP_T, I, O, N, &one, in, I, dout, O, &zero, g.p + weight,
                       I));
        bias_grad<<<O, 256>>>(g.p + bias, dout, N, O);
    }
    // Input already resides on device. All operations stay on the supplied
    // stream, allowing capture without changing the recurrence or arithmetic.
    void forward_device(bool streaming, bool additions = false, cudaStream_t stream = nullptr) {
        if (additions && (!streaming || B != 1 || T != 1))
            throw std::runtime_error("Spike additions require streaming B=T=1");
        if (additions && q.traced())
            throw std::runtime_error("Trace cell output is not ternary; use dense or graph decoding");
        cb(cublasSetStream(blas, stream));
        embedding<<<(N * q.c + 255) / 256, 256, 0, stream>>>(x[0].p, w.p + a.emb, input, N, q.c);
        for (int l = 0; l < q.l; ++l) {
            auto p = a.layers[l];
            auto &f = cache[l];
            if (streaming)
                ck(cudaMemcpyAsync(f.initial_state.p, f.state.p, f.state.n * 4, cudaMemcpyDeviceToDevice,
                                   stream));
            else
                ck(cudaMemsetAsync(f.initial_state.p, 0, f.initial_state.n * 4, stream));
            if (q.secondary()) {
                if (streaming)
                    ck(cudaMemcpyAsync(f.initial_adapt.p, f.adapt_state.p, f.adapt_state.n * 4,
                                       cudaMemcpyDeviceToDevice, stream));
                else
                    ck(cudaMemsetAsync(f.initial_adapt.p, 0, f.initial_adapt.n * 4, stream));
            }
            rms_fwd<<<N, 256, 0, stream>>>(f.norm.p, f.rs.p, x[l].p, w.p + p.gain, q.c);
            linear(f.z.p, f.norm.p, p.wi, p.bi, q.c, q.h, stream);
            if (q.gated())
                linear(f.gate.p, f.norm.p, p.gate_w, p.gate_b, q.c, q.h, stream);
            if (q.traced())
                trace_fwd<<<(B * q.h + 255) / 256, 256, 0, stream>>>(
                    f.s.p, f.u.p, f.adapt.p, f.emission.p, f.state.p, f.adapt_state.p, f.initial_state.p,
                    f.initial_adapt.p, f.z.p, w.p + p.leak, w.p + p.adapt_leak, w.p + p.adapt_scale, f.gate.p,
                    B, T, q.h, streaming, q.selective());
            else if (q.adaptive())
                alif_fwd<<<(B * q.h + 255) / 256, 256, 0, stream>>>(
                    f.s.p, f.u.p, f.adapt.p, f.state.p, f.adapt_state.p, f.initial_state.p, f.initial_adapt.p,
                    f.z.p, w.p + p.leak, w.p + p.adapt_leak, w.p + p.adapt_scale, B, T, q.h, streaming);
            else
                lif_fwd<<<(B * q.h + 255) / 256, 256, 0, stream>>>(f.s.p, f.u.p, f.state.p, f.initial_state.p,
                                                                   f.z.p, w.p + p.leak, B, T, q.h, streaming);
            if (streaming && additions)
                spike_add<<<(q.c + 255) / 256, 256, 0, stream>>>(x[l + 1].p, w.p + p.wo, f.s.p, w.p + p.bo,
                                                                 q.c, q.h);
            else
                linear(x[l + 1].p, q.traced() ? f.emission.p : f.s.p, p.wo, p.bo, q.h, q.c, stream);
            plus<<<(N * q.c + 255) / 256, 256, 0, stream>>>(x[l + 1].p, x[l].p, N * q.c);
            if (q.associative()) {
                auto &memory = *f.fast_memory;
                if (streaming)
                    ck(cudaMemcpyAsync(memory.initial.p, memory.state.p, memory.state.n * 4,
                                       cudaMemcpyDeviceToDevice, stream));
                else
                    ck(cudaMemsetAsync(memory.initial.p, 0, memory.initial.n * 4, stream));
                linear(memory.raw.p, f.emission.p, p.association_w, p.association_b, q.h, association::packed,
                       stream);
                association::prepare<<<N, 32, 0, stream>>>(memory.raw.p, memory.features.p, memory.inverse.p,
                                                           N);
                association::forward<><<<B, 256, 0, stream>>>(memory.features.p, memory.initial.p,
                                                              memory.previous.p, memory.reads.p,
                                                              memory.state.p, T, streaming);
                linear(dy.p, memory.reads.p, p.association_out, p.association_bias, association::width, q.c,
                       stream);
                plus<<<(N * q.c + 255) / 256, 256, 0, stream>>>(x[l + 1].p, dy.p, N * q.c);
            }
        }
        rms_fwd<<<N, 256, 0, stream>>>(finalnorm.p, finalrs.p, x[q.l].p, w.p + a.final_gain, q.c);
        linear(logits.p, finalnorm.p, a.head, a.bias, q.c, 256, stream);
        ck(cudaGetLastError());
    }
    float forward(const std::vector<int> &in, const std::vector<int> *targets = nullptr,
                  bool streaming = false, bool additions = false) {
        if (targets)
            require_learning("Target loss");
        if (in.size() != size_t(N) || (targets && targets->size() != size_t(N)))
            throw std::runtime_error("Token shape mismatch");
        for (int t : in)
            if (t < 0 || t > 255)
                throw std::runtime_error("Byte token outside [0,255]");
        if (targets)
            for (int t : *targets)
                if (t < 0 || t > 255)
                    throw std::runtime_error("Target outside [0,255]");
        ck(cudaMemcpy(input, in.data(), N * sizeof(int), cudaMemcpyHostToDevice));
        forward_device(streaming, additions);
        if (!targets)
            return 0;
        ck(cudaMemcpy(target, targets->data(), N * sizeof(int), cudaMemcpyHostToDevice));
        classifier<<<N, 256>>>(dlogits.p, losses.p, logits.p, target, N);
        auto ls = losses.host();
        double sum = std::accumulate(ls.begin(), ls.end(), 0.0);
        float result = float(sum / N);
        if (!std::isfinite(result))
            throw std::runtime_error("Nonfinite loss");
        return result;
    }
    float reweight_targets(const std::vector<float> &weights) {
        require_learning("Target weighting");
        double total = target_weight_sum(weights, N);
        loss_weights.put(weights);
        weight_targets<<<(N * 256 + 255) / 256, 256>>>(dlogits.p, loss_weights.p, N, float(N / total));
        ck(cudaGetLastError());
        auto values = losses.host();
        double sum = 0;
        for (size_t i = 0; i < weights.size(); ++i)
            sum += double(weights[i]) * values[i];
        return float(sum / total);
    }
    void backward(float activity_cost = 0) {
        require_learning("Backward");
        g.zero();
        linear_backward(dnorm.p, finalnorm.p, dlogits.p, a.head, a.bias, q.c, 256);
        rms_bwd<<<N + (q.c + 31) / 32, 256>>>(dx.p, g.p + a.final_gain, dnorm.p, x[q.l].p, w.p + a.final_gain,
                                              finalrs.p, N, q.c, false);
        for (int l = q.l - 1; l >= 0; --l) {
            auto p = a.layers[l];
            auto &f = cache[l];
            float *dl = B == 1 ? g.p + p.leak : neuron_partials.p;
            float *dr = q.secondary() ? (B == 1 ? g.p + p.adapt_leak : neuron_partials.p + B * q.h) : nullptr;
            float *dk =
                q.secondary() ? (B == 1 ? g.p + p.adapt_scale : neuron_partials.p + 2 * B * q.h) : nullptr;
            linear_backward(ds.p, q.traced() ? f.emission.p : f.s.p, dx.p, p.wo, p.bo, q.h, q.c);
            if (q.associative()) {
                auto &memory = *f.fast_memory;
                linear_backward(memory.dread.p, memory.reads.p, dx.p, p.association_out, p.association_bias,
                                association::width, q.c);
                association::backward<><<<B, 256>>>(memory.features.p, memory.inverse.p, memory.previous.p,
                                                    memory.dread.p, memory.draw.p, T);
                linear_backward(memory.demission.p, f.emission.p, memory.draw.p, p.association_w,
                                p.association_b, q.h, association::packed);
                plus<<<(N * q.h + 255) / 256, 256>>>(ds.p, memory.demission.p, N * q.h);
            }
            if (activity_cost > 0 && !q.traced())
                spike_cost_grad<<<(N * q.h + 255) / 256, 256>>>(ds.p, f.s.p, N * q.h,
                                                                activity_cost / (float(N) * q.h * q.l));
            if (q.traced())
                trace_bwd<<<(B * q.h + 255) / 256, 256>>>(
                    dz.p, dl, dr, dk, ds.p, f.u.p, f.s.p, f.adapt.p, f.initial_state.p, f.initial_adapt.p,
                    w.p + p.leak, w.p + p.adapt_leak, w.p + p.adapt_scale, f.gate.p, dgate.p, B, T, q.h,
                    activity_cost / (float(N) * q.h * q.l), q.selective());
            else if (q.adaptive())
                alif_bwd<<<(B * q.h + 255) / 256, 256>>>(dz.p, dl, dr, dk, ds.p, f.u.p, f.s.p, f.adapt.p,
                                                         f.initial_state.p, w.p + p.leak, w.p + p.adapt_leak,
                                                         w.p + p.adapt_scale, B, T, q.h);
            else
                lif_bwd<<<(B * q.h + 255) / 256, 256>>>(dz.p, dl, ds.p, f.u.p, f.s.p, f.initial_state.p,
                                                        w.p + p.leak, B, T, q.h);
            if (B > 1)
                neuron_parameter_grad<<<(q.h + 255) / 256, 256>>>(
                    g.p + p.leak, q.secondary() ? g.p + p.adapt_leak : nullptr,
                    q.secondary() ? g.p + p.adapt_scale : nullptr, neuron_partials.p, B, q.h, q.secondary());
            if (q.gated()) {
                linear_backward(dnorm.p, f.norm.p, dgate.p, p.gate_w, p.gate_b, q.c, q.h);
                rms_bwd<<<N + (q.c + 31) / 32, 256>>>(dx.p, g.p + p.gain, dnorm.p, x[l].p, w.p + p.gain,
                                                      f.rs.p, N, q.c, true);
            }
            linear_backward(dnorm.p, f.norm.p, dz.p, p.wi, p.bi, q.c, q.h);
            rms_bwd<<<N + (q.c + 31) / 32, 256>>>(dx.p, g.p + p.gain, dnorm.p, x[l].p, w.p + p.gain, f.rs.p,
                                                  N, q.c, true);
        }
        embedding_grad<<<256, 256>>>(g.p + a.emb, dx.p, input, N, q.c);
        ck(cudaGetLastError());
    }
    float update(int step, float lr, float wd = 0.01f, float clip = 1.f, float core_scale = 1.f) {
        require_learning("Optimizer update");
        if (!std::isfinite(core_scale) || core_scale < 0 || core_scale > 1)
            throw std::runtime_error("Core learning rate scale must be in [0,1]");
        if (synapses->active())
            synapses->prepare(g, w);
        float norm = 0;
        cb(cublasSnrm2(blas, int(a.n), g.p, 1, &norm));
        if (!std::isfinite(norm))
            throw std::runtime_error("Nonfinite gradient");
        if (clip > 0 && norm > clip) {
            float scale = clip / norm;
            cb(cublasSscal(blas, int(a.n), &scale, g.p, 1));
        }
        adam<<<unsigned((a.n + 255) / 256), 256>>>(
            w.p, m.p, v.p, g.p, decay.p, int(a.n), lr, 1.f - std::pow(.9f, float(step)),
            1.f - std::pow(.95f, float(step)), wd, int(a.final_gain), core_scale,
            synapses->active() ? synapses->path.p : nullptr,
            synapses->active() ? synapses->task_gradient.p : nullptr);
        ck(cudaGetLastError());
        if (synapses->active())
            ++synapses->updates;
        return norm;
    }
    double rate() {
        double sum = 0;
        for (auto &c : cache) {
            float r;
            cb(cublasSasum(blas, int(c.s.n), c.s.p, 1, &r));
            sum += r;
        }
        return sum / (double(N) * q.h * q.l);
    }
    void reset() {
        for (auto &c : cache) {
            c.state.zero();
            if (q.secondary())
                c.adapt_state.zero();
            if (q.associative())
                c.fast_memory->state.zero();
        }
    }
};
