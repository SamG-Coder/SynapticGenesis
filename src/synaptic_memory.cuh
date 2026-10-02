// Importance-based consolidation inspired by Synaptic Intelligence (Zenke et al.,
// 2017). Boundaries are explicit completed documents, not inferred biological age.
// This is a surrogate-gradient estimate of utility, not a measured causal value.
#pragma once

__global__ void synaptic_penalty(float *gradient, float *task_gradient, const float *weight,
                                 const float *reference, const float *importance, int n, float strength) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        task_gradient[i] = gradient[i]; // Before penalty and global clipping.
        gradient[i] += 2 * strength * importance[i] * (weight[i] - reference[i]);
    }
}

__global__ void synaptic_consolidate(float *importance, float *path, float *reference, const float *weight,
                                     int n, float damping) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float displacement = weight[i] - reference[i];
        // Rectify the aggregate trajectory estimate, not every noisy update.
        importance[i] += fmaxf(0.f, path[i]) / (displacement * displacement + damping);
        path[i] = 0;
        reference[i] = weight[i];
    }
}

uint64_t float_word(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}
float word_float(uint64_t word) {
    if (word > std::numeric_limits<uint32_t>::max())
        throw std::runtime_error("Invalid float setting encoding");
    uint32_t bits = uint32_t(word);
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

struct SynapticMemory {
    // These three arrays are durable. task_gradient is update scratch only.
    Buf importance, path, reference, task_gradient;
    float strength = 0, damping = .001f;
    uint64_t boundaries = 0, updates = 0;
    bool active() const {
        return importance.n != 0;
    }
    void initialize(const Buf &weights, float cost, float epsilon) {
        if (!std::isfinite(cost) || cost < 0 || cost > 1000000 || !std::isfinite(epsilon) || epsilon <= 0 ||
            epsilon > 1)
            throw std::runtime_error("Invalid synaptic consolidation settings");
        importance = Buf(weights.n);
        path = Buf(weights.n);
        reference = Buf(weights.n);
        task_gradient = Buf(weights.n);
        importance.zero();
        path.zero();
        ck(cudaMemcpy(reference.p, weights.p, weights.n * 4, cudaMemcpyDeviceToDevice));
        strength = cost;
        damping = epsilon;
        boundaries = updates = 0;
    }
    void prepare(Buf &gradient, const Buf &weights) {
        synaptic_penalty<<<unsigned((weights.n + 255) / 256), 256>>>(
            gradient.p, task_gradient.p, weights.p, reference.p, importance.p, int(weights.n), strength);
        ck(cudaGetLastError());
    }
    void consolidate(const Buf &weights) {
        if (!active())
            return;
        synaptic_consolidate<<<unsigned((weights.n + 255) / 256), 256>>>(importance.p, path.p, reference.p,
                                                                         weights.p, int(weights.n), damping);
        ck(cudaGetLastError());
        ++boundaries;
    }
    std::vector<float> host() const {
        std::vector<float> result;
        for (auto *buffer : {&importance, &path, &reference}) {
            auto values = buffer->host();
            result.insert(result.end(), values.begin(), values.end());
        }
        return result;
    }
    void put(const std::vector<float> &values) {
        size_t n = importance.n;
        if (!active() || values.size() != 3 * n)
            throw std::runtime_error("Synaptic memory shape mismatch");
        for (size_t i = 0; i < values.size(); ++i)
            if (!std::isfinite(values[i]) || (i < n && values[i] < 0))
                throw std::runtime_error("Invalid synaptic memory value");
        ck(cudaMemcpy(importance.p, values.data(), n * 4, cudaMemcpyHostToDevice));
        ck(cudaMemcpy(path.p, values.data() + n, n * 4, cudaMemcpyHostToDevice));
        ck(cudaMemcpy(reference.p, values.data() + 2 * n, n * 4, cudaMemcpyHostToDevice));
    }
};
