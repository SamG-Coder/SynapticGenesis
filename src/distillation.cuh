// Optional teacher-distribution objective, independent of source/replay policy.
// Call after an observed-target forward, before backward or target weighting.
// Teacher logits refer to the same input byte positions and fixed vocabulary.
#pragma once

namespace distillation {
__device__ float row_sum(float value) {
    float result = reduce_sum(value);
    __syncthreads(); // All warps must read the result before the next reduction.
    return result;
}
__device__ float row_max(float value) {
    __shared__ float values[256];
    int j = threadIdx.x;
    values[j] = value;
    __syncthreads();
    for (int d = 128; d; d >>= 1) {
        if (j < d)
            values[j] = fmaxf(values[j], values[j + d]);
        __syncthreads();
    }
    float result = values[0];
    __syncthreads();
    return result;
}

__global__ void prepare(float *probability, float *invalid, const float *a, const float *b,
                        float temperature, float mixture) {
    int row = blockIdx.x, j = threadIdx.x, at = row * 256 + j;
    float va = a[at], vb = b ? b[at] : va;
    float bad = row_sum(float(!isfinite(va) || !isfinite(vb)));
    if (j == 0)
        invalid[row] = bad;
    if (bad)
        return; // Uniform across the block; failed preparation never becomes ready.
    float ea = expf((va - row_max(va)) / temperature);
    float pa = ea / row_sum(ea);
    float eb = expf((vb - row_max(vb)) / temperature);
    float pb = eb / row_sum(eb);
    probability[at] = b ? mixture * pa + (1 - mixture) * pb : pa;
}

__global__ void penalty(float *gradient, float *loss, const float *logits, const float *probability,
                        int N, float temperature, float strength) {
    int row = blockIdx.x, j = threadIdx.x, at = row * 256 + j;
    float z = logits[at], q = probability[at];
    float logit = (z - row_max(z)) / temperature;
    float exponential = expf(logit);
    float sum = row_sum(exponential);
    float log_probability = logit - logf(sum);
    // q's finite-precision mass need not equal exactly one. Retaining it makes
    // the gradient consistent with the KL objective on the stored probabilities.
    float mass = row_sum(q);
    gradient[at] += strength * temperature * (mass * exponential / sum - q) / N;
    float term = q > 0 ? q * (logf(q) - log_probability) : 0;
    float kl = row_sum(term);
    if (j == 0)
        loss[row] = strength * temperature * temperature * kl;
}

struct Result {
    float observed_loss, teacher_penalty, total_loss;
};

class Targets {
    int rows_;
    float temperature_;
    Buf probability_, losses_;
    bool ready_ = false;
    static int valid_rows(int rows) {
        if (rows < 1 || rows > 256 * 4096)
            throw std::runtime_error("Invalid teacher target dimensions");
        return rows;
    }
    static float valid_temperature(float temperature) {
        if (!std::isfinite(temperature) || temperature < .25f || temperature > 16.f)
            throw std::runtime_error("Teacher temperature must be in [0.25,16]");
        return temperature;
    }

  public:
    Targets(int rows, float temperature)
        : rows_(valid_rows(rows)), temperature_(valid_temperature(temperature)),
          probability_(size_t(rows_) * 256), losses_(rows_) {}
    size_t gpu_bytes() const {
        return (probability_.n + losses_.n) * sizeof(float);
    }
    std::vector<float> probabilities() const {
        if (!ready_)
            throw std::runtime_error("Teacher targets are not prepared");
        return probability_.host();
    }
    void from_logits(const Buf &a, const Buf *b = nullptr, float mixture = .5f) {
        ready_ = false;
        if (a.n != probability_.n || (b && b->n != a.n) || !std::isfinite(mixture) || mixture < 0 ||
            mixture > 1)
            throw std::runtime_error("Invalid teacher logits or mixture");
        prepare<<<rows_, 256>>>(probability_.p, losses_.p, a.p, b ? b->p : nullptr,
                                temperature_, mixture);
        ck(cudaGetLastError());
        for (float value : losses_.host())
            if (value != 0)
                throw std::runtime_error("Nonfinite teacher logits");
        ready_ = true;
    }
    Result apply(Model &student, float strength, const std::vector<float> &target_weights = {}) {
        if (student.N != rows_ || !std::isfinite(strength) || strength < 0 || strength > 100 ||
            (strength > 0 && !ready_))
            throw std::runtime_error("Invalid teacher objective settings or unprepared targets");
        double denominator = target_weights.empty() ? rows_ : target_weight_sum(target_weights, rows_);
        double extra = 0;
        if (strength > 0) {
            penalty<<<rows_, 256>>>(student.dlogits.p, losses_.p, student.logits.p, probability_.p,
                                    rows_, temperature_, strength);
            ck(cudaGetLastError());
            auto values = losses_.host();
            for (int i = 0; i < rows_; ++i) {
                if (!std::isfinite(values[i]))
                    throw std::runtime_error("Nonfinite teacher penalty");
                extra += double(values[i]) * (target_weights.empty() ? 1.f : target_weights[i]);
            }
        }
        float observed;
        if (target_weights.empty()) {
            auto values = student.losses.host();
            observed = float(std::accumulate(values.begin(), values.end(), 0.0) / rows_);
        } else
            observed = student.reweight_targets(target_weights);
        if (!std::isfinite(observed))
            throw std::runtime_error("Nonfinite observed-target loss");
        float teacher = float(extra / denominator);
        if (!std::isfinite(observed + teacher))
            throw std::runtime_error("Nonfinite combined objective");
        return {observed, teacher, observed + teacher};
    }
};
} // namespace distillation
