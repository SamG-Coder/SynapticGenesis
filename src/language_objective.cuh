// Byte prediction losses and target weighting. Model forward computation stays
// shared by learning, frozen teachers and generation.
#pragma once

__global__ void classifier(float *grad, float *loss, const float *logits, const int *target, int N) {
    int row = blockIdx.x, j = threadIdx.x;
    __shared__ float sh[256];
    float v = logits[row * 256 + j];
    sh[j] = v;
    __syncthreads();
    for (int d = 128; d; d >>= 1) {
        if (j < d)
            sh[j] = fmaxf(sh[j], sh[j + d]);
        __syncthreads();
    }
    float mx = sh[0];
    __syncthreads();
    float e = expf(v - mx);
    float total = reduce_sum(e);
    grad[row * 256 + j] = (e / total - float(j == target[row])) / N;
    if (j == 0)
        loss[row] = logf(total) + mx - logits[row * 256 + target[row]];
}

__global__ void weight_targets(float *gradient, const float *weights, int N, float normalization) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N * 256)
        gradient[i] *= weights[i / 256] * normalization;
}

double target_weight_sum(const std::vector<float> &weights, int rows) {
    if (weights.size() != size_t(rows))
        throw std::runtime_error("Target weight dimensions differ");
    double total = 0;
    for (float weight : weights) {
        if (!std::isfinite(weight) || weight < 0 || weight > 1000)
            throw std::runtime_error("Invalid target weight");
        total += weight;
    }
    if (total <= 0)
        throw std::runtime_error("Target weights must have positive mass");
    return total;
}
