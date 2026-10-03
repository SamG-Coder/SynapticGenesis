// Fixed-order gradient reductions. Each destination has one writer; the
// addition order does not depend on CUDA block scheduling. Same-device/run
// reproducibility still depends on the remaining kernels and cuBLAS version.
#pragma once
constexpr const char *gradient_reduction_name = "ordered_v1";

__global__ void embedding_grad(float *dw, const float *dy, const int *x, int n, int c) {
    int token = blockIdx.x;
    for (int j = threadIdx.x; j < c; j += blockDim.x) {
        float sum = 0;
        for (int row = 0; row < n; ++row)
            if (x[row] == token)
                sum += dy[row * c + j];
        dw[token * c + j] = sum;
    }
}

// A single launch contains disjoint input-gradient and gain-gradient blocks.
// Gain blocks cover 32 neighboring channels and reduce eight row lanes in a
// fixed tree. No scratch allocation or cross-block accumulation is needed.
__global__ void rms_bwd(float *dx, float *dg, const float *dy, const float *x, const float *g, const float *r,
                        int n, int c, bool add) {
    int t = threadIdx.x;
    if (blockIdx.x < n) {
        int row = blockIdx.x;
        float v = 0;
        for (int j = t; j < c; j += 256)
            v += dy[row * c + j] * g[j] * x[row * c + j];
        float dot = reduce_sum(v) / c, rs = r[row];
        for (int j = t; j < c; j += 256) {
            int i = row * c + j;
            float d = dy[i] * g[j] * rs - x[i] * rs * rs * rs * dot;
            dx[i] = (add ? dx[i] : 0) + d;
        }
    } else {
        int j = (blockIdx.x - n) * 32 + t % 32;
        float v = 0;
        if (j < c)
            for (int row = t / 32; row < n; row += 8)
                v += dy[row * c + j] * x[row * c + j] * r[row];
        __shared__ float partial[256];
        partial[t] = v;
        __syncthreads();
        for (int stride = 128; stride >= 32; stride >>= 1) {
            if (t < stride)
                partial[t] += partial[t + stride];
            __syncthreads();
        }
        if (t < 32 && j < c)
            dg[j] += partial[t]; // The input and gate paths run in stream order.
    }
}

// Neuron kernels write one partial per batch item. B=1 writes directly to the
// parameter gradient; batched training combines partials in increasing B order.
__global__ void neuron_parameter_grad(float *dl, float *dr, float *dk, const float *partial, int batch,
                                      int hidden, bool secondary) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= hidden)
        return;
    float leak = 0, retention = 0, scale = 0;
    for (int b = 0; b < batch; ++b) {
        int i = b * hidden + j;
        leak += partial[i];
        if (secondary) {
            retention += partial[batch * hidden + i];
            scale += partial[2 * batch * hidden + i];
        }
    }
    dl[j] = leak;
    if (secondary) {
        dr[j] = retention;
        dk[j] = scale;
    }
}
