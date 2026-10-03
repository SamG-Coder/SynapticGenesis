// Bounded fast associative state driven by learned projections of spike traces.
// The delta update is part of forward execution, including generation. Adam
// learns the projections through this recurrence; it does not own the fast state.
#pragma once
namespace association {
constexpr int width = 32;
constexpr int matrix = width * width;
constexpr int packed = 3 * width + 2;
constexpr float epsilon = 1e-5f;

__device__ float warp_sum(float value) {
    for (int delta = 16; delta; delta >>= 1)
        value += __shfl_down_sync(0xffffffff, value, delta);
    return value;
}

__global__ void prepare(const float *raw, float *features, float *inverse, int N) {
    int n = blockIdx.x, j = threadIdx.x;
    if (n >= N)
        return;
    const float *input = raw + n * packed;
    float q = input[j], k = input[width + j];
    float qs = warp_sum(q * q), ks = warp_sum(k * k);
    float iq = rsqrtf(__shfl_sync(0xffffffff, qs, 0) + epsilon);
    float ik = rsqrtf(__shfl_sync(0xffffffff, ks, 0) + epsilon);
    float *out = features + n * packed;
    out[j] = q * iq;
    out[width + j] = k * ik;
    out[2 * width + j] = tanhf(input[2 * width + j]);
    if (j == 0) {
        inverse[2 * n] = iq;
        inverse[2 * n + 1] = ik;
        out[3 * width] = 1.f / (1.f + expf(-input[3 * width]));
        out[3 * width + 1] = 1.f / (1.f + expf(-input[3 * width + 1]));
    }
}

// M = a*M + b*(v - a*M*k)*k^T; read = M*q. Normalized keys
// and gates in [0,1] make the homogeneous state transition non-expansive.
__global__ void forward(const float *features, const float *initial, float *previous, float *reads,
                        float *state, int T, bool streaming) {
    __shared__ float memory[matrix];
    int b = blockIdx.x, thread = threadIdx.x, column = thread % width;
    for (int i = thread; i < matrix; i += blockDim.x)
        memory[i] = initial[b * matrix + i];
    __syncthreads();
    for (int t = 0; t < T; ++t) {
        int n = b * T + t;
        const float *f = features + n * packed;
        float decay = f[3 * width], write = f[3 * width + 1];
        for (int row = thread / width; row < width; row += blockDim.x / width) {
            int i = row * width + column;
            float old = memory[i];
            previous[size_t(n) * matrix + i] = old;
            float predicted = warp_sum(old * f[width + column]);
            predicted = __shfl_sync(0xffffffff, predicted, 0);
            float error = f[2 * width + row] - decay * predicted;
            float next = decay * old + write * error * f[width + column];
            memory[i] = next;
            float read = warp_sum(next * f[column]);
            if (column == 0)
                reads[n * width + row] = read;
        }
        __syncthreads();
    }
    if (streaming)
        for (int i = thread; i < matrix; i += blockDim.x)
            state[b * matrix + i] = memory[i];
}

// Reverse recurrence over one sequence. Each matrix cell and each projected
// feature has one writer. Incoming fast state is a detached chunk boundary.
__global__ void backward(const float *features, const float *inverse, const float *previous,
                         const float *dread, float *draw, int T) {
    __shared__ float gradient[matrix], old[matrix];
    __shared__ float error[width], dk_read[width], delta_gradient[width];
    __shared__ float dq[width], dk[width], decay_gradient[width], dots[2];
    int b = blockIdx.x, thread = threadIdx.x;
    for (int i = thread; i < matrix; i += blockDim.x)
        gradient[i] = 0;
    __syncthreads();
    for (int t = T - 1; t >= 0; --t) {
        int n = b * T + t;
        const float *f = features + n * packed, *dr = dread + n * width;
        float decay = f[3 * width], write = f[3 * width + 1];
        for (int i = thread; i < matrix; i += blockDim.x) {
            old[i] = previous[size_t(n) * matrix + i];
            gradient[i] += dr[i / width] * f[i % width];
        }
        __syncthreads();
        if (thread < width) {
            float prediction = 0, key_read = 0, contraction = 0;
            for (int column = 0; column < width; ++column) {
                int i = thread * width + column;
                prediction += old[i] * f[width + column];
                key_read += gradient[i] * f[width + column];
                contraction += gradient[i] * old[i];
            }
            error[thread] = f[2 * width + thread] - decay * prediction;
            dk_read[thread] = key_read;
            delta_gradient[thread] = write * key_read;
            decay_gradient[thread] = contraction - write * key_read * prediction;
        }
        __syncthreads();
        if (thread < width) {
            float query_gradient = 0, key_gradient = 0;
            for (int row = 0; row < width; ++row) {
                int i = row * width + thread;
                float current = decay * old[i] + write * error[row] * f[width + thread];
                query_gradient += current * dr[row];
                key_gradient += write * gradient[i] * error[row] - decay * old[i] * delta_gradient[row];
            }
            dq[thread] = query_gradient;
            dk[thread] = key_gradient;
        }
        __syncthreads();
        float *out = draw + n * packed;
        if (thread == 0) {
            float qdot = 0, kdot = 0, da = 0, db = 0;
            for (int j = 0; j < width; ++j) {
                qdot += f[j] * dq[j];
                kdot += f[width + j] * dk[j];
                da += decay_gradient[j];
                db += error[j] * dk_read[j];
            }
            dots[0] = qdot;
            dots[1] = kdot;
            out[3 * width] = da * decay * (1 - decay);
            out[3 * width + 1] = db * write * (1 - write);
        }
        __syncthreads();
        if (thread < width) {
            out[thread] = (dq[thread] - f[thread] * dots[0]) * inverse[2 * n];
            out[width + thread] = (dk[thread] - f[width + thread] * dots[1]) * inverse[2 * n + 1];
            float value = f[2 * width + thread];
            out[2 * width + thread] = delta_gradient[thread] * (1 - value * value);
        }
        for (int i = thread; i < matrix; i += blockDim.x)
            gradient[i] = decay * (gradient[i] - delta_gradient[i / width] * f[width + i % width]);
        __syncthreads();
    }
}

struct Cache {
    Buf raw, features, inverse, reads, previous, state, initial, dread, draw, demission;
    Cache(int B, int T, int H)
        : raw(size_t(B) * T * packed), features(raw.n), inverse(size_t(B) * T * 2),
          reads(size_t(B) * T * width), previous(size_t(B) * T * matrix), state(size_t(B) * matrix),
          initial(state.n), dread(reads.n), draw(raw.n), demission(size_t(B) * T * H) {
        state.zero();
    }
    static uint64_t floats(int B, int T, int H) {
        return uint64_t(B) * T * (3 * packed + 2 + 2 * width + matrix + H) + 2ull * B * matrix;
    }
};
} // namespace association
