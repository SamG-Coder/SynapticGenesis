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
// Launch contract: 256 threads. Each lane owns four matrix cells; only the
// row reductions communicate, through warp shuffles. Resident cells can stay
// in registers until the final state write without cross-warp barriers.
template <bool RegisterState = true>
__global__ void forward(const float *features, const float *initial, float *previous, float *reads,
                        float *state, int T, bool streaming) {
    __shared__ float memory[RegisterState ? 1 : matrix];
    float cells[RegisterState ? 4 : 1];
    int b = blockIdx.x, thread = threadIdx.x, column = thread % width;
    if constexpr (RegisterState) {
#pragma unroll
        for (int part = 0; part < 4; ++part)
            cells[part] = initial[b * matrix + thread + part * 256];
    } else {
        for (int i = thread; i < matrix; i += blockDim.x)
            memory[i] = initial[b * matrix + i];
        __syncthreads();
    }
    for (int t = 0; t < T; ++t) {
        int n = b * T + t;
        const float *f = features + n * packed;
        float decay = f[3 * width], write = f[3 * width + 1];
        // Hoist each lane's key/query before writing the disjoint history buffer.
        float key = f[width + column], query = f[column];
#pragma unroll
        for (int part = 0; part < 4; ++part) {
            int row = thread / width + part * (256 / width);
            int i = row * width + column;
            float old = RegisterState ? cells[part] : memory[i];
            previous[size_t(n) * matrix + i] = old;
            float predicted = warp_sum(old * key);
            predicted = __shfl_sync(0xffffffff, predicted, 0);
            float error = f[2 * width + row] - decay * predicted;
            float next = decay * old + write * error * key;
            if constexpr (RegisterState)
                cells[part] = next;
            else
                memory[i] = next;
            float read = warp_sum(next * query);
            if (column == 0)
                reads[n * width + row] = read;
        }
        if constexpr (!RegisterState)
            __syncthreads();
    }
    if (streaming) {
#pragma unroll
        for (int part = 0; part < 4; ++part) {
            int i = thread + part * 256;
            state[b * matrix + i] = RegisterState ? cells[part] : memory[i];
        }
    }
}

// Reverse recurrence over one sequence. Each matrix cell and each projected
// feature has one writer. Incoming fast state is a detached chunk boundary.
// Keep scalar reduction order while sharing the current projected features
// across all row/column contractions. Uncached/packed variants are controls.
template <int Stride = width + 1, bool CacheInputs = true>
__global__ void backward(const float *features, const float *inverse, const float *previous,
                         const float *dread, float *draw, int T) {
    static_assert(Stride >= width, "Shared stride is smaller than the matrix width");
    __shared__ float gradient[width * Stride], old[width * Stride];
    __shared__ float error[width], dk_read[width], delta_gradient[width];
    __shared__ float dq[width], dk[width], decay_gradient[width], dots[2];
    __shared__ float shared_features[CacheInputs ? packed : 1];
    __shared__ float shared_upstream[CacheInputs ? width : 1], shared_inverse[CacheInputs ? 2 : 1];
    int b = blockIdx.x, thread = threadIdx.x;
    for (int i = thread; i < matrix; i += blockDim.x)
        gradient[(i / width) * Stride + i % width] = 0;
    __syncthreads();
    for (int t = T - 1; t >= 0; --t) {
        int n = b * T + t;
        if constexpr (CacheInputs) {
            if (thread < packed)
                shared_features[thread] = features[n * packed + thread];
            if (thread < width)
                shared_upstream[thread] = dread[n * width + thread];
            if (thread < 2)
                shared_inverse[thread] = inverse[2 * n + thread];
            __syncthreads();
        }
        const float *f = CacheInputs ? shared_features : features + n * packed;
        const float *dr = CacheInputs ? shared_upstream : dread + n * width;
        const float *iv = CacheInputs ? shared_inverse : inverse + 2 * n;
        float decay = f[3 * width], write = f[3 * width + 1];
        for (int i = thread; i < matrix; i += blockDim.x) {
            int at = (i / width) * Stride + i % width;
            old[at] = previous[size_t(n) * matrix + i];
            gradient[at] += dr[i / width] * f[i % width];
        }
        __syncthreads();
        if (thread < width) {
            float prediction = 0, key_read = 0, contraction = 0;
            for (int column = 0; column < width; ++column) {
                int i = thread * Stride + column;
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
                int i = row * Stride + thread;
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
            out[thread] = (dq[thread] - f[thread] * dots[0]) * iv[0];
            out[width + thread] = (dk[thread] - f[width + thread] * dots[1]) * iv[1];
            float value = f[2 * width + thread];
            out[2 * width + thread] = delta_gradient[thread] * (1 - value * value);
        }
        for (int i = thread; i < matrix; i += blockDim.x) {
            int at = (i / width) * Stride + i % width;
            gradient[at] = decay * (gradient[at] - delta_gradient[i / width] * f[width + i % width]);
        }
        __syncthreads();
    }
}

struct Cache {
    Buf raw, features, inverse, reads, previous, state, initial, dread, draw, demission;
    Cache(int B, int T, int H, bool learning = true)
        : raw(size_t(B) * T * packed), features(raw.n), inverse(size_t(B) * T * 2),
          reads(size_t(B) * T * width), previous(size_t(B) * T * matrix), state(size_t(B) * matrix),
          initial(state.n), dread(learning ? reads.n : 0), draw(learning ? raw.n : 0),
          demission(learning ? size_t(B) * T * H : 0) {
        state.zero();
    }
    static uint64_t floats(int B, int T, int H, bool learning = true) {
        return uint64_t(B) * T * (2 * packed + 2 + width + matrix +
                                  (learning ? packed + width + H : 0)) + 2ull * B * matrix;
    }
};
} // namespace association
