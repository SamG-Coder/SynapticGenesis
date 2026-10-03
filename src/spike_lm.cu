// SynapticGenesis research prototype. Native C++17 / CUDA training and generation.
// Architecture v1: residual RMS-normalized feed-forward projections around
// recurrent, signed leaky integrate-and-fire neurons. No pretrained parameters.
// Training uses a triangular surrogate and DETACHED spike reset (see README).
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
namespace fs = std::filesystem;
void ck(cudaError_t e) {
    if (e != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(e));
}
void cb(cublasStatus_t e) {
    if (e != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("cuBLAS error " + std::to_string(e));
}
struct Buf {
    float *p = nullptr;
    size_t n = 0;
    std::shared_ptr<float> allocation;
    Buf() = default;
    explicit Buf(size_t count) : n(count) {
        if (n) {
            ck(cudaMalloc(&p, n * 4));
#ifdef SG_BUFFER_ALLOCATION_OBSERVER
            // Optional diagnostic accounting of successful explicit float
            // allocations. Sharing adds no allocation; final release removes it.
            SG_BUFFER_ALLOCATION_OBSERVER(n * 4, true);
            const size_t bytes = n * 4;
            allocation = std::shared_ptr<float>(p, [bytes](float *v) {
                cudaFree(v);
                SG_BUFFER_ALLOCATION_OBSERVER(bytes, false);
            });
#else
            allocation = std::shared_ptr<float>(p, [](float *v) { cudaFree(v); });
#endif
        }
    }
    Buf(const Buf &) = delete;
    Buf &operator=(const Buf &) = delete;
    Buf(Buf &&b) noexcept : p(b.p), n(b.n), allocation(std::move(b.allocation)) {
        b.p = nullptr;
        b.n = 0;
    }
    Buf &operator=(Buf &&b) noexcept {
        if (this != &b) {
            allocation = std::move(b.allocation);
            p = b.p;
            n = b.n;
            b.p = nullptr;
            b.n = 0;
        }
        return *this;
    }
    void share(const Buf &b) {
        allocation = b.allocation;
        p = b.p;
        n = b.n;
    }
    void zero() {
        ck(cudaMemset(p, 0, n * 4));
    }
    std::vector<float> host() const {
        std::vector<float> x(n);
        ck(cudaMemcpy(x.data(), p, n * 4, cudaMemcpyDeviceToHost));
        return x;
    }
    void put(const std::vector<float> &x) {
        if (x.size() != n)
            throw std::runtime_error("Buffer shape mismatch");
        ck(cudaMemcpy(p, x.data(), n * 4, cudaMemcpyHostToDevice));
    }
};
#include "associative_memory.cuh"
#include "synaptic_memory.cuh"
struct Config {
    int c = 256, h = 512, l = 4;
    int cell = 1; // 1: LIF, 2: ALIF, 3: trace, 4: read gate, 5: retention gate, 6: associative.
    bool associative() const {
        return cell == 6;
    }
    bool adaptive() const {
        return cell == 2;
    }
    bool traced() const {
        return cell == 3 || cell == 4 || cell == 5 || associative();
    }
    bool gated() const {
        return cell == 4 || cell == 5 || associative();
    }
    bool selective() const {
        return cell == 5 || associative();
    }
    bool secondary() const {
        return adaptive() || traced();
    }
    size_t recurrent_per_layer(int batch = 1) const {
        return size_t(batch) * (h * (secondary() ? 2 : 1) + (associative() ? association::matrix : 0));
    }
    const char *name() const {
        return associative() ? "signed_associative_trace_lif_v6"
               : selective() ? "signed_selective_trace_lif_v5"
               : gated()
                   ? "signed_gated_trace_lif_v4"
                   : (traced() ? "signed_trace_lif_v3" : (adaptive() ? "signed_alif_v2" : "signed_lif_v1"));
    }
};
struct Layer {
    size_t gain, wi, bi, wo, bo, leak, adapt_leak, adapt_scale, gate_w, gate_b;
    size_t association_w, association_b, association_out, association_bias;
};
struct Layout {
    size_t n = 0, emb, final_gain, head, bias;
    std::vector<Layer> layers;
    std::vector<std::pair<size_t, size_t>> decay;
    size_t add(size_t k, bool wd = false) {
        size_t off = n;
        n += k;
        if (wd)
            decay.push_back({off, k});
        return off;
    }
    explicit Layout(Config q) {
        if (q.c < 8 || q.c > 2048 || q.h < 8 || q.h > 8192 || q.l < 1 || q.l > 32 ||
            (q.cell < 1 || q.cell > 6))
            throw std::runtime_error("Unsupported model dimensions");
        emb = add(256ull * q.c, true);
        for (int i = 0; i < q.l; ++i) {
            Layer a{};
            a.gain = add(q.c);
            a.wi = add(size_t(q.h) * q.c, true);
            a.bi = add(q.h);
            a.wo = add(size_t(q.c) * q.h, true);
            a.bo = add(q.c);
            a.leak = add(q.h);
            if (q.secondary()) {
                a.adapt_leak = add(q.h);
                a.adapt_scale = add(q.h);
            }
            if (q.gated()) {
                a.gate_w = add(size_t(q.h) * q.c, true);
                a.gate_b = add(q.h);
            }
            if (q.associative()) {
                a.association_w = add(size_t(association::packed) * q.h, true);
                a.association_b = add(association::packed);
                a.association_out = add(size_t(q.c) * association::width, true);
                a.association_bias = add(q.c);
            }
            layers.push_back(a);
        }
        final_gain = add(q.c);
        head = add(256ull * q.c, true);
        bias = add(256);
        if (n > size_t(std::numeric_limits<int>::max()))
            throw std::runtime_error("Model too large");
    }
};
uint64_t rnd(uint64_t &s) {
    s ^= s >> 12;
    s ^= s << 25;
    s ^= s >> 27;
    return s * 2685821657736338717ull;
}
double uniform(uint64_t &s) {
    return ((rnd(s) >> 11) + 0.5) / 9007199254740992.0;
}
float normal(uint64_t &s) {
    return float(std::sqrt(-2 * std::log(uniform(s))) * std::cos(6.283185307179586 * uniform(s)));
}
uint64_t hash_bytes(const void *p, size_t n, uint64_t h = 14695981039346656037ull) {
    auto a = static_cast<const unsigned char *>(p);
    for (size_t i = 0; i < n; ++i) {
        h ^= a[i];
        h *= 1099511628211ull;
    }
    return h;
}
std::vector<float> initialize(Config q, const Layout &a, uint64_t seed) {
    if (!seed)
        throw std::runtime_error("Seed must be nonzero");
    std::vector<float> w(a.n, 0);
    uint64_t associative_seed = seed ^ 0x85ebca6b9e3779b9ull;
    if (!associative_seed)
        associative_seed = 1;
    auto fill = [&](size_t at, size_t n, float sd) {
        for (size_t i = 0; i < n; ++i)
            w[at + i] = normal(seed) * sd;
    };
    fill(a.emb, 256ull * q.c, 0.1f);
    for (auto v : a.layers) {
        std::fill_n(w.data() + v.gain, q.c, 1.f);
        fill(v.wi, size_t(q.h) * q.c, 1 / std::sqrt(float(q.c)));
        fill(v.wo, size_t(q.c) * q.h, 0.1f / std::sqrt(float(q.h * q.l)));
        for (int j = 0; j < q.h; ++j) {
            float b = 0.5f + 0.49f * j / std::max(1, q.h - 1);
            w[v.leak + j] = std::log(b / (1 - b));
            if (q.secondary()) {
                // Constants consume no RNG: all common parameters have the
                // same random initialization as the matched LIF model.
                double tau = q.traced() ? 8 * std::pow(128.0, double(j) / std::max(1, q.h - 1))
                                        : 32 * std::pow(64.0, double(j) / std::max(1, q.h - 1));
                double rho = std::exp(-1 / tau);
                w[v.adapt_leak + j] = float(std::log(rho / (1 - rho)));
                w[v.adapt_scale + j] = float(std::log(std::expm1(1.0)));
            }
        }
    }
    std::fill_n(w.data() + a.final_gain, q.c, 1.f);
    fill(a.head, 256ull * q.c, 0.1f / std::sqrt(float(q.c)));
    if (q.associative()) {
        // Separate RNG and zero readout preserve the selective model's initial
        // function and all common parameters at the same seed and dimensions.
        for (const auto &v : a.layers) {
            for (int row = 0; row < 3 * association::width; ++row)
                for (int j = 0; j < q.h; ++j)
                    w[v.association_w + size_t(row) * q.h + j] =
                        normal(associative_seed) / std::sqrt(float(q.h));
            float decay = std::exp(-1.f / 128.f);
            w[v.association_b + 3 * association::width] = std::log(decay / (1 - decay));
            w[v.association_b + 3 * association::width + 1] = -std::log(7.f);
        }
    }
    return w;
}
__device__ float reduce_sum(float v) {
    __shared__ float sh[256];
    int t = threadIdx.x;
    sh[t] = v;
    __syncthreads();
    for (int d = 128; d; d >>= 1) {
        if (t < d)
            sh[t] += sh[t + d];
        __syncthreads();
    }
    return sh[0];
}
__global__ void embedding(float *y, const float *w, const int *x, int n, int c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n * c)
        y[i] = w[x[i / c] * c + i % c];
}
__global__ void bias_add(float *x, const float *b, int n, int c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n * c)
        x[i] += b[i % c];
}
__global__ void bias_grad(float *db, const float *dy, int n, int c) {
    int j = blockIdx.x;
    float v = 0;
    for (int i = threadIdx.x; i < n; i += 256)
        v += dy[i * c + j];
    v = reduce_sum(v);
    if (threadIdx.x == 0)
        db[j] = v;
}
__global__ void plus(float *y, const float *x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        y[i] += x[i];
}
__global__ void spike_cost_grad(float *ds, const float *s, int n, float scale) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        ds[i] += scale * s[i];
}
__global__ void rms_fwd(float *y, float *r, const float *x, const float *g, int c) {
    int row = blockIdx.x, t = threadIdx.x;
    float v = 0;
    for (int j = t; j < c; j += 256)
        v += x[row * c + j] * x[row * c + j];
    float a = rsqrtf(reduce_sum(v) / c + 1e-5f);
    if (t == 0)
        r[row] = a;
    for (int j = t; j < c; j += 256)
        y[row * c + j] = x[row * c + j] * a * g[j];
}
#include "gradient_reductions.cuh"
__device__ float sigmoid(float x) {
    return 1 / (1 + expf(-x));
}
__global__ void lif_fwd(float *s, float *u, float *state, const float *initial, const float *z,
                        const float *leak, int B, int T, int H, bool streaming) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= B * H)
        return;
    int b = k / H, j = k % H;
    float beta = sigmoid(leak[j]);
    float reset = initial[k];
    for (int t = 0; t < T; ++t) {
        int i = (b * T + t) * H + j;
        float v = beta * reset + z[i];
        float sp = float(v >= 1) - float(v <= -1);
        u[i] = v;
        s[i] = sp;
        reset = v - sp;
    }
    if (streaming)
        state[k] = reset;
}
__device__ float surrogate(float u) {
    return .3f * (fmaxf(0.f, 1.f - fabsf(u - 1.f)) + fmaxf(0.f, 1.f - fabsf(u + 1.f)));
}
__global__ void lif_bwd(float *dz, float *dl, const float *ds, const float *u, const float *s,
                        const float *initial, const float *leak, int B, int T, int H) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= B * H)
        return;
    int b = k / H, j = k % H;
    float beta = sigmoid(leak[j]), carry = 0, db = 0;
    for (int t = T - 1; t >= 0; --t) {
        int i = (b * T + t) * H + j;
        float du = ds[i] * surrogate(u[i]) + beta * carry;
        float prev = t ? u[i - H] - s[i - H] : initial[k];
        dz[i] = du;
        db += du * prev;
        carry = du;
    }
    dl[k] = db * beta * (1 - beta);
}
#include "adaptive_neuron.cuh"
#include "trace_neuron.cuh"
#include "language_objective.cuh"
__global__ void adam(float *w, float *m, float *v, const float *g, const float *mask, int n, float lr,
                     float b1c, float b2c, float wd, int core_end, float core_scale, float *path = nullptr,
                     const float *task_gradient = nullptr) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float d = g[i];
        float mi = m[i] = .9f * m[i] + .1f * d;
        float vi = v[i] = .95f * v[i] + .05f * d * d;
        float eta = i < core_end ? lr * core_scale : lr;
        float previous = w[i];
        w[i] -= eta * (mi / b1c / (sqrtf(vi / b2c) + 1e-8f) + wd * mask[i] * w[i]);
        if (path)
            path[i] -= task_gradient[i] * (w[i] - previous);
    }
}
// Exact sparse ternary addition for the LIF output projection, used optionally
// in streaming inference. Training uses dense cuBLAS; no speedup is assumed.
__global__ void spike_add(float *y, const float *w, const float *s, const float *b, int C, int H) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < C) {
        float v = b[j];
        for (int k = 0; k < H; ++k) {
            float sp = s[k];
            if (sp > 0)
                v += w[j * H + k];
            else if (sp < 0)
                v -= w[j * H + k];
        }
        y[j] = v;
    }
}
// W8ASpike adaptive threshold/count rule and signed-magnitude bit-plane roundtrip.
// This is a codec reference, distinct from the trainable LIF recurrence above.
__global__ void adaptive_codec(float *out, const float *x, int C, float k) {
    int row = blockIdx.x, j = threadIdx.x;
    float v = 0;
    for (int c = j; c < C; c += 256)
        v += fabsf(x[row * C + c]);
    float th = fminf(1e4f, fmaxf(1e-5f, reduce_sum(v) / C / k));
    for (int c = j; c < C; c += 256) {
        int q = __float2int_rn(x[row * C + c] / th);
        unsigned mag = q < 0 ? unsigned(-int64_t(q)) : unsigned(q), decoded = 0;
        for (int bit = 0; bit < 32; ++bit)
            if ((mag >> bit) & 1u)
                decoded |= 1u << bit;
        out[row * C + c] = (q < 0 ? -float(decoded) : float(decoded)) * th;
    }
}
#include "membrane_policy.cuh"
#include "model.cuh"
#include "model_memory.cuh"
#include "distillation.cuh"

#include "checkpoint.cuh"
struct Data {
    std::vector<unsigned char> bytes;
    std::vector<size_t> start, ends;
    size_t windows = 0;
    uint64_t hash = 0;
    int T;
    Data(const fs::path &path, int t) : T(t) {
        std::ifstream f(path, std::ios::binary);
        if (!f)
            throw std::runtime_error("Cannot open corpus: " + path.string());
        bytes.assign(std::istreambuf_iterator<char>(f), {});
        hash = hash_bytes(bytes.data(), bytes.size());
        size_t begin = 0;
        for (size_t i = 0; i <= bytes.size(); ++i)
            if (i == bytes.size() || bytes[i] == 30) {
                if (i - begin > size_t(T)) {
                    start.push_back(begin);
                    windows += i - begin - T;
                    ends.push_back(windows);
                }
                begin = i + 1;
            }
        if (!windows)
            throw std::runtime_error("Corpus has no document longer than context");
    }
    void batch(int B, uint64_t &r, std::vector<int> &x, std::vector<int> &y) const {
        x.resize(size_t(B) * T);
        y.resize(x.size());
        for (int b = 0; b < B; ++b) {
            size_t k = rnd(r) % windows, j = std::upper_bound(ends.begin(), ends.end(), k) - ends.begin();
            size_t at = start[j] + k - (j ? ends[j - 1] : 0);
            for (int t = 0; t < T; ++t) {
                x[b * T + t] = bytes[at + t];
                y[b * T + t] = bytes[at + t + 1];
            }
        }
    }
};
struct Args {
    std::map<std::string, std::string> values;
    Args(int argc, char **argv) {
        for (int i = 2; i < argc; ++i) {
            std::string k = argv[i];
            if (k.rfind("--", 0))
                throw std::runtime_error("Expected --option");
            if (values.count(k))
                throw std::runtime_error("Duplicate option " + k);
            if (k == "--allow-new-corpus" || k == "--spike-add" || k == "--fast" || k == "--graph")
                values[k] = "1";
            else {
                if (++i >= argc)
                    throw std::runtime_error("Missing value for " + k);
                values[k] = argv[i];
            }
        }
    }
    std::string get(std::string k, std::string d = "") const {
        auto i = values.find("--" + k);
        return i == values.end() ? d : i->second;
    }
    int num(std::string k, int d) const {
        auto v = get(k);
        if (v.empty())
            return d;
        size_t n;
        int r = std::stoi(v, &n);
        if (n != v.size())
            throw std::runtime_error("Invalid integer " + k);
        return r;
    }
    float real(std::string k, float d) const {
        auto v = get(k);
        if (v.empty())
            return d;
        size_t n;
        float r = std::stof(v, &n);
        if (n != v.size() || !std::isfinite(r))
            throw std::runtime_error("Invalid float " + k);
        return r;
    }
    void allow(std::initializer_list<const char *> keys) const {
        for (auto &v : values) {
            bool found = false;
            for (auto k : keys)
                if (v.first == std::string("--") + k)
                    found = true;
            if (!found)
                throw std::runtime_error("Unknown option " + v.first);
        }
    }
};
int cell_version(const Args &args) {
    auto cell = args.get("cell", "lif");
    if (cell == "lif")
        return 1;
    if (cell == "alif")
        return 2;
    if (cell == "trace")
        return 3;
    if (cell == "gated")
        return 4;
    if (cell == "selective")
        return 5;
    if (cell == "associative")
        return 6;
    throw std::runtime_error("--cell must be lif, alif, trace, gated, selective or associative");
}
Config checkpoint_config(const State &s) {
    return {int(s.meta[2]), int(s.meta[3]), int(s.meta[4]), int(s.meta[1])};
}
volatile std::sig_atomic_t interrupted = 0;
void interrupt_handler(int) {
    interrupted = 1;
}
float evaluate(Model &m, const Data &d, int batches) {
    uint64_t r = 712367;
    std::vector<int> x, y;
    double loss = 0;
    for (int i = 0; i < batches; ++i) {
        d.batch(m.B, r, x, y);
        loss += m.forward(x, &y);
    }
    return float(loss / batches);
}
void train(const Args &args) {
    args.allow({"data",    "validation",       "out",        "steps",         "batch",
                "context", "layers",           "channels",   "hidden",        "seed",
                "lr",      "warmup",           "eval-every", "eval-batches",  "save-every",
                "resume",  "allow-new-corpus", "fast",       "activity-cost", "cell",
                "burn-in", "burn-policy"});
    fs::path out = args.get("out", "runs/pilot");
    bool resumed = !args.get("resume").empty();
    State s;
    Config q;
    int B = args.num("batch", 16), T = args.num("context", 128);
    if (resumed) {
        s = header(args.get("resume"));
        if (s.meta[17])
            throw std::runtime_error("Use live --resume to preserve this checkpoint's continuous state");
        q = checkpoint_config(s);
        B = int(s.meta[5]);
        T = int(s.meta[6]);
        for (auto k :
             {"batch", "context", "layers", "channels", "hidden", "seed", "lr", "warmup", "fast", "cell"})
            if (!args.get(k).empty())
                throw std::runtime_error(std::string("Resume preserves ") + k +
                                         "; start a new run to change it");
    } else {
        q = {args.num("channels", 256), args.num("hidden", 512), args.num("layers", 4), cell_version(args)};
        s.meta[5] = B;
        s.meta[6] = T;
        s.meta[10] = s.meta[11] = args.num("seed", 1337);
        s.meta[9] = args.num("warmup", 100);
        s.hp[0] = args.real("lr", .001f);
        s.meta[16] = !args.get("fast").empty();
    }
    int steps = args.num("steps", resumed ? int(s.meta[8]) : 2000), every = args.num("eval-every", 200),
        vb = args.num("eval-batches", 16), save_every = args.num("save-every", 500);
    // These metadata slots are batch-training policy only when meta[17] == 0.
    // Live initialization replaces them with its own cursor and stream policy.
    int burn_in = args.num("burn-in", int(s.meta[19]));
    std::string burn_policy = args.get("burn-policy", s.meta[20] ? "reset" : "warm");
    if (burn_in < 0 || burn_in > 2048 || (burn_policy != "warm" && burn_policy != "reset") ||
        (burn_policy == "reset" && !burn_in))
        throw std::runtime_error("Invalid --burn-in or --burn-policy");
    if (B < 1 || B > 256 || T < 1 || T > 4096 || steps <= int(s.meta[7]) || every < 1 || vb < 1 ||
        save_every < 1 || s.hp[0] <= 0 || s.meta[10] == 0 || s.meta[9] > uint64_t(steps))
        throw std::runtime_error("Invalid training configuration");
    s.meta[8] = steps;
    Data data(args.get("data", "data/prepared/foundations-v1/train.dat"), T + burn_in),
        val(args.get("validation", "data/prepared/foundations-v1/validation.dat"), T);
    if (data.hash == val.hash)
        throw std::runtime_error("Training and validation corpora are identical");
    Model model(q, B, T);
    if (resumed) {
        int target_steps = steps;
        load(args.get("resume"), model, s);
        s.meta[8] = target_steps;
        if (s.meta[12] != data.hash || s.meta[13] != val.hash) {
            if (args.get("allow-new-corpus").empty())
                throw std::runtime_error(
                    "Corpus changed; use a new output directory and --allow-new-corpus explicitly");
            s.hp[3] = std::numeric_limits<float>::max();
            std::cout << "New corpus version; reset best-validation comparison.\n";
        }
    } else
        model.w.put(initialize(q, model.a, s.meta[11]));
    s.hp[4] = args.real("activity-cost", s.hp[4]);
    if (s.hp[4] < 0 || s.hp[4] > 100)
        throw std::runtime_error("Activity cost must be in [0,100]");
    model.fast(s.meta[16] != 0);
    s.meta[19] = burn_in;
    s.meta[20] = burn_policy == "reset";
    std::unique_ptr<Model> history;
    if (burn_in) {
        history = std::make_unique<Model>(model, burn_in, ModelViewState::shared);
        history->fast(s.meta[16] != 0);
    }
    s.meta[12] = data.hash;
    s.meta[13] = val.hash;
    if (!resumed && fs::exists(out / "initial.ckpt"))
        throw std::runtime_error("Run already exists; use --resume or a new output directory");
    fs::create_directories(out);
    if (fs::exists(out / "STOP"))
        throw std::runtime_error("Remove the run's STOP file before continuing");
    if (!resumed)
        save(out / "initial.ckpt", model, s);
    std::ofstream metrics(out / "metrics.jsonl", std::ios::app);
    if (!metrics)
        throw std::runtime_error("Cannot write metrics");
    std::cout << "architecture=" << q.name() << " parameters=" << model.a.n
              << " initialization=" << (resumed ? "own_checkpoint" : "random")
              << " byte_vocabulary=256 train_bytes=" << data.bytes.size() << " val_bytes=" << val.bytes.size()
              << " B=" << B << " T=" << T << " burn_in=" << burn_in << " burn_policy=" << burn_policy << "\n";
    float initial = evaluate(model, val, vb);
    std::cout << "step=" << s.meta[7] << " validation_loss=" << initial
              << " bits_per_byte=" << initial / std::log(2.f) << "\n";
    metrics << "{\"event\":\"start\",\"step\":" << s.meta[7] << ",\"validation_loss\":" << initial
            << ",\"gradient_reductions\":\"" << gradient_reduction_name << "\""
            << ",\"parameters\":" << model.a.n << ",\"train_hash\":\"" << data.hash << "\",\"val_hash\":\""
            << val.hash << "\",\"activity_cost\":" << s.hp[4] << ",\"burn_in\":" << burn_in
            << ",\"burn_policy\":\"" << burn_policy << "\""
            << ",\"from_random_initialization\":" << (resumed ? "false" : "true")
            << ",\"resumed\":" << (resumed ? "true" : "false") << "}\n";
    metrics.flush();
    if (initial < s.hp[3]) {
        s.hp[3] = initial;
        save(out / "best.ckpt", model, s);
    }
    std::signal(SIGINT, interrupt_handler);
    std::signal(SIGTERM, interrupt_handler);
    std::vector<int> x, y, full_x, full_y, prefix_x;
    double running = 0;
    int count = 0;
    auto begin = std::chrono::steady_clock::now();
    uint64_t startstep = s.meta[7];
    for (int step = int(s.meta[7]) + 1; step <= steps; ++step) {
        if (history) {
            data.batch(B, s.meta[10], full_x, full_y);
            prefix_x.clear();
            x.clear();
            y.clear();
            for (int b = 0; b < B; ++b) {
                auto at = full_x.begin() + b * (burn_in + T);
                prefix_x.insert(prefix_x.end(), at, at + burn_in);
                x.insert(x.end(), at + burn_in, at + burn_in + T);
                auto target_at = full_y.begin() + b * (burn_in + T) + burn_in;
                y.insert(y.end(), target_at, target_at + T);
            }
            history->reset();
            history->forward(prefix_x, nullptr, true);
            if (burn_policy == "reset")
                model.reset();
        } else
            data.batch(B, s.meta[10], x, y);
        float loss = model.forward(x, &y, history != nullptr);
        model.backward(s.hp[4]);
        float warm = s.meta[9] ? std::min(1.f, float(step) / s.meta[9]) : 1.f;
        float progress = std::max(0.f, float(step - int(s.meta[9])) / std::max(1, steps - int(s.meta[9])));
        float lr = s.hp[0] * warm * (.1f + .9f * .5f * (1 + std::cos(3.14159265359f * progress)));
        float norm = model.update(step, lr, s.hp[1], s.hp[2]);
        s.meta[7] = step;
        running += loss;
        ++count;
        bool stopping = interrupted || fs::exists(out / "STOP");
        bool eval = step % every == 0 || step == steps || stopping;
        if (step % 50 == 0 || eval) {
            ck(cudaDeviceSynchronize());
            double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
            double rate = model.rate();
            float vl = eval ? evaluate(model, val, vb) : -1;
            double avg = running / count;
            std::cout << "step=" << step << " train_loss=" << avg;
            if (eval)
                std::cout << " val_loss=" << vl;
            std::cout << " spike_rate=" << rate << " gradient_norm=" << norm
                      << " bytes_per_sec=" << (step - startstep) * B * T / seconds << "\n";
            metrics << std::setprecision(9) << "{\"event\":\"train\",\"step\":" << step
                    << ",\"train_loss\":" << avg
                    << ",\"validation_loss\":" << (eval ? std::to_string(vl) : "null")
                    << ",\"spike_rate\":" << rate << ",\"gradient_norm\":" << norm
                    << ",\"learning_rate\":" << lr << ",\"elapsed_seconds\":" << seconds << "}\n";
            metrics.flush();
            running = 0;
            count = 0;
            if (eval && vl < s.hp[3]) {
                s.hp[3] = vl;
                save(out / "best.ckpt", model, s);
            }
        }
        if (step % save_every == 0 || step == steps || stopping)
            save(out / "latest.ckpt", model, s);
        if (stopping) {
            std::cout << "Stopped after checkpoint at step " << step << "\n";
            break;
        }
    }
}
int draw_byte(const std::vector<float> &logits, int top, float temp, uint64_t &rng) {
    std::vector<int> ids(256);
    std::iota(ids.begin(), ids.end(), 0);
    std::partial_sort(ids.begin(), ids.begin() + top, ids.end(),
                      [&](int a, int b) { return logits[a] == logits[b] ? a < b : logits[a] > logits[b]; });
    double total = 0;
    std::vector<double> p(top);
    for (int j = 0; j < top; ++j)
        total += p[j] = std::exp((logits[ids[j]] - logits[ids[0]]) / temp);
    double coin = uniform(rng) * total;
    for (int j = 0; j < top; ++j) {
        coin -= p[j];
        if (coin <= 0)
            return ids[j];
    }
    return ids[top - 1];
}
#include "graph_decoder.cuh"
void sample(const Args &args) {
    args.allow(
        {"checkpoint", "prompt", "tokens", "temperature", "top-k", "seed", "output", "spike-add", "graph"});
    auto path = args.get("checkpoint");
    if (path.empty())
        throw std::runtime_error("Need --checkpoint");
    State s = header(path);
    Config q = checkpoint_config(s);
    Model model(q, 1, 1);
    load(path, model, s);
    std::string prompt = args.get("prompt", "The bird ");
    if (prompt.empty())
        throw std::runtime_error("Prompt is empty");
    int count = args.num("tokens", 400), top = args.num("top-k", 40);
    float temp = args.real("temperature", .8f);
    uint64_t rng = args.num("seed", 42);
    if (count < 0 || count > 100000 || top < 1 || top > 256 || temp <= 0 || !rng)
        throw std::runtime_error("Invalid sampling parameters");
    bool additions = !args.get("spike-add").empty();
    std::unique_ptr<GraphDecoder> graph;
    if (!args.get("graph").empty())
        graph = std::make_unique<GraphDecoder>(model, additions);
    auto step = [&](int byte) {
        if (graph)
            return graph->step(byte);
        model.forward({byte}, nullptr, true, additions);
        return model.logits.host();
    };
    std::vector<float> logits;
    for (unsigned char c : prompt)
        logits = step(int(c));
    std::string text = prompt;
    std::cout << prompt;
    for (int t = 0; t < count; ++t) {
        int chosen = draw_byte(logits, top, temp, rng);
        text.push_back(char(chosen));
        if (chosen == 10 || chosen == 9 || (chosen >= 32 && chosen <= 126))
            std::cout << char(chosen);
        else
            std::cout << "<" << chosen << ">";
        logits = step(chosen);
    }
    std::cout << "\n";
    if (!args.get("output").empty()) {
        std::ofstream f(args.get("output"), std::ios::binary);
        write_raw(f, text.data(), text.size());
    }
}
void require(bool x, const char *why) {
    if (!x)
        throw std::runtime_error(why);
}
float maxdiff(const std::vector<float> &a, const std::vector<float> &b) {
    require(a.size() == b.size(), "Comparison dimensions");
    float e = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        require(std::isfinite(a[i]) && std::isfinite(b[i]), "Nonfinite comparison");
        e = std::max(e, std::abs(a[i] - b[i]));
    }
    return e;
}
template <class T> void dump(const fs::path &p, const std::vector<T> &x) {
    std::ofstream f(p, std::ios::binary);
    write_raw(f, x.data(), x.size());
}
void self_test(const Args &args) {
    args.allow({"out"});
    fs::path out = args.get("out", "reports/native-tests");
    fs::create_directories(out);
    Config q{32, 64, 2};
    Model m(q, 2, 8);
    auto weights = initialize(q, m.a, 123);
    m.w.put(weights);
    std::vector<int> x(16), y(16);
    for (int i = 0; i < 16; ++i) {
        x[i] = (i * 37 + 91) % 256;
        y[i] = (i * 19 + 13) % 256;
    }
    float loss = m.forward(x, &y);
    auto logits = m.logits.host();
    m.backward();
    auto gradients = m.g.host();
    dump(out / "weights.f32", weights);
    dump(out / "gradients.f32", gradients);
    dump(out / "logits.f32", logits);
    dump(out / "inputs.i32", x);
    dump(out / "targets.i32", y);
    m.backward(1.f);
    dump(out / "gradients_regularized.f32", m.g.host());
    std::ofstream conf(out / "fixture.json");
    conf << "{\"channels\":32,\"hidden\":64,\"layers\":2,\"batch\":2,\"context\":8,\"loss\":"
         << std::setprecision(10) << loss << ",\"parameters\":" << m.a.n << "}";
    conf.close();
    auto changed = x;
    changed[7] ^= 127;
    changed[15] ^= 127;
    m.forward(changed);
    auto causal = m.logits.host();
    for (int b = 0; b < 2; ++b)
        for (int t = 0; t < 7; ++t)
            for (int j = 0; j < 256; ++j)
                require(std::abs(logits[(b * 8 + t) * 256 + j] - causal[(b * 8 + t) * 256 + j]) < 2e-6,
                        "Causality failed");
    Model stream(q, 1, 1);
    stream.w.put(weights);
    Model add(q, 1, 1);
    add.w.put(weights);
    float stream_error = 0, addition_error = 0;
    for (int t = 0; t < 8; ++t) {
        stream.forward({x[t]}, nullptr, true);
        add.forward({x[t]}, nullptr, true, true);
        auto a = stream.logits.host(), b = add.logits.host();
        addition_error = std::max(addition_error, maxdiff(a, b));
        for (int j = 0; j < 256; ++j)
            stream_error = std::max(stream_error, std::abs(a[j] - logits[t * 256 + j]));
    }
    require(stream_error < 2e-5, "Streaming does not match batched execution");
    require(addition_error < 2e-5, "Sparse spike addition does not match dense multiplication");
    m.forward(x, &y);
    m.backward();
    m.update(1, .001f, 0, 0);
    dump(out / "updated.f32", m.w.host());
    State state;
    state.meta[5] = 2;
    state.meta[6] = 8;
    state.meta[7] = 1;
    save(out / "resume.ckpt", m, state);
    Model resumed(q, 2, 8);
    State rs;
    load(out / "resume.ckpt", resumed, rs);
    require(maxdiff(m.w.host(), resumed.w.host()) == 0, "Checkpoint weights differ");
    require(maxdiff(m.m.host(), resumed.m.host()) == 0, "Optimizer restore differs");
    m.forward(x, &y);
    m.backward();
    m.update(2, .001f, 0, 0);
    resumed.forward(x, &y);
    resumed.backward();
    resumed.update(2, .001f, 0, 0);
    float resume_error = maxdiff(m.w.host(), resumed.w.host());
    require(resume_error < 2e-6, "Resume update differs");
    fs::copy_file(out / "resume.ckpt", out / "corrupt.ckpt", fs::copy_options::overwrite_existing);
    {
        std::fstream f(out / "corrupt.ckpt", std::ios::in | std::ios::out | std::ios::binary);
        f.seekg(320);
        char c;
        f.read(&c, 1);
        c ^= 1;
        f.seekp(320);
        f.write(&c, 1);
    }
    bool rejected = false;
    try {
        load(out / "corrupt.ckpt", resumed, rs);
    } catch (const std::exception &) {
        rejected = true;
    }
    require(rejected, "Corrupt checkpoint accepted");
    Buf input(4), output(4);
    input.put({-.5f, .5f, -2.5f, 2.5f});
    adaptive_codec<<<1, 256>>>(output.p, input.p, 4, 1.5f);
    require(maxdiff(output.host(), {0, 0, -2, 2}) == 0, "Adaptive codec round-to-even failed");
    input.put({0, 0, 0, 0});
    adaptive_codec<<<1, 256>>>(output.p, input.p, 4, 3);
    require(maxdiff(output.host(), {0, 0, 0, 0}) == 0, "Zero codec failed");
    Model toy(q, 2, 8);
    toy.w.put(weights);
    std::vector<int> tx(16), ty(16);
    for (int i = 0; i < 16; ++i) {
        tx[i] = 'a' + i % 3;
        ty[i] = 'a' + (i + 1) % 3;
    }
    float first = toy.forward(tx, &ty), last = first;
    for (int i = 1; i <= 100; ++i) {
        toy.forward(tx, &ty);
        toy.backward();
        toy.update(i, .003f, 0);
        last = toy.forward(tx, &ty);
    }
    require(last < first * .35f, "Synthetic learning test did not converge");
    std::ofstream report(out / "native.json");
    report << "{\"passed\":true,\"streaming_max_error\":" << stream_error
           << ",\"spike_add_max_error\":" << addition_error << ",\"resume_max_error\":" << resume_error
           << ",\"toy_initial_loss\":" << first << ",\"toy_final_loss\":" << last << "}";
    std::cout << "PASS native tests: causality, streaming, ternary additions, checkpoint/optimizer resume, "
                 "corruption rejection, adaptive codec, learning. Toy loss "
              << first << " -> " << last << ". CPU-autograd fixture: " << out.string() << "\n";
}
#include "adaptive_tests.cuh"
#include "associative_bench.cuh"
#include "associative_tests.cuh"
#include "context_bench.cuh"
#include "curriculum_tests.cuh"
#include "decode_tests.cuh"
#include "evolution.cuh"
#include "evolution_tests.cuh"
#include "feedback_tests.cuh"
#include "indexed_storage.cuh"
#include "language_probes.cuh"
#include "live.cuh"
#include "memory_bench.cuh"
#include "population_live.cuh"
#include "reduction_tests.cuh"
#include "retention_bench.cuh"
#include "stage_replay_tests.cuh"
#include "synaptic_tests.cuh"
#include "test_fixtures.cuh"
#include "distillation_tests.cuh"
#include "teacher_replay_tests.cuh"
int main(int argc, char **argv) {
    try {
        std::cout.setf(std::ios::unitbuf);
        if (argc < 2) {
            std::cout
                << "synapticgenesis train --data train.dat --validation validation.dat --out runs/pilot "
                   "[--steps "
                   "2000] [--cell lif|alif|trace|gated|selective|associative] [--burn-in 512 --burn-policy "
                   "warm|reset]\n"
                << "synapticgenesis sample --checkpoint runs/pilot/best.ckpt --prompt \"The bird \" "
                   "[--graph] "
                   "[--spike-add]\n"
                << "synapticgenesis evaluate --checkpoint runs/pilot/best.ckpt --data test.dat\n"
                << "synapticgenesis live --data train.dat --out runs/live [--checkpoint "
                   "runs/pilot/best.ckpt] "
                   "[--updates 1000] [--consolidation si --si-strength 0.1]\n"
                << "synapticgenesis live --data train.dat --out runs/live --resume runs/live/latest.ckpt "
                   "--updates "
                   "2000\n"
                << "synapticgenesis live --curriculum data/prepared/foundations-live-v1/curriculum.sg "
                   "--out runs/development --replay reservoir --graph\n"
                << "synapticgenesis storage-bench --checkpoint runs/pilot/best.ckpt --data test.dat --out "
                   "reports/storage\n"
                << "synapticgenesis decode-bench --checkpoint runs/pilot/best.ckpt --out reports/decode\n"
                << "synapticgenesis context-bench --checkpoint runs/pilot/best.ckpt --data validation.dat "
                   "--prefix 512 --output reports/context.json\n"
                << "synapticgenesis language-probes --checkpoint runs/pilot/best.ckpt --probes "
                   "data/prepared/relations-v1/development.sgprobe --output reports/probes.json\n"
                << "synapticgenesis retention-bench --train-a reading.dat --train-b science.dat "
                   "--validation-a reading-heldout.dat --validation-b science-heldout.dat --out "
                   "runs/retention\n"
                << "synapticgenesis memory-bench --cell alif --delay 128 --steps 2000 --out "
                   "runs/memory-test\n"
                << "synapticgenesis adaptive-test --out reports/adaptive-tests\n"
                << "synapticgenesis trace-test --out reports/trace-tests\n"
                << "synapticgenesis gated-test --out reports/gated-tests\n"
                << "synapticgenesis selective-test --out reports/selective-tests\n"
                << "synapticgenesis associative-test --out reports/associative-tests\n"
                << "synapticgenesis association-layout-bench --out reports/association-layout\n"
                << "synapticgenesis synaptic-test --out reports/synaptic-tests\n"
                << "synapticgenesis self-test --out reports/native-tests\n"
                << "synapticgenesis population-add --population runs/population --id founder-a "
                   "--checkpoint runs/founder/best.ckpt --data validation.dat\n"
                << "synapticgenesis evolve --population runs/population --data validation.dat "
                   "--round generation-1 --children 2 --seed 1337\n"
                << "synapticgenesis population-live --population runs/population --id generation-1-child-0 "
                   "--curriculum curriculum.sg --validation validation.dat\n"
                << "synapticgenesis teacher-pack --teacher-a parent-a.ckpt [--teacher-b parent-b.ckpt] "
                   "--data selected-prefix.dat --out runs/teachers [--temperature 2 --strength .5 --mixture .5]\n"
                << "synapticgenesis live --curriculum curriculum.sg --replay stage --teacher-bundle "
                   "runs/teachers --out runs/taught [--teacher-memory-mib 512 --teaching on]\n";
            return 0;
        }
        Args args(argc, argv);
        std::string cmd = argv[1];
        if (cmd == "train")
            train(args);
        else if (cmd == "sample")
            sample(args);
        else if (cmd == "self-test")
            self_test(args);
        else if (cmd == "evaluate")
            eval_command(args);
        else if (cmd == "storage-bench")
            storage_bench(args);
        else if (cmd == "live")
            live_command(args);
        else if (cmd == "live-test")
            live_test(args);
        else if (cmd == "curriculum-test")
            curriculum_test(args);
        else if (cmd == "graph-test")
            graph_test(args);
        else if (cmd == "decode-bench")
            decode_bench(args);
        else if (cmd == "replay-test")
            replay_test(args);
        else if (cmd == "stage-replay-test")
            stage_replay_test(args);
        else if (cmd == "reduction-test")
            reduction_test(args);
        else if (cmd == "adaptive-test")
            adaptive_test(args);
        else if (cmd == "trace-test")
            adaptive_test(args, 3);
        else if (cmd == "gated-test")
            adaptive_test(args, 4);
        else if (cmd == "selective-test")
            adaptive_test(args, 5);
        else if (cmd == "associative-test")
            associative_test(args);
        else if (cmd == "association-layout-bench")
            association_layout_bench(args);
        else if (cmd == "memory-bench")
            memory_bench(args);
        else if (cmd == "context-bench")
            context_bench(args);
        else if (cmd == "language-probes")
            probes::run(args);
        else if (cmd == "feedback-test")
            feedback_test(args);
        else if (cmd == "distillation-test")
            distillation_test(args);
        else if (cmd == "teacher-replay-test")
            teacher_replay_test(args);
        else if (cmd == "retention-bench")
            retention_bench(args);
        else if (cmd == "synaptic-test")
            synaptic_test(args);
        else if (cmd == "population-add")
            evolution::add(args);
        else if (cmd == "population-live")
            evolution::live(args);
        else if (cmd == "teacher-pack")
            teacher_pack_command(args);
        else if (cmd == "evolve")
            evolution::run(args);
        else if (cmd == "evolution-test")
            evolution_test(args);
        else
            throw std::runtime_error("Unknown command");
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}
