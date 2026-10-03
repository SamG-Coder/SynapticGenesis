// Signed LIF with a learned postsynaptic spike trace. The output projection sees
// a fast spike plus a slower filtered signal; it is no longer ternary-only.
// Time is measured in byte steps, not biological milliseconds.
#pragma once
#include "membrane_penalty.cuh"
__global__ void trace_fwd(float *spikes, float *u, float *trace, float *emission, float *state,
                          float *trace_state, const float *initial, const float *initial_trace,
                          const float *z, const float *leak, const float *trace_leak, const float *scale,
                          const float *gate, int B, int T, int H, bool streaming, bool selective) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= B * H)
        return;
    int b = k / H, j = k % H;
    float beta = sigmoid(leak[j]), rho = sigmoid(trace_leak[j]), gamma = softplus(scale[j]);
    float reset = initial[k], a = initial_trace[k];
    for (int t = 0; t < T; ++t) {
        int i = (b * T + t) * H + j;
        float v = beta * reset + z[i];
        float s = float(v >= 1) - float(v <= -1);
        float retain = selective ? sigmoid(trace_leak[j] + gate[i]) : rho;
        a = retain * a + (1 - retain) * s;
        spikes[i] = s;
        u[i] = v;
        trace[i] = a;
        float read = gate && !selective ? 2 * sigmoid(gate[i]) : 1;
        emission[i] = s + gamma * read * a;
        reset = v - s;
    }
    if (streaming) {
        state[k] = reset;
        trace_state[k] = a;
    }
}
template <bool regularized = false>
__global__ void trace_bwd(float *dz, float *dl, float *dr, float *dk, const float *dout, const float *u,
                          const float *spikes, const float *trace, const float *initial,
                          const float *initial_trace, const float *leak, const float *trace_leak,
                          const float *scale, const float *gate, float *dgate, int B, int T, int H,
                          float activity_scale, bool selective, float membrane_scale = 0,
                          float membrane_band = 1.5f) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= B * H)
        return;
    int b = k / H, j = k % H;
    float beta = sigmoid(leak[j]), rho = sigmoid(trace_leak[j]), gamma = softplus(scale[j]);
    float carry_u = 0, carry_a = 0, db = 0, d_rho = 0, d_gamma = 0, next_retain = 0;
    for (int t = T - 1; t >= 0; --t) {
        int i = (b * T + t) * H + j;
        float retain = selective ? sigmoid(trace_leak[j] + gate[i]) : rho;
        float read = gate && !selective ? 2 * sigmoid(gate[i]) : 1;
        // The future trace contributes its own retention coefficient, not
        // this timestep's gate. The chunk's incoming trace is detached.
        float da = gamma * read * dout[i] + (selective ? next_retain : rho) * carry_a;
        if (gate && !selective)
            dgate[i] = gamma * trace[i] * dout[i] * read * (1 - .5f * read);
        float ds = dout[i] + (1 - retain) * da + activity_scale * spikes[i];
        float du = ds * surrogate(u[i]) + beta * carry_u;
        // Direct membrane derivative: not multiplied by the spike surrogate.
        // Reset spikes and the incoming recurrent state remain detached.
        if constexpr (regularized)
            du += membrane_scale * membrane_penalty::excess(u[i], membrane_band).derivative;
        float previous_u = t ? u[i - H] - spikes[i - H] : initial[k];
        float previous_a = t ? trace[i - H] : initial_trace[k];
        dz[i] = du;
        db += du * previous_u;
        float drho = da * (previous_a - spikes[i]);
        if (selective) {
            drho *= retain * (1 - retain);
            dgate[i] = drho;
        }
        d_rho += drho;
        d_gamma += read * dout[i] * trace[i];
        carry_u = du;
        carry_a = da;
        next_retain = retain;
    }
    dl[k] = db * beta * (1 - beta);
    dr[k] = selective ? d_rho : d_rho * rho * (1 - rho);
    dk[k] = d_gamma * sigmoid(scale[j]);
}
