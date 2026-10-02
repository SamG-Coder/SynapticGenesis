// Signed LIF with a learned postsynaptic spike trace. The output projection sees
// a fast spike plus a slower filtered signal; it is no longer ternary-only.
// Time is measured in byte steps, not biological milliseconds.
#pragma once
__global__ void trace_fwd(float *spikes, float *u, float *trace, float *emission, float *state,
                          float *trace_state, const float *initial, const float *initial_trace,
                          const float *z, const float *leak, const float *trace_leak, const float *scale,
                          int B, int T, int H, bool streaming) {
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
        a = rho * a + (1 - rho) * s;
        spikes[i] = s;
        u[i] = v;
        trace[i] = a;
        emission[i] = s + gamma * a;
        reset = v - s;
    }
    if (streaming) {
        state[k] = reset;
        trace_state[k] = a;
    }
}
__global__ void trace_bwd(float *dz, float *dl, float *dr, float *dk, const float *dout, const float *u,
                          const float *spikes, const float *trace, const float *initial,
                          const float *initial_trace, const float *leak, const float *trace_leak,
                          const float *scale, int B, int T, int H, float activity_scale) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= B * H)
        return;
    int b = k / H, j = k % H;
    float beta = sigmoid(leak[j]), rho = sigmoid(trace_leak[j]), gamma = softplus(scale[j]);
    float carry_u = 0, carry_a = 0, db = 0, d_rho = 0, d_gamma = 0;
    for (int t = T - 1; t >= 0; --t) {
        int i = (b * T + t) * H + j;
        float da = gamma * dout[i] + rho * carry_a;
        float ds = dout[i] + (1 - rho) * da + activity_scale * spikes[i];
        float du = ds * surrogate(u[i]) + beta * carry_u;
        float previous_u = t ? u[i - H] - spikes[i - H] : initial[k];
        float previous_a = t ? trace[i - H] : initial_trace[k];
        dz[i] = du;
        db += du * previous_u;
        d_rho += da * (previous_a - spikes[i]);
        d_gamma += dout[i] * trace[i];
        carry_u = du;
        carry_a = da;
    }
    atomicAdd(dl + j, db * beta * (1 - beta));
    atomicAdd(dr + j, d_rho * rho * (1 - rho));
    atomicAdd(dk + j, d_gamma * sigmoid(scale[j]));
}
