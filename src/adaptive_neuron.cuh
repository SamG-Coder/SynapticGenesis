// Signed adaptive LIF. One step is one byte, not a biological millisecond.
// a is an EMA of absolute spikes; theta = 1 + softplus(k) * a.
// The entire threshold-scaled reset pulse is stop-gradient. The adaptation
// feedback through |spike| remains differentiable through the surrogate.
#pragma once
__device__ float softplus(float x) {
    return fmaxf(x, 0.f) + log1pf(expf(-fabsf(x)));
}
__global__ void alif_fwd(float *s, float *u, float *adapt, float *state, float *adapt_state,
                         const float *initial, const float *initial_adapt, const float *z, const float *leak,
                         const float *adapt_leak, const float *scale, int B, int T, int H, bool streaming) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= B * H)
        return;
    int b = k / H, j = k % H;
    float beta = sigmoid(leak[j]), rho = sigmoid(adapt_leak[j]), gamma = softplus(scale[j]);
    float reset = initial[k], a = initial_adapt[k];
    for (int t = 0; t < T; ++t) {
        int i = (b * T + t) * H + j;
        float v = beta * reset + z[i], theta = 1 + gamma * a;
        float spike = float(v >= theta) - float(v <= -theta);
        u[i] = v;
        s[i] = spike;
        adapt[i] = a;
        reset = v - theta * spike;
        a = rho * a + (1 - rho) * fabsf(spike);
    }
    if (streaming) {
        state[k] = reset;
        adapt_state[k] = a;
    }
}
__global__ void alif_bwd(float *dz, float *dl, float *dr, float *dk, const float *ds, const float *u,
                         const float *s, const float *adapt, const float *initial, const float *leak,
                         const float *adapt_leak, const float *scale, int B, int T, int H) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= B * H)
        return;
    int b = k / H, j = k % H;
    float beta = sigmoid(leak[j]), rho = sigmoid(adapt_leak[j]), gamma = softplus(scale[j]);
    float carry_u = 0, carry_a = 0, db = 0, d_rho = 0, d_gamma = 0;
    for (int t = T - 1; t >= 0; --t) {
        int i = (b * T + t) * H + j;
        float a = adapt[i], theta = 1 + gamma * a;
        float positive = .3f * fmaxf(0.f, 1.f - fabsf(u[i] - theta));
        float negative = .3f * fmaxf(0.f, 1.f - fabsf(u[i] + theta));
        float d_spike = ds[i] + carry_a * (1 - rho) * s[i];
        float d_theta = d_spike * (negative - positive);
        float du = d_spike * (positive + negative) + beta * carry_u;
        float previous = t ? u[i - H] - (1 + gamma * adapt[i - H]) * s[i - H] : initial[k];
        dz[i] = du;
        db += du * previous;
        d_gamma += d_theta * a;
        d_rho += carry_a * (a - fabsf(s[i]));
        carry_a = rho * carry_a + gamma * d_theta;
        carry_u = du;
    }
    atomicAdd(dl + j, db * beta * (1 - beta));
    atomicAdd(dr + j, d_rho * rho * (1 - rho));
    atomicAdd(dk + j, d_gamma * sigmoid(scale[j]));
}
