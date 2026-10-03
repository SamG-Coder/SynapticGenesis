// Explicit model buffers; CUDA/cuBLAS overhead requires a separate reserve.
#pragma once

uint64_t model_working_bytes(Config q, int batch, int context, ModelBuffers buffers = ModelBuffers::learning) {
    uint64_t n = uint64_t(batch) * context, c = q.c, h = q.h, l = q.l;
    bool learning = buffers == ModelBuffers::learning;
    uint64_t floats = (learning ? 5ull : 1ull) * Layout(q).n +
                      n * (learning ? 4 * c + 517 + (2 + int(q.gated())) * h
                                    : (1 + int(q.associative())) * c + 258) + (l + 1) * n * c +
                      l * (n * c + n + (3 + int(q.secondary()) + int(q.traced()) + int(q.gated())) * n * h +
                           2 * uint64_t(batch) * h * (1 + int(q.secondary())));
    if (learning && batch > 1)
        floats += uint64_t(batch) * h * (q.secondary() ? 3 : 1);
    if (q.associative())
        floats += l * association::Cache::floats(batch, context, q.h, learning);
    return 4 * floats;
}
