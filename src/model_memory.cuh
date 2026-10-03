// Explicit model buffers; CUDA/cuBLAS overhead requires a separate reserve.
#pragma once

uint64_t model_working_bytes(Config q, int batch, int context) {
    uint64_t n = uint64_t(batch) * context, c = q.c, h = q.h, l = q.l;
    uint64_t floats = 5ull * Layout(q).n + n * (4 * c + 517 + (2 + int(q.gated())) * h) + (l + 1) * n * c +
                      l * (n * c + n + (3 + int(q.secondary()) + int(q.traced()) + int(q.gated())) * n * h +
                           2 * uint64_t(batch) * h * (1 + int(q.secondary())));
    if (batch > 1)
        floats += uint64_t(batch) * h * (q.secondary() ? 3 : 1);
    if (q.associative())
        floats += l * association::Cache::floats(batch, context, q.h);
    return 4 * floats;
}
