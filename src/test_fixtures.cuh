// Test-only perturbations exercise paths that intentionally start at zero.
#pragma once
// Count actual standalone allocations, independently of admission formulas.
uint64_t explicit_model_bytes(const Model &model) {
    uint64_t bytes = (uint64_t(model.input != nullptr) + uint64_t(model.target != nullptr)) *
                     model.N * sizeof(int);
    for (const Buf *buffer :
         {&model.w, &model.g, &model.m, &model.v, &model.decay, &model.finalnorm, &model.finalrs,
          &model.logits, &model.dlogits, &model.losses, &model.loss_weights, &model.dx, &model.dy,
          &model.dnorm, &model.ds, &model.dz, &model.dgate, &model.neuron_partials})
        bytes += buffer->n * 4;
    for (const auto &buffer : model.x)
        bytes += buffer.n * 4;
    for (const auto &cache : model.cache) {
        for (const Buf *buffer :
             {&cache.norm, &cache.rs, &cache.z, &cache.u, &cache.s, &cache.state, &cache.initial_state,
              &cache.adapt, &cache.adapt_state, &cache.initial_adapt, &cache.emission, &cache.gate})
            bytes += buffer->n * 4;
        if (cache.fast_memory) {
            const auto &fast = *cache.fast_memory;
            for (const Buf *buffer :
                 {&fast.raw, &fast.features, &fast.inverse, &fast.reads, &fast.previous, &fast.state,
                  &fast.initial, &fast.dread, &fast.draw, &fast.demission})
                bytes += buffer->n * 4;
        }
    }
    return bytes;
}

void activate_association_fixture(Config q, const Layout &layout, std::vector<float> &weights) {
    if (!q.associative())
        return;
    for (const auto &layer : layout.layers) {
        for (int i = 0; i < q.c * association::width; ++i)
            weights[layer.association_out + i] = .03f * std::sin(float(i) * .23f);
        for (int i = 0; i < 2 * q.h; ++i)
            weights[layer.association_w + 3 * association::width * q.h + i] =
                .015f * std::cos(float(i) * .17f);
    }
}
