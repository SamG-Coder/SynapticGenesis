// Test-only perturbations exercise paths that intentionally start at zero.
#pragma once
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
