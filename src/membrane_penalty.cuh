// The caller supplies strength / (batch * time * hidden * layers).
#pragma once
namespace membrane_penalty {
template <typename T> struct Term {
    T cost;
    T derivative;
};
template <typename T>
__host__ __device__ inline Term<T> excess(T membrane, T band) {
    const T magnitude = membrane < T(0) ? -membrane : membrane;
    const T distance = magnitude > band ? magnitude - band : T(0);
    return {T(0.5) * distance * distance, membrane < T(0) ? -distance : distance};
}
} // namespace membrane_penalty
