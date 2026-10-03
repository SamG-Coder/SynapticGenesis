// Host numerical references and a compile-only device instantiation. No CUDA API calls.
#include "../experiments/membrane_penalty.cuh"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

__global__ void membrane_penalty_compile(float *cost, float *gradient, const float *u, int count,
                                         float band, float scale) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        auto term = membrane_penalty::excess(u[i], band);
        cost[i] = scale * term.cost;
        gradient[i] = scale * term.derivative;
    }
}

namespace {
void require(bool value, const char *message) {
    if (!value) throw std::runtime_error(message);
}
double loss(const std::vector<double> &z, const std::vector<double> &fixed_spikes,
            double beta, double initial, double band) {
    double reset = initial, total = 0;
    for (size_t t = 0; t < z.size(); ++t) {
        double u = beta * reset + z[t];
        total += membrane_penalty::excess(u, band).cost;
        reset = u - fixed_spikes[t];
    }
    return total / double(z.size());
}
} // namespace

int main(int argc, char **argv) {
    try {
        require(argc == 3 && std::string(argv[1]) == "--out", "Expected --out <fresh-report.json>");
        std::filesystem::path out(argv[2]);
        require(!std::filesystem::exists(out), "Use a fresh host report");
        double scalar_error = 0, recurrence_error = 0, leak_error = 0;
        int scalar_checks = 0, recurrence_checks = 0;
        constexpr double epsilon = 1e-5, band = 1.5;
        for (double u : {-50., -2., -1.6, -1.5, -1., 0., 1., 1.5, 1.6, 2., 50.}) {
            auto term = membrane_penalty::excess(u, band);
            double independent = std::max(0., std::abs(u) - band);
            require(term.cost == .5 * independent * independent, "Point cost differs");
            require(term.derivative == (u < 0 ? -independent : independent), "Point derivative differs");
            auto opposite = membrane_penalty::excess(-u, band);
            require(term.cost == opposite.cost && term.derivative == -opposite.derivative, "Signed symmetry differs");
            double numerical = (membrane_penalty::excess(u + epsilon, band).cost -
                                membrane_penalty::excess(u - epsilon, band).cost) / (2 * epsilon);
            scalar_error = std::max(scalar_error, std::abs(term.derivative - numerical));
            auto single = membrane_penalty::excess(float(u), float(band));
            require(std::isfinite(single.cost) && std::isfinite(single.derivative), "Nonfinite float term");
            ++scalar_checks;
        }
        // Freeze the baseline spike resets while perturbing inputs: this is the
        // production detached-reset derivative convention, not a smooth spike approximation.
        for (const auto &z : std::vector<std::vector<double>>{
                 {3., 4., 2., 3., 2., 4., 3.}, {-3., -4., -2., -3., -2., -4., -3.},
                 {2.3, -3.2, .2, 4.4, -2.7, -1.9, 3.1}, {.1, .2, -.1, .3, -.2, .1, .2}}) {
            const double beta = .73, initial = .2;
            double reset = initial;
            std::vector<double> u, previous, spikes, dz(z.size());
            for (double drive : z) {
                previous.push_back(reset);
                u.push_back(beta * reset + drive);
                spikes.push_back(double(u.back() >= 1) - double(u.back() <= -1));
                reset = u.back() - spikes.back();
            }
            double carry = 0, db = 0;
            for (int t = int(z.size()) - 1; t >= 0; --t) {
                double du = membrane_penalty::excess(u[t], band).derivative / double(z.size()) + beta * carry;
                dz[t] = du;
                db += du * previous[t];
                carry = du;
            }
            for (size_t t = 0; t < z.size(); ++t) {
                auto plus = z, minus = z;
                plus[t] += epsilon;
                minus[t] -= epsilon;
                double numerical = (loss(plus, spikes, beta, initial, band) -
                                    loss(minus, spikes, beta, initial, band)) / (2 * epsilon);
                recurrence_error = std::max(recurrence_error, std::abs(dz[t] - numerical));
                ++recurrence_checks;
            }
            double numerical = (loss(z, spikes, beta + epsilon, initial, band) -
                                loss(z, spikes, beta - epsilon, initial, band)) / (2 * epsilon);
            leak_error = std::max(leak_error, std::abs(db - numerical));
        }
        require(scalar_error < 3e-6 && recurrence_error < 1e-7 && leak_error < 1e-6,
                "Detached-recurrence finite difference exceeds tolerance");
        std::filesystem::create_directories(out.parent_path());
        std::ofstream stream(out);
        require(bool(stream), "Cannot write host report");
        stream << std::setprecision(17)
               << "{\n  \"passed\": true,\n  \"scalar_checks\": " << scalar_checks
               << ",\n  \"recurrence_drive_checks\": " << recurrence_checks
               << ",\n  \"recurrence_leak_checks\": 4,\n  \"band\": " << band
               << ",\n  \"maximum_scalar_error\": " << scalar_error
               << ",\n  \"maximum_recurrence_error\": " << recurrence_error
               << ",\n  \"maximum_leak_error\": " << leak_error
               << ",\n  \"cuda_execution\": false,\n  \"production_integration\": false,"
               << "\n  \"fixed_spike_resets_during_finite_difference\": true\n}\n";
        stream.close();
        require(bool(stream), "Host report write failed");
        std::cout << "PASS: point objective and detached recurrence; no CUDA execution\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
