// Read-only observations around the production source update. Model mathematics
// remain in the shared learner; sampled coordinates are not full tensor norms.
#pragma once

namespace learning_scale {
struct Moments {
    uint64_t count = 0;
    double absolute = 0, squares = 0, maximum = 0;
    void add(double value) {
        require(std::isfinite(value), "Nonfinite diagnostic value");
        ++count;
        absolute += std::abs(value);
        squares += value * value;
        maximum = std::max(maximum, std::abs(value));
    }
    void json(std::ostream &out) const {
        require(count > 0, "Empty diagnostic statistic");
        out << "{\"count\":" << count << ",\"mean_absolute\":" << absolute / count
            << ",\"rms\":" << std::sqrt(squares / count) << ",\"maximum_absolute\":" << maximum << '}';
    }
};

std::vector<uint64_t> coordinates(size_t begin, size_t size, size_t limit = 64) {
    require(size > 0 && limit > 0, "Invalid coordinate sample size");
    size_t count = std::min(size, limit);
    std::vector<uint64_t> result;
    for (size_t i = 0; i < count; ++i)
        result.push_back(begin + (count == 1 ? 0 : i * (size - 1) / (count - 1)));
    return result;
}

std::vector<uint64_t> points(const std::string &text, uint64_t end) {
    require(!text.empty(), "Missing observation points");
    std::istringstream input(text);
    std::string token;
    std::vector<uint64_t> result;
    while (std::getline(input, token, ',')) {
        require(!token.empty() && token.find_first_not_of("0123456789") == std::string::npos,
                "Observation points must be positive decimal integers");
        uint64_t value = std::stoull(token);
        require(value > 0 && value <= end && (result.empty() || value > result.back()),
                "Observation points must increase and fit the run");
        result.push_back(value);
    }
    require(text.back() != ',' && !result.empty() && result.front() == 1 && result.back() == end,
            "Observation points must include the first and final source updates");
    return result;
}

struct Role {
    std::string name;
    size_t tensor_begin, tensor_size, sample_begin, sample_size;
};
struct Plan {
    std::vector<Role> roles;
    std::vector<uint64_t> offsets;
    explicit Plan(Config q) {
        Layout a(q);
        require(q.associative(), "Learning-scale observer requires the associative cell");
        auto add = [&](const std::string &name, size_t begin, size_t count) {
            auto selected = coordinates(begin, count);
            roles.push_back({name, begin, count, offsets.size(), selected.size()});
            offsets.insert(offsets.end(), selected.begin(), selected.end());
        };
        add("embedding", a.emb, 256ull * q.c);
        for (size_t i = 0; i < a.layers.size(); ++i) {
            auto p = a.layers[i];
            std::string prefix = "layer-" + std::to_string(i) + "/";
            add(prefix + "normalization_gain", p.gain, q.c);
            add(prefix + "input_projection", p.wi, size_t(q.h) * q.c);
            add(prefix + "input_bias", p.bi, q.h);
            add(prefix + "output_projection", p.wo, size_t(q.c) * q.h);
            add(prefix + "output_bias", p.bo, q.c);
            add(prefix + "membrane_leak_logit", p.leak, q.h);
            add(prefix + "trace_retention_logit", p.adapt_leak, q.h);
            add(prefix + "trace_gain_logit", p.adapt_scale, q.h);
            add(prefix + "retention_projection", p.gate_w, size_t(q.h) * q.c);
            add(prefix + "retention_bias", p.gate_b, q.h);
            add(prefix + "association_projection", p.association_w, size_t(association::packed) * q.h);
            add(prefix + "association_bias", p.association_b, association::packed);
            add(prefix + "association_output", p.association_out, size_t(q.c) * association::width);
            add(prefix + "association_output_bias", p.association_bias, q.c);
        }
        add("final_normalization_gain", a.final_gain, q.c);
        add("vocabulary_projection", a.head, 256ull * q.c);
        add("vocabulary_bias", a.bias, 256);
        size_t covered = 0;
        for (const auto &role : roles) {
            require(role.tensor_begin == covered, "Coordinate role layout has a gap or overlap");
            covered += role.tensor_size;
        }
        require(covered == a.n, "Coordinate roles omit model parameters");
    }
    void json(std::ostream &out) const {
        out << "{\"sample_limit_per_tensor\":64,\"roles\":[";
        for (size_t i = 0; i < roles.size(); ++i) {
            const auto &r = roles[i];
            if (i) out << ',';
            out << "{\"name\":\"" << r.name << "\",\"tensor_begin\":" << r.tensor_begin
                << ",\"tensor_size\":" << r.tensor_size << ",\"sample_begin\":" << r.sample_begin
                << ",\"sample_size\":" << r.sample_size << '}';
        }
        out << "],\"offsets\":[";
        for (size_t i = 0; i < offsets.size(); ++i) {
            if (i) out << ',';
            out << offsets[i];
        }
        out << "]}\n";
    }
};

__global__ void gather(const float *weights, const float *gradient, const uint64_t *offsets,
                       float *selected_weights, float *selected_gradient, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        selected_weights[i] = weights[offsets[i]];
        if (gradient) selected_gradient[i] = gradient[offsets[i]];
    }
}

struct Activity {
    Moments membrane, drive, emission, normalized_input;
    uint64_t events = 0, zero_surrogate = 0;
    void add(float u, float z, float spike, float emitted) {
        require(spike == float((u >= 1.f) - (u <= -1.f)), "Observed spike disagrees with membrane");
        membrane.add(u);
        drive.add(z);
        emission.add(emitted);
        events += spike != 0;
        float derivative = .3f * (std::max(0.f, 1.f - std::abs(u - 1.f)) +
                                   std::max(0.f, 1.f - std::abs(u + 1.f)));
        zero_surrogate += derivative == 0;
    }
    void json(std::ostream &out) const {
        require(membrane.count > 0, "Empty neuron activity record");
        out << "{\"neuron_positions\":" << membrane.count
            << ",\"spike_event_fraction\":" << double(events) / membrane.count
            << ",\"zero_local_spike_surrogate_fraction\":" << double(zero_surrogate) / membrane.count
            << ",\"membrane\":";
        membrane.json(out);
        out << ",\"drive\":";
        drive.json(out);
        out << ",\"emission\":";
        emission.json(out);
        out << ",\"normalized_input\":";
        normalized_input.json(out);
        out << '}';
    }
};

struct Observer : LiveSourceObserver {
    Plan plan;
    const fs::path out;
    const std::vector<uint64_t> observations;
    uint64_t *device_offsets = nullptr;
    Buf selected_weights, selected_gradient;
    std::vector<float> initial, before, after, clipped_gradient;
    std::vector<Activity> activity;
    Episode episode{};
    uint64_t source_update = 0, source_global_update = 0;
    double seconds = 0;
    size_t records = 0;
    bool active = false;
    std::ofstream output;
    Observer(Model &model, fs::path directory, std::vector<uint64_t> selected)
        : plan(model.q), out(std::move(directory)), observations(std::move(selected)),
          selected_weights(plan.offsets.size()), selected_gradient(plan.offsets.size()),
          output(out / "observations.jsonl", std::ios::binary) {
        require(bool(output), "Cannot create observation output");
        output << std::setprecision(17);
        ck(cudaMalloc(&device_offsets, plan.offsets.size() * sizeof(uint64_t)));
        ck(cudaMemcpy(device_offsets, plan.offsets.data(), plan.offsets.size() * sizeof(uint64_t), cudaMemcpyHostToDevice));
        initial = read(model, false).first;
        dump(out / "initial-coordinates.f32", initial);
        std::ofstream manifest(out / "coordinates.json", std::ios::binary);
        plan.json(manifest);
        manifest.close();
        require(bool(manifest), "Cannot write coordinate manifest");
    }
    ~Observer() override { cudaFree(device_offsets); }
    std::pair<std::vector<float>, std::vector<float>> read(Model &model, bool gradients) {
        gather<<<unsigned((plan.offsets.size() + 255) / 256), 256>>>(model.w.p, gradients ? model.g.p : nullptr,
            device_offsets, selected_weights.p, selected_gradient.p, int(plan.offsets.size()));
        ck(cudaGetLastError());
        return {selected_weights.host(), gradients ? selected_gradient.host() : std::vector<float>{}};
    }
    void before_source(LiveEngine &engine, const LiveCorpus &, const State &state, const Episode &current) override {
        source_update = state.meta[24] + 1;
        active = std::binary_search(observations.begin(), observations.end(), source_update);
        if (!active) return;
        auto started = std::chrono::steady_clock::now();
        episode = current;
        before = read(engine.root, false).first;
        seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    }
    void after_source(LiveEngine &engine, const LiveCorpus &, const State &state, const Episode &current) override {
        if (!active) return;
        auto started = std::chrono::steady_clock::now();
        require(current.document == episode.document && current.offset == episode.offset && current.length == episode.length,
                "Observed source episode changed");
        auto &model = engine.view(int(current.length));
        auto values = read(model, true);
        after = std::move(values.first);
        clipped_gradient = std::move(values.second);
        source_global_update = state.meta[7];
        activity.clear();
        for (auto &cache : model.cache) {
            auto u = cache.u.host(), z = cache.z.host(), spikes = cache.s.host(), emission = cache.emission.host();
            Activity item;
            require(u.size() == size_t(model.N) * model.q.h && z.size() == u.size() &&
                    spikes.size() == u.size() && emission.size() == u.size(), "Malformed activity cache");
            for (size_t i = 0; i < u.size(); ++i) item.add(u[i], z[i], spikes[i], emission[i]);
            for (float value : cache.norm.host()) item.normalized_input.add(value);
            activity.push_back(item);
        }
        seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    }
    void finish(const LiveResult &result, const State &state) {
        if (!active) return;
        auto started = std::chrono::steady_clock::now();
        std::string prefix = "source-" + std::to_string(source_update);
        dump(out / (prefix + "-before.f32"), before);
        dump(out / (prefix + "-after.f32"), after);
        dump(out / (prefix + "-clipped-gradient.f32"), clipped_gradient);
        output << "{\"source_update\":" << source_update << ",\"source_global_update\":" << source_global_update
               << ",\"global_updates_after_tick\":" << state.meta[7] << ",\"document\":" << episode.document
               << ",\"byte_offset\":" << episode.offset << ",\"target_bytes\":" << episode.length
               << ",\"source_loss\":" << result.loss << ",\"source_gradient_norm_before_clip\":" << result.gradient_norm
               << ",\"source_clip_factor\":" << (state.hp[2] > 0 && result.gradient_norm > state.hp[2] ? state.hp[2] / result.gradient_norm : 1.f)
               << ",\"replayed_bytes_this_tick\":" << result.replayed << ",\"generated_bytes_this_tick\":" << result.speech.size()
               << ",\"roles\":[";
        for (size_t i = 0; i < plan.roles.size(); ++i) {
            const auto &role = plan.roles[i];
            Moments old, current, delta, displacement, gradient;
            for (size_t j = role.sample_begin; j < role.sample_begin + role.sample_size; ++j) {
                old.add(before[j]); current.add(after[j]);
                delta.add(double(after[j]) - before[j]);
                displacement.add(double(after[j]) - initial[j]);
                gradient.add(clipped_gradient[j]);
            }
            if (i) output << ',';
            output << "{\"name\":\"" << role.name << "\",\"weights_before\":"; old.json(output);
            output << ",\"weights_after\":"; current.json(output);
            output << ",\"source_update_delta\":"; delta.json(output);
            output << ",\"displacement_from_initial\":"; displacement.json(output);
            output << ",\"clipped_gradient\":"; gradient.json(output);
            output << '}';
        }
        output << "],\"pre_update_forward_layers\":[";
        for (size_t i = 0; i < activity.size(); ++i) {
            if (i) output << ',';
            activity[i].json(output);
        }
        output << "]}\n";
        output.flush();
        require(bool(output), "Observation output failed");
        ++records;
        seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    }
};

void host_test() {
    require(coordinates(10, 1) == std::vector<uint64_t>{10}, "Single coordinate differs");
    require(coordinates(10, 8, 4) == std::vector<uint64_t>({10, 12, 14, 17}), "Spaced coordinates differ");
    Plan small(Config{8, 32, 2, 6});
    Plan large(Config{1024, 4096, 8, 6});
    require(small.roles.size() == 32 && large.roles.size() == 116 && large.offsets.size() == 116 * 64,
            "Parameter role coverage differs");
    require(std::adjacent_find(large.offsets.begin(), large.offsets.end(), [](uint64_t a, uint64_t b) { return a >= b; }) == large.offsets.end(),
            "Sampled coordinates overlap");
    Moments m;
    for (double v : {-3., 4., 0.}) m.add(v);
    require(m.count == 3 && m.absolute == 7 && m.squares == 25 && m.maximum == 4, "Moments differ");
    Activity a;
    for (float u : {-2.f, -1.f, 0.f, 1.f, 2.f, 1e-9f, -1e-9f})
        a.add(u, u, float((u >= 1.f) - (u <= -1.f)), u);
    require(a.events == 4 && a.zero_surrogate == 5 && a.membrane.count == 7, "Threshold/cancellation statistics differ");
    require(points("1,4,16", 16) == std::vector<uint64_t>({1, 4, 16}), "Observation schedule differs");
    int rejected = 0;
    for (const std::string &text : {"", "1,1,16", "0,16", "1,17", "1,4", "1,16,", "1,-2,16", "1,2x,16"}) {
        try { points(text, 16); } catch (const std::exception &) { ++rejected; }
    }
    try { m.add(std::numeric_limits<double>::infinity()); } catch (const std::exception &) { ++rejected; }
    try { a.add(0.f, 0.f, 1.f, 0.f); } catch (const std::exception &) { ++rejected; }
    try { coordinates(0, 0); } catch (const std::exception &) { ++rejected; }
    require(rejected == 11, "Malformed diagnostic inputs were accepted");
    std::cout << "{\"passed\":true,\"native_model_calls\":0,\"gpu_work\":false,\"rejected_cases\":11,"
                 "\"roles_large_model\":116,\"sampled_coordinates_large_model\":7424}\n";
}
} // namespace learning_scale
