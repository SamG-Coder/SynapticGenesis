// Neuron-indexed storage experiment. Included after the model definitions.
// Only outgoing LIF projection weights are paged here. The full model is retained
// as a resident correctness oracle: the cache size is NOT total process VRAM.
// Forecast misses always cause a real fetch; predictions never suppress a spike.

void eval_command(const Args &args) {
    args.allow({"checkpoint", "data", "batches", "batch", "context", "output"});
    auto path = args.get("checkpoint");
    State s = header(path);
    Config q = checkpoint_config(s);
    int B = args.num("batch", int(s.meta[5])), T = args.num("context", int(s.meta[6])),
        batches = args.num("batches", 64);
    if (B < 1 || B > 256 || T < 1 || T > 4096 || batches < 1)
        throw std::runtime_error("Invalid evaluation shape");
    Model model(q, B, T);
    load(path, model, s);
    Data data(args.get("data"), T);
    float loss = evaluate(model, data, batches);
    std::string result = "{\"step\":" + std::to_string(s.meta[7]) +
                         ",\"loss_nats_per_byte\":" + std::to_string(loss) +
                         ",\"bits_per_byte\":" + std::to_string(loss / std::log(2.f)) +
                         ",\"evaluated_bytes\":" + std::to_string(uint64_t(B) * T * batches) +
                         ",\"corpus_hash\":\"" + std::to_string(data.hash) + "\"}";
    std::cout << result << "\n";
    if (!args.get("output").empty()) {
        std::ofstream f(args.get("output"));
        f << result << "\n";
    }
}

__global__ void indexed_add(float *out, const float *columns, const float *spikes, int start, int count,
                            int C) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < C) {
        float sum = 0;
        for (int j = 0; j < count; ++j) {
            float s = spikes[start + j];
            if (s > 0)
                sum += columns[j * C + c];
            else if (s < 0)
                sum -= columns[j * C + c];
        }
        out[c] += sum;
    }
}

void export_neurons(const fs::path &path, Model &model) {
    auto weights = model.w.host();
    auto q = model.q;
    std::vector<float> columns(size_t(q.l) * q.h * q.c);
    for (int l = 0; l < q.l; ++l)
        for (int h = 0; h < q.h; ++h)
            for (int c = 0; c < q.c; ++c)
                columns[(size_t(l) * q.h + h) * q.c + c] =
                    weights[model.a.layers[l].wo + size_t(c) * q.h + h];
    uint64_t neurons = uint64_t(q.l) * q.h;
    std::array<uint64_t, 8> info{0x315844494e504742ull,
                                 1,
                                 uint64_t(q.c),
                                 uint64_t(q.h),
                                 uint64_t(q.l),
                                 neurons,
                                 hash_bytes(columns.data(), columns.size() * 4),
                                 0};
    std::vector<uint64_t> offsets(neurons + 1);
    uint64_t base = 64 + (neurons + 1) * 8;
    for (size_t i = 0; i < offsets.size(); ++i)
        offsets[i] = base + i * q.c * 4;
    std::ofstream f(path, std::ios::binary);
    write_raw(f, info.data(), 8);
    write_raw(f, offsets.data(), offsets.size());
    write_raw(f, columns.data(), columns.size());
}

struct NeuronCache {
    std::ifstream file;
    int C, H, L, block, slots;
    uint64_t clock = 0;
    std::vector<uint64_t> offsets, ages;
    std::vector<int> ids;
    Buf gpu;
    float *pinned = nullptr;
    uint64_t demand_reads = 0, prefetch_reads = 0, hits = 0, bytes = 0;
    NeuronCache(const fs::path &path, int b, int count)
        : file(path, std::ios::binary), block(b), slots(count) {
        std::array<uint64_t, 8> info{};
        read_raw(file, info.data(), 8);
        if (info[0] != 0x315844494e504742ull || info[1] != 1)
            throw std::runtime_error("Invalid neuron index");
        C = int(info[2]);
        H = int(info[3]);
        L = int(info[4]);
        if (C < 1 || C > 2048 || H < 1 || H > 8192 || L < 1 || L > 32 || H % block || slots < 1)
            throw std::runtime_error("Invalid indexed dimensions");
        offsets.resize(size_t(H) * L + 1);
        read_raw(file, offsets.data(), offsets.size());
        uint64_t base = 64 + offsets.size() * 8;
        for (size_t i = 0; i < offsets.size(); ++i)
            if (offsets[i] != base + i * C * 4)
                throw std::runtime_error("Corrupt neuron offset");
        if (fs::file_size(path) != offsets.back())
            throw std::runtime_error("Invalid indexed file length");
        std::vector<float> payload(size_t(L) * H * C);
        read_raw(file, payload.data(), payload.size());
        if (hash_bytes(payload.data(), payload.size() * 4) != info[6])
            throw std::runtime_error("Indexed payload checksum mismatch");
        gpu = Buf(size_t(slots) * block * C);
        ids.assign(slots, -1);
        ages.assign(slots, 0);
        ck(cudaMallocHost(&pinned, size_t(block) * C * 4));
    }
    ~NeuronCache() {
        if (pinned)
            cudaFreeHost(pinned);
    }
    float *get(int key, bool prefetch = false) {
        auto at = std::find(ids.begin(), ids.end(), key);
        int slot;
        if (at != ids.end()) {
            slot = int(at - ids.begin());
            if (!prefetch)
                ++hits;
        } else {
            slot = int(std::min_element(ages.begin(), ages.end()) - ages.begin());
            file.clear();
            file.seekg(std::streamoff(offsets.at(size_t(key) * block)));
            read_raw(file, pinned, size_t(block) * C);
            ck(cudaMemcpy(gpu.p + size_t(slot) * block * C, pinned, size_t(block) * C * 4,
                          cudaMemcpyHostToDevice));
            ids[slot] = key;
            bytes += uint64_t(block) * C * 4;
            if (prefetch)
                ++prefetch_reads;
            else
                ++demand_reads;
        }
        ages[slot] = ++clock;
        return gpu.p + size_t(slot) * block * C;
    }
};

void storage_bench(const Args &args) {
    args.allow({"checkpoint", "data", "steps", "block", "cache-blocks", "prefetch", "out"});
    fs::path out = args.get("out", "reports/storage");
    fs::create_directories(out);
    auto path = args.get("checkpoint");
    State s = header(path);
    Config q = checkpoint_config(s);
    int steps = args.num("steps", 128), block = args.num("block", 8), slots = args.num("cache-blocks", 16),
        prefetch = args.num("prefetch", 2);
    if (steps < 1 || steps > 10000 || block < 1 || block > q.h || q.h % block || slots < 1 ||
        slots > q.l * q.h / block || prefetch < 0 || prefetch > slots)
        throw std::runtime_error("Invalid paging benchmark options");
    Model model(q, 1, 1);
    load(path, model, s);
    auto weights = model.w.host();
    export_neurons(out / "outgoing-weights.sni", model);
    NeuronCache page(out / "outgoing-weights.sni", block, slots);
    Data data(args.get("data"), 1);
    Buf resident(q.c), sparse(q.c), paged(q.c);
    std::vector<std::vector<int>> prediction(q.l);
    uint64_t active = 0, needed = 0, predicted = 0, useful = 0;
    double dense_ms = 0, sparse_ms = 0, paged_ms = 0;
    float max_error = 0, sparse_error = 0;
    auto elapsed = [](auto t) {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
    };
    size_t begin = std::min<size_t>(4096, data.bytes.size() / 4);
    for (int t = 0; t < steps; ++t) {
        int token = data.bytes[(begin + t) % data.bytes.size()];
        if (token == 30) {
            model.reset();
            for (auto &v : prediction)
                v.clear();
            token = 10;
        }
        model.forward({token}, nullptr, true);
        ck(cudaDeviceSynchronize());
        for (int l = 0; l < q.l; ++l) {
            auto p = model.a.layers[l];
            auto &cache = model.cache[l];
            auto start = std::chrono::steady_clock::now();
            model.linear(resident.p, cache.s.p, p.wo, p.bo, q.h, q.c);
            ck(cudaDeviceSynchronize());
            dense_ms += elapsed(start);
            start = std::chrono::steady_clock::now();
            spike_add<<<(q.c + 255) / 256, 256>>>(sparse.p, model.w.p + p.wo, cache.s.p, model.w.p + p.bo,
                                                  q.c, q.h);
            ck(cudaDeviceSynchronize());
            sparse_ms += elapsed(start);
            start = std::chrono::steady_clock::now();
            // Forecast is from the preceding token, before reading current activity.
            for (int b : prediction[l])
                page.get(l * (q.h / block) + b, true);
            auto spikes = cache.s.host();
            std::vector<int> blocks;
            for (int b = 0; b < q.h / block; ++b) {
                bool fired = false;
                for (int j = 0; j < block; ++j) {
                    if (spikes[b * block + j] != 0) {
                        ++active;
                        fired = true;
                    }
                }
                if (fired)
                    blocks.push_back(b);
            }
            predicted += prediction[l].size();
            for (int b : prediction[l])
                if (std::find(blocks.begin(), blocks.end(), b) != blocks.end())
                    ++useful;
            needed += blocks.size();
            ck(cudaMemcpy(paged.p, model.w.p + p.bo, q.c * 4, cudaMemcpyDeviceToDevice));
            for (int b : blocks) {
                auto columns = page.get(l * (q.h / block) + b);
                indexed_add<<<(q.c + 255) / 256, 256>>>(paged.p, columns, cache.s.p, b * block, block, q.c);
            }
            // Simple membrane forecast: next input current approximately equals the
            // previous current. This is speculative and is never trusted for correctness.
            auto state = cache.state.host(), z = cache.z.host();
            auto adaptation = q.adaptive() ? cache.adapt_state.host() : std::vector<float>{};
            std::vector<std::pair<int, int>> scores;
            for (int b = 0; b < q.h / block; ++b) {
                int score = 0;
                for (int j = 0; j < block; ++j) {
                    int h = b * block + j;
                    float beta = 1 / (1 + std::exp(-weights[p.leak + h]));
                    float threshold = 1;
                    if (q.adaptive()) {
                        float k = weights[p.adapt_scale + h];
                        threshold += (std::max(k, 0.f) + std::log1p(std::exp(-std::abs(k)))) * adaptation[h];
                    }
                    if (std::abs(beta * state[h] + z[h]) >= threshold)
                        ++score;
                }
                if (score)
                    scores.push_back({score, b});
            }
            std::sort(scores.begin(), scores.end(), [](auto a, auto b) {
                return a.first > b.first || (a.first == b.first && a.second < b.second);
            });
            prediction[l].clear();
            for (int i = 0; i < std::min(prefetch, int(scores.size())); ++i)
                prediction[l].push_back(scores[i].second);
            ck(cudaDeviceSynchronize());
            paged_ms += elapsed(start);
            auto reference = resident.host();
            max_error = std::max(max_error, maxdiff(reference, paged.host()));
            sparse_error = std::max(sparse_error, maxdiff(reference, sparse.host()));
        }
    }
    require(max_error < 5e-4, "Paged projection differs from resident projection");
    require(sparse_error < 5e-4, "Sparse projection differs from resident projection");
    double calls = double(steps) * q.l;
    std::ofstream f(out / "storage.json");
    f << std::setprecision(10)
      << "{\n  \"scope\":\"outgoing LIF projections only; resident full model retained as oracle\",\n"
      << "  \"steps\":" << steps << ",\n  \"checkpoint_step\":" << s.meta[7]
      << ",\n  \"block_neurons\":" << block << ",\n  \"cache_blocks\":" << slots << ",\n"
      << "  \"outgoing_weight_bytes\":" << uint64_t(q.l) * q.h * q.c * 4
      << ",\n  \"logical_cache_bytes\":" << uint64_t(slots) * block * q.c * 4 << ",\n"
      << "  \"active_neuron_fraction\":" << active / (calls * q.h)
      << ",\n  \"required_block_fraction\":" << needed / (calls * (q.h / block)) << ",\n"
      << "  \"prefetch_precision\":" << (predicted ? double(useful) / predicted : 0)
      << ",\n  \"prefetch_block_recall\":" << (needed ? double(useful) / needed : 0) << ",\n"
      << "  \"prefetch_reads\":" << page.prefetch_reads << ",\n  \"demand_reads\":" << page.demand_reads
      << ",\n  \"cache_hits\":" << page.hits << ",\n  \"transferred_bytes\":" << page.bytes << ",\n"
      << "  \"resident_dense_us_per_projection\":" << 1000 * dense_ms / calls
      << ",\n  \"resident_spike_add_us_per_projection\":" << 1000 * sparse_ms / calls
      << ",\n  \"indexed_paged_us_per_projection\":" << 1000 * paged_ms / calls << ",\n"
      << "  \"paged_max_abs_error\":" << max_error << ",\n  \"spike_add_max_abs_error\":" << sparse_error
      << ",\n"
      << "  \"timing_note\":\"Local warm-cache microbenchmark. Indexed timing includes CPU spike readback, "
         "serialized file reads, transfers, LRU bookkeeping and forecast generation. No asynchronous "
         "overlap, cold-SSD test or end-to-end speedup claim.\",\n"
      << "  \"correctness\":\"All observed active blocks fetched, including forecast misses; no "
         "predicted-inactive neurons discarded.\"\n}\n";
    std::cout << "Indexed projection PASS: error=" << max_error << " active=" << active / (calls * q.h)
              << " block_coverage=" << needed / (calls * (q.h / block))
              << " resident_us=" << 1000 * dense_ms / calls << " paged_us=" << 1000 * paged_ms / calls
              << "\n";
}
