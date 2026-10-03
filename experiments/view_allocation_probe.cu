// Optional ownership/allocation diagnostics; no observer in ordinary builds.
#include <cstddef>
#include <cstdint>
namespace view_allocation {
struct Ledger {
    uint64_t live = 0, peak = 0, calls = 0, watched_bytes = 0, watched_calls = 0;
    bool underflow = false;
    void begin(uint64_t watch = 0) {
        peak = live;
        calls = watched_calls = 0;
        watched_bytes = watch;
    }
    void event(size_t bytes, bool allocate) noexcept {
        if (allocate) {
            live += bytes;
            if (live > peak) peak = live;
            ++calls;
            if (bytes == watched_bytes) ++watched_calls;
        } else if (bytes <= live) {
            live -= bytes;
        } else {
            underflow = true;
        }
    }
} ledger;
} // namespace view_allocation
#define SG_BUFFER_ALLOCATION_OBSERVER(bytes, allocate) view_allocation::ledger.event(bytes, allocate)
#define main synapticgenesis_cli_entry
#include "../src/spike_lm.cu"
#undef main
#undef SG_BUFFER_ALLOCATION_OBSERVER

namespace view_allocation {
void write(const fs::path &out, const std::string &name, const std::string &text) {
    std::ofstream file(out / name, std::ios::binary);
    file << text << '\n';
    file.close();
    require(bool(file), "Cannot write allocation diagnostic result");
}
template <class Action> void reject(Action action, const std::string &expected) {
    const auto calls = ledger.calls, live = ledger.live;
    bool rejected = false;
    try { action(); } catch (const std::runtime_error &error) {
        rejected = std::string(error.what()) == expected;
    }
    require(rejected, ("Expected constructor rejection: " + expected).c_str());
    require(ledger.calls == calls && ledger.live == live, "Rejected constructor allocated float buffers");
}
void host_test(const fs::path &out) {
    // Layout and rejected constructors only. No successful Model construction,
    // cuBLAS handle, CUDA initialization, device query or kernel is requested.
    std::ostringstream rows;
    const std::array<Config, 4> shapes{{{256, 512, 4, 6}, {512, 2048, 8, 6},
                                      {1024, 4096, 8, 6}, {2048, 8192, 8, 6}}};
    const std::array<uint64_t, 4> expected{{1951624, 27260432, 104851472, 411028496}};
    for (size_t i = 0; i < shapes.size(); ++i) {
        auto q = shapes[i];
        Layout layout(q);
        uint64_t formula = 513ull * q.c + 256 + uint64_t(q.l) *
            (3ull * q.c * q.h + 35ull * q.c + 103ull * q.h + 98);
        require(layout.n == formula && formula == expected[i], "Associative layout count differs");
        rows << (i ? "," : "") << "{\"parameters\":" << layout.n
             << ",\"spiking_neurons\":" << q.h * q.l
             << ",\"five_parameter_arrays_bytes\":" << 20 * layout.n << "}";
    }
    int rejected = 0;
    Config valid{32, 64, 2, 6};
    for (auto buffers : {ModelBuffers::learning, ModelBuffers::frozen_forward}) {
        for (int batch : {-1, 0, 257, std::numeric_limits<int>::max()}) {
            reject([&]() { Model invalid(valid, batch, 16, buffers); }, "Invalid batch/context");
            ++rejected;
        }
        for (int time : {-1, 0, 4097, std::numeric_limits<int>::max()}) {
            reject([&]() { Model invalid(valid, 1, time, buffers); }, "Invalid batch/context");
            ++rejected;
        }
        for (Config q : {Config{7,64,2,6}, Config{2049,64,2,6}, Config{32,7,2,6},
                         Config{32,8193,2,6}, Config{32,64,0,6}, Config{32,64,33,6},
                         Config{32,64,2,0}, Config{32,64,2,7}}) {
            reject([&]() { Model invalid(q, 1, 16, buffers); }, "Unsupported model dimensions");
            ++rejected;
        }
    }
    require(!ledger.calls && !ledger.live && !ledger.underflow, "Host test reached float allocation");
    std::ostringstream report;
    report << "{\"passed\":true,\"cuda_work_requested\":false,\"layouts\":[" << rows.str()
           << "],\"rejected_constructors\":" << rejected << ",\"float_allocations\":0}";
    write(out, "host.json", report.str());
    std::cout << "PASS host: 4 layouts, " << rejected << " rejected constructors; no CUDA work requested\n";
}

struct Snapshot {
    std::vector<float> weights, gradient, first, second, decay, recurrence, synapses, synaptic_gradient;
    uint64_t updates, boundaries;
    explicit Snapshot(const Model &model)
        : weights(model.w.host()), gradient(model.g.host()), first(model.m.host()),
          second(model.v.host()), decay(model.decay.host()), recurrence(model.membranes()),
          synapses(model.synapses->host()), synaptic_gradient(model.synapses->task_gradient.host()),
          updates(model.synapses->updates), boundaries(model.synapses->boundaries) {}
    void same(const Model &model) const {
        require(weights == model.w.host() && gradient == model.g.host() && first == model.m.host() &&
                    second == model.v.host() && decay == model.decay.host() && recurrence == model.membranes() &&
                    synapses == model.synapses->host() && synaptic_gradient == model.synapses->task_gradient.host() &&
                    updates == model.synapses->updates && boundaries == model.synapses->boundaries,
                "View construction or destruction changed owner data");
    }
};
void seed(Model &model) {
    auto weights = initialize(model.q, model.a, 4321);
    activate_association_fixture(model.q, model.a, weights);
    if (model.q.gated())
        for (auto layer : model.a.layers)
            for (int i = 0; i < model.q.c * model.q.h; ++i)
                weights[layer.gate_w + i] = .04f * std::sin(float(i) * .17f);
    model.w.put(weights);
    model.g.put(std::vector<float>(model.a.n, .03f));
    model.m.put(std::vector<float>(model.a.n, .0002f));
    model.v.put(std::vector<float>(model.a.n, .002f));
    auto state = model.membranes();
    for (size_t i = 0; i < state.size(); ++i) state[i] = .08f * std::sin(float(i) * .13f);
    model.membranes(state);
    model.synapses->initialize(model.w, .01f, .001f);
    model.synapses->importance.put(std::vector<float>(model.a.n, .02f));
    model.synapses->path.put(std::vector<float>(model.a.n, .0001f));
    model.synapses->task_gradient.put(std::vector<float>(model.a.n, .07f));
    model.synapses->updates = 3;
    model.synapses->boundaries = 2;
}
void aliases(const Model &owner, const Model &view, bool shared) {
    for (auto pair : {std::make_pair(&owner.w, &view.w), {&owner.g, &view.g}, {&owner.m, &view.m},
                      {&owner.v, &view.v}, {&owner.decay, &view.decay}})
        require(pair.first->p == pair.second->p && pair.first->allocation == pair.second->allocation,
                "Execution view did not share its parameter workspace");
    require(owner.synapses == view.synapses, "Execution view did not share synaptic memory");
    require(owner.input != view.input && owner.target != view.target && owner.logits.p != view.logits.p &&
                owner.x.front().p != view.x.front().p, "Execution view aliased temporary computation buffers");
    for (int layer = 0; layer < owner.q.l; ++layer) {
        auto &a = owner.cache[layer];
        auto &b = view.cache[layer];
        require((a.state.p == b.state.p) == shared && a.initial_state.p != b.initial_state.p &&
                    a.norm.p != b.norm.p && a.u.p != b.u.p, "Execution view state ownership differs");
        if (owner.q.secondary())
            require((a.adapt_state.p == b.adapt_state.p) == shared && a.initial_adapt.p != b.initial_adapt.p,
                    "Execution view secondary state ownership differs");
        if (owner.q.associative())
            require((a.fast_memory->state.p == b.fast_memory->state.p) == shared &&
                        a.fast_memory->initial.p != b.fast_memory->initial.p &&
                        a.fast_memory->raw.p != b.fast_memory->raw.p,
                    "Execution view associative state ownership differs");
    }
}
void self_test(const fs::path &out) {
    std::ostringstream allocation_rows;
    int allocation_cases = 0, numerical_cases = 0, owner_rejections = 0;
    for (int cell = 1; cell <= 6; ++cell)
        for (bool shared : {false, true}) {
            require(!ledger.live, "Float buffers survived prior allocation case");
            Model owner(Config{32,64,2,cell}, 3, 16);
            seed(owner);
            Snapshot before(owner);
            const uint64_t baseline = ledger.live;
            ledger.begin(owner.a.n * 4);
            uint64_t legacy_peak, legacy_final;
            {
                Model legacy(owner.q, owner.B, 7);
                LiveViews::share(legacy, owner, shared);
                legacy_peak = ledger.peak;
                legacy_final = ledger.live;
                require(ledger.watched_calls == 5, "Legacy control did not allocate five parameter arrays");
                aliases(owner, legacy, shared);
                before.same(owner);
            }
            require(ledger.live == baseline, "Legacy view leaked float buffers");
            ledger.begin(owner.a.n * 4);
            {
                Model view(owner, 7, shared ? ModelViewState::shared : ModelViewState::independent);
                require(!ledger.watched_calls, "Borrowing view allocated a parameter-sized array");
                require(legacy_peak == ledger.peak + 20 * owner.a.n && legacy_final == ledger.live,
                        "Measured construction peak saving or retained arrays differ");
                aliases(owner, view, shared);
                before.same(owner);
                allocation_rows << (allocation_cases++ ? "," : "") << "{\"cell\":" << cell
                    << ",\"shared_recurrence\":" << (shared ? "true" : "false")
                    << ",\"legacy_float_peak_bytes\":" << legacy_peak
                    << ",\"borrowed_float_peak_bytes\":" << ledger.peak
                    << ",\"retained_float_bytes\":" << ledger.live
                    << ",\"avoided_float_bytes\":" << 20 * owner.a.n << "}";
            }
            before.same(owner);
            require(ledger.live == baseline && !ledger.underflow, "Borrowing view leaked float buffers");
            for (int time : {0,4097}) {
                reject([&]() { Model invalid(owner, time, ModelViewState::shared); }, "Invalid batch/context");
                ++owner_rejections;
            }
        }
    // Separate reference allocations compare the entire learned state, with
    // nonzero optimizer and synaptic history. Owner destruction is deliberate.
    for (int cell = 1; cell <= 6; ++cell)
        for (int batch : {1,3})
            for (bool fast : {false,true})
                for (bool shared : {false,true}) {
                    require(!ledger.live, "Float buffers survived prior numerical case");
                    Config q{32,64,2,cell};
                    auto owner = std::make_unique<Model>(q, batch, 16);
                    Model reference(q, batch, 7);
                    seed(*owner); seed(reference);
                    auto view = std::make_unique<Model>(*owner, 7,
                        shared ? ModelViewState::shared : ModelViewState::independent);
                    aliases(*owner, *view, shared);
                    if (!shared) {
                        auto zero = view->membranes();
                        require(std::all_of(zero.begin(), zero.end(), [](float v) { return v == 0; }),
                                "Independent recurrence was not initialized to zero");
                        reference.reset();
                    }
                    auto owner_state = owner->membranes();
                    reference.fast(fast); view->fast(fast);
                    std::vector<int> input(view->N), target(view->N);
                    std::iota(input.begin(), input.end(), 41);
                    for (int step = 4; step <= 6; ++step) {
                        if (step == 5) owner.reset();
                        for (size_t i = 0; i < input.size(); ++i) target[i] = (input[i] + 7) % 256;
                        bool streaming = step != 6;
                        float expected_loss = reference.forward(input, &target, streaming);
                        float actual_loss = view->forward(input, &target, streaming);
                        require(expected_loss == actual_loss && reference.logits.host() == view->logits.host() &&
                                    reference.membranes() == view->membranes(), "View forward/state arithmetic differs");
                        reference.backward(); view->backward();
                        require(reference.g.host() == view->g.host(), "View backward arithmetic differs");
                        require(reference.update(step, .0003f) == view->update(step, .0003f),
                                "View optimizer gradient norm differs");
                        reference.synapses->consolidate(reference.w);
                        view->synapses->consolidate(view->w);
                        Snapshot(reference).same(*view);
                        if (owner)
                            require(owner->membranes() == (shared ? view->membranes() : owner_state),
                                    "View updated the wrong recurrent state");
                        for (auto &token : input) token = (token + 17) % 256;
                    }
                    ++numerical_cases;
                }
    require(!ledger.live && !ledger.underflow, "Float buffers survived numerical tests");
    {
        Model frozen(Config{32,64,2,6}, 1, 8, ModelBuffers::frozen_forward);
        for (auto state : {ModelViewState::shared, ModelViewState::independent}) {
            reject([&]() { Model invalid(frozen, 7, state); }, "Execution view requires learning buffers");
            ++owner_rejections;
        }
    }
    require(!ledger.live && !ledger.underflow, "Diagnostic float allocations were not released");
    std::ostringstream report;
    report << "{\"passed\":true,\"allocation_cases\":[" << allocation_rows.str()
           << "],\"numerical_cases\":" << numerical_cases << ",\"owner_rejections\":" << owner_rejections
           << ",\"forward_backward_optimizer_state_exact\":true,\"owner_lifetime_checked\":true,"
              "\"float_buffers_released\":true,\"whole_process_peak_measured\":false}";
    write(out, "self-test.json", report.str());
    std::cout << "PASS CUDA: " << allocation_cases << " allocation cases, " << numerical_cases
              << " exact numerical/lifetime cases, " << owner_rejections << " owner guards\n";
}
void measure(const Args &args, const fs::path &out) {
    Config q{args.num("channels",1024), args.num("hidden",4096), args.num("layers",8), 6};
    Layout layout(q);
    ck(cudaFree(nullptr));
    size_t before = 0, total = 0;
    ck(cudaMemGetInfo(&before, &total));
    ledger.begin();
    std::ostringstream rows;
    size_t minimum_boundary_free = before;
    int boundaries = 0;
    {
        LiveEngine engine(q, 128, false);
        auto record = [&](const std::string &phase, int length) {
            ck(cudaDeviceSynchronize());
            size_t free = 0;
            ck(cudaMemGetInfo(&free, &total));
            minimum_boundary_free = std::min(minimum_boundary_free, free);
            rows << (boundaries++ ? "," : "") << "{\"phase\":\"" << phase << "\",\"context\":" << length
                 << ",\"explicit_float_live_bytes\":" << ledger.live
                 << ",\"explicit_float_peak_bytes\":" << ledger.peak
                 << ",\"boundary_free_bytes\":" << free << ",\"tail_views\":" << engine.tails.size()
                 << ",\"replay_views\":" << engine.replay_views.size() << "}";
        };
        record("root-and-speaker",128);
        for (int length : {17,31,63,127,7,16}) {
            engine.view(length);
            record("tail",length);
        }
        for (int length : {17,31,63,127,7,16}) {
            engine.replay_view(length);
            record("replay",length);
        }
        require(engine.tails.size() == 4 && engine.replay_views.size() == 4,
                "Production view caches exceeded their bounds");
    }
    require(!ledger.live && !ledger.underflow, "Measured float buffers were not released");
    std::ostringstream report;
    report << "{\"complete\":true,\"parameters\":" << layout.n << ",\"spiking_neurons\":" << q.h * q.l
           << ",\"channels\":" << q.c << ",\"hidden\":" << q.h << ",\"layers\":" << q.l
           << ",\"chunk\":128,\"baseline_free_bytes\":" << before
           << ",\"minimum_boundary_free_bytes\":" << minimum_boundary_free
           << ",\"explicit_float_peak_bytes\":" << ledger.peak
           << ",\"boundaries\":[" << rows.str() << "],\"float_buffers_released\":true,"
              "\"forward_or_learning_executed\":false,\"whole_process_peak_measured\":false}";
    write(out, "allocation.json", report.str());
    std::cout << "Allocated " << layout.n << " parameters and exercised view eviction; no model computation\n";
}
} // namespace view_allocation
int main(int argc, char **argv) {
    try {
        std::cout.setf(std::ios::unitbuf);
        require(argc >= 2, "Expected host-test, self-test or measure");
        std::string command(argv[1]);
        require(command == "host-test" || command == "self-test" || command == "measure", "Unknown allocation command");
        Args args(argc,argv);
        if (command == "measure") args.allow({"out","channels","hidden","layers"});
        else args.allow({"out"});
        fs::path out = args.get("out");
        require(!out.empty() && !fs::exists(out), "Allocation probe requires a fresh output directory");
        fs::create_directories(out);
        if (command == "host-test") view_allocation::host_test(out);
        else if (command == "self-test") view_allocation::self_test(out);
        else view_allocation::measure(args,out);
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
