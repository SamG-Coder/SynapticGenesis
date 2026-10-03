// One persistent runtime: observe a labelled chunk, update, then optionally speak.
// Execution views share actual GPU weights, Adam moments and membrane allocations.
// Updates and graph inference hand off at completion boundaries; no races.
// Learning is surrogate-gradient truncated BPTT, not a biological plasticity rule.
#pragma once

// Live extension of State.meta (0..16 retain the ordinary checkpoint meanings):
// 17 version, 18 membrane floats, 19 document, 20 input byte offset, 21 epochs,
// 22 observed next-byte pairs, 23 speech RNG, 24 online updates, 25 reset pending,
// 26 speech interval, 27 speech length, 28 top-k, 29 prompt hash, 30 spoken bytes.
// 31 extension word count (live v2); hp[0] online rate, hp[5] temperature,
// hp[6] core rate multiplier. See live_replay.cuh for the extension payload.
// Version 3 adds durable per-parameter synaptic memory after the replay payload.
// Version 4 binds an append-only curriculum (extra[14:15], hp[7] base LR),
// retaining optional SI history and the replay reservoir across stage changes.
struct LiveCorpus {
    std::vector<unsigned char> bytes;
    std::vector<std::pair<size_t, size_t>> docs;
    // Earlier documents remain addressable by replay. The online cursor may
    // explicitly repeat only the newly introduced suffix of a curriculum stage.
    size_t first_document = 0;
    struct Feedback {
        size_t answer_start;
        float scale;
    };
    std::map<size_t, Feedback> feedback;
    uint64_t hash;
    explicit LiveCorpus(const fs::path &path) {
        std::ifstream f(path, std::ios::binary);
        if (!f)
            throw std::runtime_error("Cannot open live corpus: " + path.string());
        bytes.assign(std::istreambuf_iterator<char>(f), {});
        hash = hash_bytes(bytes.data(), bytes.size());
        size_t start = 0;
        for (size_t i = 0; i <= bytes.size(); ++i)
            if (i == bytes.size() || bytes[i] == 30) {
                if (i - start >= 2)
                    docs.emplace_back(start, i);
                start = i + 1;
            }
        if (docs.empty())
            throw std::runtime_error("Live corpus has no document with two bytes");
    }
    void validate(const State &s) const {
        if (s.meta[12] != hash)
            throw std::runtime_error(
                "Live resume requires the identical corpus; start a new live run to change it");
        if (first_document >= docs.size() || s.meta[19] < first_document || s.meta[19] >= docs.size() ||
            s.meta[25] > 1)
            throw std::runtime_error("Invalid live document cursor");
        auto doc = docs[size_t(s.meta[19])];
        if (s.meta[20] >= doc.second - doc.first - 1 || (s.meta[25] && s.meta[20]))
            throw std::runtime_error("Invalid live byte cursor");
    }
    void emphasize_answers(size_t first, size_t end, float scale) {
        require(first < end && end <= docs.size() && std::isfinite(scale) && scale >= 1 && scale <= 1000,
                "Invalid answer emphasis range/scale");
        if (scale == 1)
            return;
        const std::string marker = "\nAnswer: ";
        for (size_t i = first; i < end; ++i) {
            auto doc = docs[i];
            std::string text(bytes.begin() + doc.first, bytes.begin() + doc.second);
            size_t at = text.find(marker);
            require(at != std::string::npos && at + marker.size() < text.size() &&
                        text.find(marker, at + marker.size()) == std::string::npos,
                    "Answer emphasis requires exactly one nonempty Answer field per new document");
            feedback[i] = {at + marker.size(), scale};
        }
    }
    float emphasize(Model &model, size_t document, size_t offset, size_t n, float ordinary_loss) const {
        auto it = feedback.find(document);
        if (it == feedback.end())
            return ordinary_loss;
        std::vector<float> weights(n, 1.f);
        bool changed = false;
        for (size_t i = 0; i < n; ++i)
            if (offset + i + 1 >= it->second.answer_start) {
                weights[i] = it->second.scale;
                changed = true;
            }
        return changed ? model.reweight_targets(weights) : ordinary_loss;
    }
    void next(const State &s, int chunk, std::vector<int> &x, std::vector<int> &y) const {
        validate(s);
        auto doc = docs[size_t(s.meta[19])];
        size_t at = doc.first + size_t(s.meta[20]);
        size_t n = std::min(size_t(chunk), doc.second - at - 1);
        x.assign(bytes.begin() + at, bytes.begin() + at + n);
        y.assign(bytes.begin() + at + 1, bytes.begin() + at + n + 1);
    }
    void advance(State &s, size_t n) const {
        auto doc = docs[size_t(s.meta[19])];
        s.meta[20] += n;
        s.meta[22] += n;
        if (s.meta[20] == doc.second - doc.first - 1) {
            s.meta[20] = 0;
            s.meta[25] = 1;
            if (++s.meta[19] == docs.size()) {
                s.meta[19] = first_document;
                ++s.meta[21];
            }
        }
    }
};

void validate_live_holdout(const LiveCorpus &training, const LiveCorpus &holdout) {
    require(training.hash != holdout.hash, "Validation corpus must differ from live learning corpus");
    // Whole-document equality only; source preparation handles substantial
    // paragraph duplicates. This is not a semantic contamination detector.
    std::map<std::pair<size_t, uint64_t>, bool> documents;
    for (auto doc : holdout.docs) {
        size_t length = doc.second - doc.first;
        documents[{length, hash_bytes(holdout.bytes.data() + doc.first, length)}] = true;
    }
    for (auto doc : training.docs) {
        size_t length = doc.second - doc.first;
        require(!documents.count({length, hash_bytes(training.bytes.data() + doc.first, length)}),
                "Learning corpus contains a held-out document");
    }
}

#include "live_replay.cuh"

struct LiveEngine {
    Model root;
    Model speaker;
    std::map<int, std::unique_ptr<Model>> tails;
    std::map<int, std::unique_ptr<Model>> replay_views;
    std::unique_ptr<GraphDecoder> decoder;
    bool fast_math;
    LiveEngine(Config q, int chunk, bool fast) : root(q, 1, chunk), speaker(q, 1, 1), fast_math(fast) {
        speaker.share_runtime(root);
        root.fast(fast);
        speaker.fast(fast);
    }
    Model &view(int n) {
        if (n == root.T)
            return root;
        if (n == 1)
            return speaker;
        auto &p = tails[n];
        if (!p) {
            p = std::make_unique<Model>(root.q, 1, n);
            p->share_runtime(root);
            p->fast(fast_math);
        }
        return *p;
    }
    Model &replay_view(int n) {
        auto &p = replay_views[n];
        if (!p) {
            p = std::make_unique<Model>(root.q, 1, n);
            p->share_parameters(root); // Separate recurrence; shared mutable synapses.
            p->fast(fast_math);
        }
        return *p;
    }
    std::string speak(const std::string &prompt, State &s) {
        ReplayMemory memory(s);
        if (memory.graph() && !decoder)
            decoder = std::make_unique<GraphDecoder>(speaker);
        if (decoder)
            decoder->begin();
        auto step = [&](int byte) {
            if (decoder)
                return decoder->step(byte);
            speaker.forward({byte}, nullptr, true);
            return speaker.logits.host();
        };
        std::vector<float> logits;
        for (unsigned char c : prompt)
            logits = step(int(c));
        std::string result = prompt;
        for (uint64_t i = 0; i < s.meta[27]; ++i) {
            int byte = draw_byte(logits, int(s.meta[28]), s.hp[5], s.meta[23]);
            result.push_back(char(byte));
            logits = step(byte);
        }
        s.meta[30] += s.meta[27];
        return result;
    }
};

struct LiveResult {
    float loss, gradient_norm;
    double activity;
    size_t observed;
    std::string speech;
    float replay_loss = 0;
    size_t replayed = 0;
};
LiveResult live_tick(LiveEngine &engine, const LiveCorpus &data, State &s, const std::string &prompt) {
    std::vector<int> x, y;
    data.next(s, engine.root.T, x, y);
    Episode current{s.meta[19], s.meta[20], x.size()};
    ReplayMemory memory(s);
    if (s.meta[25]) {
        engine.root.reset();
        s.meta[25] = 0;
    }
    auto &model = engine.view(int(x.size()));
    float loss = model.forward(x, &y, true);
    loss = data.emphasize(model, size_t(current.document), size_t(current.offset), x.size(), loss);
    double activity = model.rate();
    model.backward(s.hp[4]);
    float norm = model.update(int(++s.meta[7]), s.hp[0], s.hp[1], s.hp[2], memory.core_scale());
    ++s.meta[24];
    data.advance(s, x.size());
    float replay_loss = 0;
    size_t replayed = 0;
    if (memory.due()) {
        auto episode = memory.choose(current);
        ReplayMemory::validate_episode(episode, data, engine.root.T);
        size_t at = data.docs[size_t(episode.document)].first + size_t(episode.offset);
        replayed = size_t(episode.length);
        std::vector<int> rx(data.bytes.begin() + at, data.bytes.begin() + at + replayed);
        std::vector<int> ry(data.bytes.begin() + at + 1, data.bytes.begin() + at + replayed + 1);
        auto &replay = engine.replay_view(int(replayed));
        replay_loss = replay.forward(rx, &ry); // Reset-state window; no live membrane mutation.
        replay_loss =
            data.emphasize(replay, size_t(episode.document), size_t(episode.offset), replayed, replay_loss);
        replay.backward(s.hp[4]);
        replay.update(int(++s.meta[7]), s.hp[0], s.hp[1], s.hp[2], memory.core_scale());
        memory.completed(replayed);
    }
    memory.remember(current); // Reservoir selection above only sees older observations.
    if (engine.root.synapses->active()) {
        // Consolidate after all learning at the document boundary, including
        // scheduled replay, and before any speech. No extra optimizer step.
        if (s.meta[25])
            engine.root.synapses->consolidate(engine.root.w);
        s.extra[12] = engine.root.synapses->boundaries;
        s.extra[13] = engine.root.synapses->updates;
    }
    std::string speech;
    if (s.meta[26] && s.meta[24] % s.meta[26] == 0)
        speech = engine.speak(prompt, s);
    return {loss, norm, activity, x.size(), speech, replay_loss, replayed};
}

void configure_live(State &s, const Args &args, const LiveCorpus &data, const std::string &prompt) {
    s.synaptic.clear();
    for (int i = 17; i < 32; ++i)
        s.meta[i] = 0;
    s.meta[17] = 1;
    s.meta[12] = data.hash;
    s.meta[23] = args.num("seed", 42);
    s.meta[25] = 1;
    int every = args.num("speak-every", 200), tokens = args.num("tokens", 160), top = args.num("top-k", 40);
    if (every < 0 || tokens < 1 || tokens > 10000 || top < 1 || top > 256 || !s.meta[23])
        throw std::runtime_error("Invalid live speech settings");
    s.meta[26] = every;
    s.meta[27] = tokens;
    s.meta[28] = top;
    s.meta[29] = hash_bytes(prompt.data(), prompt.size());
    s.hp[0] = args.real("lr", .00001f);
    s.hp[4] = args.real("activity-cost", s.hp[4]);
    s.hp[5] = args.real("temperature", .8f);
    if (s.hp[0] <= 0 || s.hp[0] > .1f || s.hp[4] < 0 || s.hp[4] > 100 || s.hp[5] <= 0)
        throw std::runtime_error("Invalid live learning settings");
    ReplayMemory::configure(s, args);
    auto consolidation = args.get("consolidation", "none");
    if (consolidation != "none" && consolidation != "si")
        throw std::runtime_error("--consolidation must be none or si");
    if (consolidation == "si") {
        float strength = args.real("si-strength", .1f), damping = args.real("si-damping", .001f);
        if (strength < 0 || strength > 1000000 || damping <= 0 || damping > 1)
            throw std::runtime_error("Invalid --si-strength or --si-damping");
        s.meta[17] = 3;
        s.extra[9] = 1;
        s.extra[10] = float_word(strength);
        s.extra[11] = float_word(damping);
    } else if (!args.get("si-strength").empty() || !args.get("si-damping").empty())
        throw std::runtime_error("SI settings require --consolidation si");
}

// Bounded telemetry for indefinitely resumed streams. Each logarithmic bin spans
// a factor of 2^(1/32), about 2.2%; reported percentiles are bin upper bounds.
struct LiveLatency {
    std::array<uint64_t, 2048> bins{};
    uint64_t count = 0;
    void add(double ms) {
        double position = std::log2(std::max(1e-6, ms) / 1e-6) * 32;
        size_t index = size_t(std::min(2047.0, std::max(0.0, std::ceil(position))));
        ++bins[index];
        ++count;
    }
    double percentile(double fraction) const {
        if (!count)
            return 0;
        uint64_t needed = uint64_t(std::ceil(count * fraction)), sum = 0;
        for (size_t i = 0; i < bins.size(); ++i) {
            sum += bins[i];
            if (sum >= needed)
                return 1e-6 * std::exp2(double(i) / 32);
        }
        return 1e-6 * std::exp2(double(bins.size() - 1) / 32);
    }
};

#include "live_curriculum.cuh"

void live_command(const Args &args, const fs::path &member_checkpoint = {}) {
    args.allow({"data",
                "out",
                "checkpoint",
                "resume",
                "chunk",
                "channels",
                "hidden",
                "layers",
                "seed",
                "lr",
                "activity-cost",
                "fast",
                "updates",
                "speak-every",
                "tokens",
                "top-k",
                "temperature",
                "prompt",
                "log-every",
                "save-every",
                "validation",
                "eval-batches",
                "replay",
                "replay-every",
                "replay-capacity",
                "replay-seed",
                "core-scale",
                "graph",
                "cell",
                "consolidation",
                "si-strength",
                "si-damping",
                "curriculum",
                "extend-curriculum"});
    fs::path out = args.get("out", "runs/live");
    // Population-owned sessions publish directly to the canonical member file.
    // There is one checkpoint authority, including at an interrupted save.
    fs::path latest = member_checkpoint.empty() ? out / "latest.ckpt" : member_checkpoint;
    bool resume = !args.get("resume").empty();
    std::string source = args.get(resume ? "resume" : "checkpoint");
    if (resume && !args.get("checkpoint").empty())
        throw std::runtime_error("Choose --checkpoint for a new live stream or --resume for its saved state");
    if (!resume && fs::exists(out / "initial.ckpt"))
        throw std::runtime_error("Live run exists; use --resume or a new output directory");
    if (fs::exists(out / "STOP"))
        throw std::runtime_error("Remove the live run's STOP file before continuing");
    State s = source.empty() ? State{} : header(source);
    std::unique_ptr<LiveCurriculum> curriculum, previous_curriculum;
    if (!args.get("curriculum").empty()) {
        if (!args.get("data").empty())
            throw std::runtime_error("Curriculum specifies its data; omit --data");
        curriculum = std::make_unique<LiveCurriculum>(args.get("curriculum"));
    }
    if (resume && s.meta[17] == 4 && !curriculum)
        throw std::runtime_error("Curriculum checkpoint resume requires --curriculum");
    if (!args.get("extend-curriculum").empty()) {
        require(resume && s.meta[17] == 4 && curriculum,
                "--extend-curriculum requires --resume from a curriculum checkpoint and its original "
                "--curriculum");
        previous_curriculum = std::move(curriculum);
        curriculum = std::make_unique<LiveCurriculum>(args.get("extend-curriculum"));
        curriculum->validate_extension(*previous_curriculum);
    }
    Config q = source.empty() ? Config{args.num("channels", 256), args.num("hidden", 512),
                                       args.num("layers", 4), cell_version(args)}
                              : checkpoint_config(s);
    if (!source.empty())
        for (auto k : {"channels", "hidden", "layers", "cell"})
            if (!args.get(k).empty())
                throw std::runtime_error(std::string("Checkpoint preserves ") + k);
    if (resume) {
        if ((s.meta[17] < 1 || s.meta[17] > 4) || s.meta[5] != 1)
            throw std::runtime_error(
                "--resume needs a live checkpoint; use --checkpoint to begin a new stream");
        for (auto k : {"chunk", "seed", "activity-cost", "fast", "speak-every", "tokens", "top-k",
                       "temperature", "replay", "replay-capacity", "replay-seed", "core-scale", "graph",
                       "consolidation", "si-damping"})
            if (!args.get(k).empty())
                throw std::runtime_error(std::string("Live resume preserves ") + k);
    }
    int chunk = resume ? int(s.meta[6]) : args.num("chunk", 128);
    int updates = args.num("updates", curriculum ? int(curriculum->stages.back().end_update) : 1000),
        log_every = args.num("log-every", 100), save_every = args.num("save-every", 500);
    int eval_batches = args.num("eval-batches", 16);
    if (chunk < 1 || chunk > 4096 || updates < 1 || updates > 100000000 || log_every < 1 || save_every < 1 ||
        eval_batches < 1)
        throw std::runtime_error("Invalid live run limits");
    if (curriculum && uint64_t(updates) > curriculum->stages.back().end_update)
        throw std::runtime_error("Requested updates exceed the curriculum schedule");
    std::string prompt = args.get("prompt", "The bird ");
    if (prompt.empty() || prompt.size() > 65536)
        throw std::runtime_error("Live prompt must contain 1..65536 bytes");
    bool fast = resume ? s.meta[16] != 0 : (!args.get("fast").empty() || s.meta[16] != 0);
    LiveEngine engine(q, chunk, fast);
    if (!source.empty())
        load(source, engine.root, s, resume);
    else {
        int seed = args.num("seed", 1337);
        if (seed < 1)
            throw std::runtime_error("Seed must be positive");
        s.meta[10] = s.meta[11] = seed;
        engine.root.w.put(initialize(q, engine.root.a, uint64_t(seed)));
    }
    if (previous_curriculum)
        curriculum->extend(s, *previous_curriculum);
    LiveCorpus data = curriculum ? (resume && s.meta[17] == 4 ? curriculum->corpus(s)
                                                              : LiveCorpus(curriculum->stages.front().corpus))
                                 : LiveCorpus(args.get("data", "data/prepared/foundations-v1/train.dat"));
    if (resume) {
        data.validate(s);
        ReplayMemory(s).validate(data, chunk);
        if (curriculum && s.meta[17] != 4)
            curriculum->initialize(s); // Explicitly bind an existing stream without losing history.
        // Explicit stage changes preserve all learned history, moments, state
        // and RNG. Omitted controls retain their checkpoint values.
        if (curriculum) {
            curriculum->validate(s);
            curriculum->rate(s, args.real("lr", s.hp[7]));
        } else
            s.hp[0] = args.real("lr", s.hp[0]);
        if (!args.get("si-strength").empty()) {
            if (!engine.root.synapses->active())
                throw std::runtime_error("Cannot create missing SI history on resume");
            float strength = args.real("si-strength", 0);
            if (strength < 0 || strength > 1000000)
                throw std::runtime_error("Invalid --si-strength");
            engine.root.synapses->strength = strength;
            s.extra[10] = float_word(strength);
        }
        if (!args.get("replay-every").empty()) {
            int every = args.num("replay-every", 0);
            if (!ReplayMemory(s).mode() || every < 1 || every > 1000000000)
                throw std::runtime_error("Replay cadence override needs an existing replay policy");
            s.extra[2] = every;
        }
        if (s.meta[29] != hash_bytes(prompt.data(), prompt.size()))
            throw std::runtime_error("Live resume requires the same --prompt");
        if (s.meta[23] == 0 || s.meta[27] < 1 || s.meta[27] > 10000 || s.meta[28] < 1 || s.meta[28] > 256 ||
            !std::isfinite(s.hp[0]) || s.hp[0] <= 0 || s.hp[0] > .1f || !std::isfinite(s.hp[5]) ||
            s.hp[5] <= 0)
            throw std::runtime_error("Invalid live checkpoint settings");
    } else {
        float inherited_rate = s.hp[0];
        configure_live(s, args, data, prompt);
        if (curriculum) {
            s.hp[0] = args.real("lr", source.empty() ? .0003f : inherited_rate);
            curriculum->initialize(s);
            curriculum->validate(s);
        }
        if (has_synaptic_history(s))
            engine.root.synapses->initialize(engine.root.w, word_float(s.extra[10]), word_float(s.extra[11]));
    }
    if (curriculum)
        curriculum->apply_feedback(data, size_t(s.extra[15]));
    s.meta[16] = fast;
    data.validate(s);
    ReplayMemory memory(s);
    memory.validate(data, chunk);
    uint64_t extra_updates =
        memory.every() ? uint64_t(updates) / memory.every() - s.meta[24] / memory.every() : 0;
    if (uint64_t(updates) <= s.meta[24] ||
        uint64_t(updates) - s.meta[24] + extra_updates + s.meta[7] > 1000000000ull)
        throw std::runtime_error(
            "--updates must exceed the saved online update count and fit the optimizer limit");
    std::unique_ptr<Model> evaluator;
    std::unique_ptr<Data> validation;
    float initial_val = 0, final_val = 0;
    if (!args.get("validation").empty()) {
        LiveCorpus holdout(args.get("validation"));
        if (curriculum) {
            // The last cumulative edition contains all current and future data.
            LiveCorpus all_stages(curriculum->stages.back().corpus);
            validate_live_holdout(all_stages, holdout);
        } else
            validate_live_holdout(data, holdout);
        validation = std::make_unique<Data>(args.get("validation"), 128);
        evaluator = std::make_unique<Model>(q, 16, 128);
        evaluator->w.share(engine.root.w); // Isolated held-out recurrent state.
        initial_val = evaluate(*evaluator, *validation, eval_batches);
        std::cout << "initial_validation_loss=" << initial_val << "\n";
    }
    fs::create_directories(out);
    if (!resume)
        save(out / "initial.ckpt", engine.root, s);
    std::ofstream metrics(out / "metrics.jsonl", std::ios::app);
    std::ofstream transcript(out / "transcript.txt", std::ios::app | std::ios::binary);
    if (!metrics || !transcript)
        throw std::runtime_error("Cannot write live logs");
    if (previous_curriculum)
        metrics << "{\"event\":\"curriculum_extension\",\"online_update\":" << s.meta[24]
                << ",\"global_update\":" << s.meta[7] << ",\"previous_curriculum_hash\":\""
                << previous_curriculum->hash << "\",\"curriculum_hash\":\"" << curriculum->hash
                << "\",\"previous_stages\":" << previous_curriculum->stages.size()
                << ",\"stages\":" << curriculum->stages.size()
                << ",\"previous_end_update\":" << previous_curriculum->stages.back().end_update
                << ",\"end_update\":" << curriculum->stages.back().end_update
                << ",\"current_stage\":" << s.extra[15] + 1 << ",\"document\":" << s.meta[19]
                << ",\"byte_offset\":" << s.meta[20] << ",\"replay_windows_preserved\":" << memory.count()
                << ",\"replay_updates\":" << memory.updates() << ",\"generated_bytes\":" << s.meta[30]
                << ",\"consolidation_events\":" << engine.root.synapses->boundaries << "}\n";
    metrics << std::setprecision(10) << "{\"event\":\"session_start\",\"online_update\":" << s.meta[24]
            << ",\"learning_rate\":" << s.hp[0] << ",\"replay_every\":" << memory.every()
            << ",\"si_strength\":" << engine.root.synapses->strength << ",\"curriculum_hash\":\""
            << (curriculum ? curriculum->hash : 0)
            << "\",\"curriculum_stage\":" << (curriculum ? s.extra[15] + 1 : 0)
            << ",\"curriculum_base_lr\":" << (curriculum ? s.hp[7] : 0) << ",\"corpus_hash\":\"" << data.hash
            << "\",\"online_first_document\":" << data.first_document
            << ",\"resume\":" << (resume ? "true" : "false") << "}\n";
    std::cout << "live parameters=" << engine.root.a.n
              << " shared_weights=yes persistent_membranes=yes learning=observed_bytes_only"
              << " online_update=" << s.meta[24] << " global_update=" << s.meta[7] << "\n";
    if (engine.root.synapses->active())
        std::cout << "consolidation=si boundary=document_or_curriculum strength="
                  << engine.root.synapses->strength << " damping=" << engine.root.synapses->damping << "\n";
    transcript << "\n[session starts at online update " << s.meta[24] << "]\n";
    std::signal(SIGINT, interrupt_handler);
    double loss_sum = 0, activity_sum = 0, replay_loss_sum = 0;
    size_t observed = 0, replayed = 0;
    uint64_t start_observed = s.meta[22], start_updates = s.meta[24];
    LiveLatency tick_ms, speak_ms;
    auto started = std::chrono::steady_clock::now();
    for (; s.meta[24] < uint64_t(updates);) {
        auto tick_start = std::chrono::steady_clock::now();
        if (curriculum && curriculum->advance(engine, data, s, &metrics)) {
            metrics.flush();
            std::cout << "curriculum_stage=" << s.extra[15] + 1 << " new_document=" << s.meta[19]
                      << " replay_windows_preserved=" << memory.count() << '\n';
        }
        auto result = live_tick(engine, data, s, prompt);
        ck(cudaDeviceSynchronize());
        double duration =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tick_start).count();
        (result.speech.empty() ? tick_ms : speak_ms).add(duration);
        loss_sum += result.loss * result.observed;
        activity_sum += result.activity * result.observed;
        observed += result.observed;
        replay_loss_sum += result.replay_loss * result.replayed;
        replayed += result.replayed;
        if (!result.speech.empty()) {
            transcript << "\n[online update " << s.meta[24] << "; global update " << s.meta[7] << "]\n"
                       << result.speech << "\n";
            transcript.flush();
            std::cout << "spoke_bytes=" << s.meta[27] << " at_online_update=" << s.meta[24] << "\n";
        }
        bool stop = interrupted || fs::exists(out / "STOP"), last = s.meta[24] == uint64_t(updates);
        if (s.meta[24] % uint64_t(log_every) == 0 || stop || last) {
            double loss = loss_sum / observed, activity = activity_sum / observed;
            metrics << std::setprecision(10) << "{\"online_update\":" << s.meta[24]
                    << ",\"curriculum_stage\":" << (curriculum ? s.extra[15] + 1 : 0)
                    << ",\"global_update\":" << s.meta[7] << ",\"observed_pairs\":" << s.meta[22]
                    << ",\"epochs\":" << s.meta[21] << ",\"document\":" << s.meta[19]
                    << ",\"byte_offset\":" << s.meta[20] << ",\"loss\":" << loss
                    << ",\"spike_rate\":" << activity << ",\"gradient_norm\":" << result.gradient_norm
                    << ",\"replay_updates\":" << memory.updates() << ",\"replay_pairs\":" << memory.pairs()
                    << ",\"replay_loss\":" << (replayed ? std::to_string(replay_loss_sum / replayed) : "null")
                    << ",\"consolidation_events\":" << engine.root.synapses->boundaries << "}\n";
            metrics.flush();
            std::cout << "online_update=" << s.meta[24] << " loss=" << loss << " spike_rate=" << activity
                      << " observed_pairs=" << s.meta[22] << "\n";
            loss_sum = activity_sum = 0;
            observed = 0;
            replayed = 0;
            replay_loss_sum = 0;
        }
        if (s.meta[24] % uint64_t(save_every) == 0 || stop || last)
            save(latest, engine.root, s);
        if (curriculum && s.meta[24] == curriculum->stages[size_t(s.extra[15])].end_update)
            save(out / ("stage-" + std::to_string(s.extra[15] + 1) + ".ckpt"), engine.root, s);
        if (stop)
            break;
    }
    double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    if (evaluator)
        final_val = evaluate(*evaluator, *validation, eval_batches);
    std::ofstream report(out / "session.json");
    report << std::setprecision(10) << "{\"online_updates\":" << s.meta[24]
           << ",\"curriculum_stage\":" << (curriculum ? s.extra[15] + 1 : 0) << ",\"curriculum_hash\":\""
           << (curriculum ? curriculum->hash : 0) << "\""
           << ",\"curriculum_extended\":" << (previous_curriculum ? "true" : "false")
           << ",\"curriculum_base_lr\":" << (curriculum ? s.hp[7] : 0) << ",\"global_updates\":" << s.meta[7]
           << ",\"online_first_document\":" << data.first_document
           << ",\"online_document_count\":" << data.docs.size() - data.first_document
           << ",\"answer_emphasized_documents\":" << data.feedback.size()
           << ",\"observed_pairs\":" << s.meta[22] << ",\"generated_bytes\":" << s.meta[30]
           << ",\"elapsed_seconds\":" << seconds
           << ",\"session_online_updates\":" << s.meta[24] - start_updates
           << ",\"session_observed_pairs\":" << s.meta[22] - start_observed
           << ",\"observed_pairs_per_second\":" << (s.meta[22] - start_observed) / seconds
           << ",\"replay_updates\":" << memory.updates() << ",\"replay_pairs\":" << memory.pairs()
           << ",\"replay_mode\":" << memory.mode() << ",\"replay_items\":" << memory.count()
           << ",\"core_scale\":" << memory.core_scale() << ",\"learning_rate\":" << s.hp[0]
           << ",\"replay_every\":" << memory.every()
           << ",\"graph_decode\":" << (memory.graph() ? "true" : "false")
           << ",\"update_tick_p50_ms\":" << tick_ms.percentile(.5)
           << ",\"update_tick_p95_ms\":" << tick_ms.percentile(.95)
           << ",\"update_and_speech_tick_p50_ms\":" << speak_ms.percentile(.5)
           << ",\"update_and_speech_tick_p95_ms\":" << speak_ms.percentile(.95)
           << ",\"latency_percentile_bin_relative_width\":" << std::exp2(1.0 / 32) - 1
           << ",\"shared_weights\":true,\"persistent_membranes\":true,\"trains_on_generated_text\":false";
    if (engine.root.synapses->active()) {
        auto importance = engine.root.synapses->importance.host();
        double sum = std::accumulate(importance.begin(), importance.end(), 0.0);
        auto nonzero = std::count_if(importance.begin(), importance.end(), [](float x) { return x > 0; });
        report << ",\"consolidation\":\"si\",\"consolidation_events\":" << engine.root.synapses->boundaries
               << ",\"si_strength\":" << engine.root.synapses->strength
               << ",\"si_damping\":" << engine.root.synapses->damping
               << ",\"synaptic_state_bytes\":" << 12 * engine.root.a.n
               << ",\"synaptic_scratch_bytes\":" << 4 * engine.root.a.n
               << ",\"importance_mean\":" << sum / importance.size()
               << ",\"importance_max\":" << *std::max_element(importance.begin(), importance.end())
               << ",\"importance_positive_fraction\":" << double(nonzero) / importance.size();
    } else
        report << ",\"consolidation\":\"none\"";
    if (evaluator)
        report << ",\"initial_validation_loss\":" << initial_val
               << ",\"final_validation_loss\":" << final_val;
    report << "}\n";
    std::cout << "Live state saved: " << latest.string();
    if (evaluator)
        std::cout << " validation_loss=" << initial_val << " -> " << final_val;
    std::cout << "\n";
}

void live_test(const Args &args) {
    args.allow({"out"});
    fs::path out = args.get("out", "reports/live-tests");
    fs::create_directories(out);
    Config q{32, 64, 2};
    Model gradient(q, 2, 8);
    auto weights = initialize(q, gradient.a, 123);
    gradient.w.put(weights);
    std::vector<float> initial(size_t(q.l) * 2 * q.h);
    for (size_t i = 0; i < initial.size(); ++i)
        initial[i] = 1.7f * std::sin(float(i) * .39f);
    gradient.membranes(initial);
    std::vector<int> x(16), y(16);
    for (int i = 0; i < 16; ++i) {
        x[i] = (i * 37 + 91) % 256;
        y[i] = (i * 19 + 13) % 256;
    }
    float loss = gradient.forward(x, &y, true);
    dump(out / "weights.f32", weights);
    dump(out / "initial_state.f32", initial);
    dump(out / "final_state.f32", gradient.membranes());
    dump(out / "logits.f32", gradient.logits.host());
    dump(out / "inputs.i32", x);
    dump(out / "targets.i32", y);
    gradient.backward();
    dump(out / "gradients.f32", gradient.g.host());
    gradient.backward(1.f);
    dump(out / "gradients_regularized.f32", gradient.g.host());
    gradient.backward();
    gradient.update(1, .001f, 0, 0);
    dump(out / "updated.f32", gradient.w.host());
    std::ofstream conf(out / "fixture.json");
    conf << "{\"channels\":32,\"hidden\":64,\"layers\":2,\"batch\":2,\"context\":8,\"loss\":"
         << std::setprecision(10) << loss << ",\"parameters\":" << gradient.a.n << "}";
    conf.close();
    Model chunk(q, 1, 4), step(q, 1, 1);
    chunk.w.put(weights);
    step.w.put(weights);
    std::vector<float> start(q.l * q.h);
    for (size_t i = 0; i < start.size(); ++i)
        start[i] = initial[i];
    chunk.membranes(start);
    step.membranes(start);
    float chunk_error = 0;
    for (int i = 0; i < 8; i += 4) {
        chunk.forward(std::vector<int>(x.begin() + i, x.begin() + i + 4), nullptr, true);
        auto logits = chunk.logits.host();
        for (int j = 0; j < 4; ++j) {
            step.forward({x[i + j]}, nullptr, true);
            chunk_error = std::max(
                chunk_error, maxdiff(step.logits.host(), std::vector<float>(logits.begin() + j * 256,
                                                                            logits.begin() + (j + 1) * 256)));
        }
        chunk_error = std::max(chunk_error, maxdiff(step.membranes(), chunk.membranes()));
    }
    require(chunk_error < 3e-5, "Persistent chunks differ from token streaming");
    std::string text = "A girl counts the seeds.\x1e"
                       "The child sees the bird.";
    dump(out / "corpus.dat", std::vector<char>(text.begin(), text.end()));
    LiveCorpus data(out / "corpus.dat");
    State s;
    configure_live(s, args, data, "A");
    s.meta[26] = 2;
    s.meta[27] = 7;
    s.hp[0] = .001f;
    LiveEngine first(q, 8, false);
    first.root.w.put(weights);
    require(first.speaker.w.p == first.root.w.p && first.speaker.m.p == first.root.m.p &&
                first.speaker.v.p == first.root.v.p,
            "Live views do not share parameters and optimizer");
    for (int l = 0; l < q.l; ++l)
        require(first.speaker.cache[l].state.p == first.root.cache[l].state.p,
                "Live views do not share membranes");
    for (int i = 0; i < 4; ++i)
        live_tick(first, data, s, "A");
    save(out / "resume.ckpt", first.root, s);
    LiveEngine second(q, 8, false);
    State restored;
    load(out / "resume.ckpt", second.root, restored, true);
    data.validate(restored);
    require(s.meta == restored.meta && s.hp == restored.hp, "Live metadata did not roundtrip");
    require(maxdiff(first.root.membranes(), second.root.membranes()) == 0,
            "Live membranes did not roundtrip");
    std::string uninterrupted, resumed;
    for (int i = 0; i < 8; ++i) {
        uninterrupted += live_tick(first, data, s, "A").speech;
        resumed += live_tick(second, data, restored, "A").speech;
    }
    require(uninterrupted == resumed, "Live resumed generation differs");
    require(s.meta == restored.meta, "Live cursor or RNG differs after resume");
    float resume_error = maxdiff(first.root.w.host(), second.root.w.host());
    resume_error = std::max(resume_error, maxdiff(first.root.m.host(), second.root.m.host()));
    resume_error = std::max(resume_error, maxdiff(first.root.v.host(), second.root.v.host()));
    resume_error = std::max(resume_error, maxdiff(first.root.membranes(), second.root.membranes()));
    require(resume_error < 3e-5, "Live resumed state or optimizer differs");
    require(s.meta[21] == 2 && s.meta[22] == 92,
            "Live stream skipped the document tail or crossed a separator");
    auto before = first.root.w.host(), before_m = first.root.m.host(), before_v = first.root.v.host();
    first.speak("A", s);
    require(maxdiff(before, first.root.w.host()) == 0 && maxdiff(before_m, first.root.m.host()) == 0 &&
                maxdiff(before_v, first.root.v.host()) == 0,
            "Generated text silently trained the weights");
    auto check_rejection = [&](const fs::path &dest, std::streamoff at) {
        fs::copy_file(out / "resume.ckpt", dest, fs::copy_options::overwrite_existing);
        std::fstream f(dest, std::ios::in | std::ios::out | std::ios::binary);
        f.seekg(at);
        char c;
        f.read(&c, 1);
        c ^= 1;
        f.seekp(at);
        f.write(&c, 1);
        f.close();
        bool rejected = false;
        try {
            load(dest, second.root, restored, true);
        } catch (const std::exception &) {
            rejected = true;
        }
        require(rejected, "Corrupted live state or cursor accepted");
    };
    check_rejection(out / "corrupt-membrane.ckpt", std::streamoff(fs::file_size(out / "resume.ckpt") - 1));
    check_rejection(out / "corrupt-cursor.ckpt", 20 * 8);
    std::ofstream report(out / "native.json");
    report << "{\"passed\":true,\"persistent_chunk_max_error\":" << chunk_error
           << ",\"live_resume_max_error\":" << resume_error
           << ",\"resumed_generation_identical\":true,\"shared_allocations\":true,\"speech_does_not_train\":"
              "true,\"document_tails_verified\":true}\n";
    std::cout
        << "PASS live tests: shared allocations, persistent chunk equivalence, complete document coverage, "
           "state/optimizer/RNG resume, identical resumed speech, state/cursor corruption rejection.\n";
}

void replay_test(const Args &args) {
    args.allow({"out"});
    fs::path out = args.get("out", "reports/replay-tests");
    fs::create_directories(out);
    Config q{32, 64, 2};
    std::string text = "A girl counts the seeds.\x1eThe child sees the bird.";
    dump(out / "corpus.dat", std::vector<char>(text.begin(), text.end()));
    LiveCorpus data(out / "corpus.dat");
    Args options = args;
    options.values["--replay"] = "reservoir";
    options.values["--replay-capacity"] = "3";
    options.values["--replay-every"] = "2";
    options.values["--core-scale"] = "0.25";
    options.values["--graph"] = "1";
    State s;
    configure_live(s, options, data, "A");
    s.meta[26] = 3;
    s.meta[27] = 7;
    s.hp[0] = .001f;
    LiveEngine first(q, 8, false), second(q, 8, false);
    auto weights = initialize(q, first.root.a, 123);
    first.root.w.put(weights);
    for (int i = 0; i < 5; ++i)
        live_tick(first, data, s, "A");
    save(out / "resume.ckpt", first.root, s);
    State restored;
    load(out / "resume.ckpt", second.root, restored, true);
    ReplayMemory(s).validate(data, 8);
    ReplayMemory(restored).validate(data, 8);
    require(s.extra == restored.extra, "Replay descriptors/RNG/settings did not roundtrip");
    float error = 0;
    uint64_t replayed = 0;
    for (int i = 0; i < 19; ++i) {
        auto a = live_tick(first, data, s, "A");
        auto b = live_tick(second, data, restored, "A");
        require(a.speech == b.speech && a.replayed == b.replayed,
                "Replay resumed schedule or speech differs");
        require(s.extra == restored.extra, "Replay memory/RNG differs after resume");
        error = std::max(error, std::abs(a.loss - b.loss));
        error = std::max(error, std::abs(a.replay_loss - b.replay_loss));
        replayed += a.replayed;
    }
    error = std::max(error, maxdiff(first.root.w.host(), second.root.w.host()));
    error = std::max(error, maxdiff(first.root.m.host(), second.root.m.host()));
    error = std::max(error, maxdiff(first.root.v.host(), second.root.v.host()));
    error = std::max(error, maxdiff(first.root.membranes(), second.root.membranes()));
    require(error < 3e-5f, "Replay resumed optimizer/state/loss differs");
    require(s.meta == restored.meta && s.meta[22] == 184 && s.meta[7] == 36 &&
                ReplayMemory(s).updates() == 12 && ReplayMemory(s).count() == 3 && replayed > 0,
            "Replay counters, bounded memory or document tails failed");

    // Replay changes shared weights, but cannot perturb live membranes.
    auto live_state = first.root.membranes();
    auto &view = first.replay_view(8);
    std::vector<int> x{'A', ' ', 'k', 'i', 'n', 'g', ' ', 'c'}, y{' ', 'k', 'i', 'n', 'g', ' ', 'c', 'o'};
    view.forward(x, &y);
    view.backward();
    view.update(int(++s.meta[7]), .001f);
    require(maxdiff(live_state, first.root.membranes()) == 0, "Replay contaminated live recurrence");
    require(view.w.p == first.root.w.p && view.cache[0].state.p != first.root.cache[0].state.p,
            "Replay state/weight ownership is wrong");

    // Independent parameter-group update check, including decoupled weight decay.
    Model full(q, 1, 8), scaled(q, 1, 8), frozen(q, 1, 8);
    for (Model *m : {&full, &scaled, &frozen})
        m->w.put(weights);
    full.forward(x, &y);
    full.backward();
    // Isolate the update rule from nondeterministic atomic gradient reduction.
    auto gradient = full.g.host();
    scaled.g.put(gradient);
    frozen.g.put(gradient);
    full.update(1, .001f, .01f, 1.f, 1.f);
    scaled.update(1, .001f, .01f, 1.f, .25f);
    frozen.update(1, .001f, .01f, 1.f, 0.f);
    auto fw = full.w.host(), sw = scaled.w.host(), zw = frozen.w.host();
    float scale_error = 0, head_change = 0;
    for (size_t i = 0; i < weights.size(); ++i) {
        float expected = i < full.a.final_gain ? weights[i] + .25f * (fw[i] - weights[i]) : fw[i];
        scale_error = std::max(scale_error, std::abs(sw[i] - expected));
        require(i < full.a.final_gain ? zw[i] == weights[i] : zw[i] == fw[i],
                "Frozen core changed or readout froze");
        if (i >= full.a.final_gain)
            head_change = std::max(head_change, std::abs(zw[i] - weights[i]));
    }
    require(scale_error < 5e-7f && head_change > .0001f, "Core/readout rate scaling failed");
    require(maxdiff(full.m.host(), scaled.m.host()) == 0 && maxdiff(full.v.host(), scaled.v.host()) == 0,
            "Rate scaling unexpectedly changed moments");

    // Legacy live v1 is still readable and uses the original update rate.
    State legacy;
    configure_live(legacy, args, data, "A");
    legacy.meta[17] = 1;
    legacy.extra.clear();
    legacy.meta[31] = 0;
    LiveEngine old(q, 8, false);
    old.root.w.put(weights);
    live_tick(old, data, legacy, "A");
    save(out / "legacy.ckpt", old.root, legacy);
    load(out / "legacy.ckpt", second.root, restored, true);
    ReplayMemory(restored).validate(data, 8);
    require(ReplayMemory(restored).core_scale() == 1.f, "Legacy core rate changed");

    auto bytes = fs::file_size(out / "resume.ckpt");
    fs::copy_file(out / "resume.ckpt", out / "corrupt.ckpt", fs::copy_options::overwrite_existing);
    {
        std::fstream f(out / "corrupt.ckpt", std::ios::in | std::ios::out | std::ios::binary);
        f.seekg(std::streamoff(bytes - 1));
        char byte;
        f.read(&byte, 1);
        byte ^= 1;
        f.seekp(std::streamoff(bytes - 1));
        f.write(&byte, 1);
    }
    bool rejected = false;
    try {
        load(out / "corrupt.ckpt", second.root, restored, true);
    } catch (const std::exception &) {
        rejected = true;
    }
    require(rejected, "Corrupt replay payload accepted");
    load(out / "resume.ckpt", second.root, restored, true);
    restored.extra[16] = data.docs.size();
    rejected = false;
    try {
        ReplayMemory(restored).validate(data, 8);
    } catch (const std::exception &) {
        rejected = true;
    }
    require(rejected, "Replay descriptor outside selected corpus accepted");
    std::ofstream f(out / "native.json");
    f << "{\"passed\":true,\"resume_max_error\":" << error << ",\"core_scale_max_error\":" << scale_error
      << ",\"resumed_speech_identical\":true,\"bounded_reservoir\":true,\"isolated_replay_membranes\":true,"
         "\"legacy_live_resume\":true,\"corrupt_replay_rejected\":true,\"invalid_descriptor_rejected\":true}"
         "\n";
    std::cout << "PASS replay: bounded memory, exact schedule/RNG/speech resume, separate recurrence, core "
                 "scaling, legacy and corruption checks. Error "
              << error << "\n";
}
