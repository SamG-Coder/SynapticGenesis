#pragma once
namespace evolution {
struct TeacherAuthority final : teachers::Authority {
    fs::path root;
    uint64_t tick;
    TeacherAuthority(fs::path directory, uint64_t current) : root(std::move(directory)), tick(current) {}
    void validate(const teachers::Bundle &bundle, bool attaching, bool active) const override {
        require(bundle.identity.words[14] == 1, "Population learning requires registered teacher origins");
        for (size_t i = 0; i < bundle.members.size(); ++i) {
            const auto &id = bundle.members[i];
            require(valid_id(id), "Invalid teacher member ID");
            auto directory = root / id;
            auto member = read_member(directory / "member.sg");
            size_t at = bundle.identity.entry(i);
            require(member.id == id && teachers::file_hash(directory / "member.sg") == bundle.identity.words[at + 7],
                    "Teacher population identity changed");
            if (active || attaching)
                require(alive(member, tick), "Teacher died of old age; disable teaching to continue source learning");
            if (attaching)
                require(teachers::file_hash(directory / "latest.ckpt") == bundle.identity.words[at],
                        "New teacher attachment requires the current registered checkpoint");
        }
    }
};
} // namespace evolution

void teacher_pack_command(const Args &args) {
    args.allow({"teacher-a", "teacher-b", "population", "data", "out", "temperature", "strength", "mixture"});
    require(!args.get("teacher-a").empty() && !args.get("data").empty() && !args.get("out").empty(),
            "teacher-pack requires --teacher-a, selected --data and --out");
    std::vector<teachers::Origin> origins;
    std::unique_ptr<evolution::PopulationLock> lock;
    bool population = !args.get("population").empty();
    fs::path root = args.get("population");
    evolution::Rules rules;
    if (population) {
        require(fs::exists(root / "population.sg"), "Unknown teacher population");
        lock = std::make_unique<evolution::PopulationLock>(root);
        rules = evolution::read_rules(root / "population.sg");
    }
    for (const auto *key : {"teacher-a", "teacher-b"}) {
        auto selected = args.get(key);
        if (selected.empty())
            continue;
        if (population) {
            require(evolution::valid_id(selected), "Invalid teacher member ID");
            auto directory = root / selected;
            auto member = evolution::read_member(directory / "member.sg");
            require(member.id == selected && evolution::alive(member, rules.tick),
                    "Teacher died of old age or its member identity differs");
            origins.push_back({directory / "latest.ckpt", selected, teachers::file_hash(directory / "member.sg")});
        } else
            origins.push_back({selected, "", 0});
    }
    teachers::create_bundle(args.get("out"), args.get("data"), origins, args.real("temperature", 2.f),
                            args.real("strength", .5f), args.real("mixture", .5f), population);
}
