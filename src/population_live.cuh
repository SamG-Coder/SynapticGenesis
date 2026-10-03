#pragma once
#include "evolution.cuh"
#include "live.cuh"
#include "population_teachers.cuh"

namespace evolution {
void live(const Args &args) {
    for (auto key : {"out", "checkpoint", "resume", "channels", "hidden", "layers", "cell"})
        if (!args.get(key).empty())
            throw std::runtime_error(std::string("Population owns ") + key +
                                     "; specify --population and --id");
    std::string id = args.get("id");
    require(valid_id(id), "Invalid population member ID");
    fs::path root = args.get("population", "runs/population");
    PopulationLock lock(root);
    Rules rules = read_rules(root / "population.sg");
    fs::path directory = root / id, checkpoint = directory / "latest.ckpt", output = directory / "live";
    Member member = read_member(directory / "member.sg");
    require(member.id == id, "Member ID does not match its directory");
    require(alive(member, rules.tick),
            "Member died of old age; its archived checkpoint cannot start a population learning session");
    require(!args.get("validation").empty(),
            "Population live learning requires --validation with the registered evaluation corpus");
    require(!args.get("data").empty() || !args.get("curriculum").empty(),
            "Population live learning requires selected --data or --curriculum");
    LiveCorpus heldout(args.get("validation"));
    require(heldout.hash == rules.corpus, "Population evaluation corpus changed");
    State before = header(checkpoint);
    Args options = args;
    options.values.erase("--population");
    options.values.erase("--id");
    options.values["--out"] = output.string();
    options.values[before.meta[17] ? "--resume" : "--checkpoint"] = checkpoint.string();
    // A new child's inherited/mutated rate must not silently become the generic
    // single-corpus live default. Resumed curricula already keep their base LR.
    if (!before.meta[17] && options.get("lr").empty()) {
        std::ostringstream inherited_rate;
        inherited_rate << std::setprecision(std::numeric_limits<float>::max_digits10) << before.hp[0];
        options.values["--lr"] = inherited_rate.str();
    }
    std::cout << "population_member=" << id << " generation=" << member.generation
              << " simulation_age=" << rules.tick - member.born_tick << '/' << member.lifespan
              << " resume=" << (before.meta[17] ? "yes" : "no") << '\n';
    TeacherAuthority teacher_authority(root, rules.tick);
    live_command(options, checkpoint, &teacher_authority);
    State after = header(checkpoint);
    // Population identity and lifespan remain in member.sg. Learning does not
    // reset the individual's age, create a new generation, or self-award fitness.
    std::ofstream journal(output / "population.jsonl", std::ios::app);
    journal << "{\"member\":" << std::quoted(id) << ",\"generation\":" << member.generation
            << ",\"population_tick\":" << rules.tick << ",\"born_tick\":" << member.born_tick
            << ",\"age\":" << rules.tick - member.born_tick << ",\"lifespan\":" << member.lifespan
            << ",\"parent_a\":" << std::quoted(member.parent_a)
            << ",\"parent_b\":" << std::quoted(member.parent_b) << ",\"starting_payload_hash\":\""
            << before.meta[15] << "\",\"saved_payload_hash\":\"" << after.meta[15]
            << "\",\"starting_online_updates\":" << before.meta[24]
            << ",\"saved_online_updates\":" << after.meta[24]
            << ",\"saved_optimizer_updates\":" << after.meta[7] << ",\"registered_validation_hash\":\""
            << rules.corpus << "\",\"automatic_fitness_promotion\":false}\n";
    journal.close();
    require(bool(journal), "Member checkpoint saved, but population learning journal could not be written");
}
} // namespace evolution
