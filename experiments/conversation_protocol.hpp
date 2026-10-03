// Host-only command protocol. No CUDA headers or model implementation here.
#pragma once
#include <cctype>
#include <cstdint>
#include <iomanip>
#include <istream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace conversation {
inline void ensure(bool condition, const char *message) {
    if (!condition)
        throw std::runtime_error(message);
}
struct Command {
    std::string kind, label, question, source;
    uint64_t endpoint = 0;
};
inline bool line(std::istream &input, std::string &value, size_t limit = 16384) {
    value.clear();
    char c;
    while (input.get(c)) {
        if (c == '\n')
            return true;
        ensure(value.size() < limit, "Conversation line exceeds its byte limit");
        value += c;
    }
    ensure(!input.bad(), "Conversation input failed");
    return !value.empty();
}
inline std::string label_key(const std::string &label) {
    ensure(!label.empty() && label.size() <= 64, "Conversation label must contain 1..64 characters");
    std::string result;
    for (unsigned char c : label) {
        ensure((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') ||
                   c == '_' || c == '-', "Conversation labels allow only ASCII letters, digits, '_' and '-'");
        result += char(std::tolower(c));
    }
    return result;
}
inline Command parse(const std::string &line) {
    ensure(line.size() <= 16384 && line.find('\0') == std::string::npos,
           "Conversation command exceeds the byte limit or contains NUL");
    Command command;
    command.source = line;
    std::istringstream row(line);
    row >> command.kind;
    if (command.kind.empty() || command.kind[0] == '#') {
        command.kind.clear();
        return command;
    }
    if (command.kind == "ask") {
        row >> std::quoted(command.label) >> std::quoted(command.question);
        ensure(bool(row) && command.question.size() <= 8192 &&
                   command.question.find_first_not_of(" \t\r\n") != std::string::npos &&
                   command.question.find_first_of("\r\n\x1e") == std::string::npos,
               "ask needs a label and one nonempty single-line question");
        label_key(command.label);
    } else if (command.kind == "learn") {
        std::string count;
        row >> count;
        ensure(!count.empty() && count.size() <= 9 &&
                   count.find_first_not_of("0123456789") == std::string::npos,
               "learn needs a positive cumulative update count");
        command.endpoint = std::stoull(count);
        ensure(command.endpoint > 0 && command.endpoint <= 100000000, "Learning endpoint is out of range");
    } else if (command.kind == "save") {
        row >> std::quoted(command.label);
        ensure(bool(row), "save needs a label");
        label_key(command.label);
    } else
        ensure(command.kind == "quit", "Unknown conversation command");
    std::string excess;
    ensure(!(row >> excess), "Unexpected trailing command fields");
    return command;
}
struct Sequence {
    uint64_t endpoint, limit;
    size_t count = 0, saves = 0;
    bool finished = false;
    std::set<std::string> answers, snapshots;
    Sequence(uint64_t start, uint64_t end) : endpoint(start), limit(end) {
        ensure(start <= end && end <= 100000000, "Invalid conversation learning range");
    }
    void accept(const Command &command) {
        if (command.kind.empty())
            return;
        ensure(!finished && count < 4096, "Conversation is finished or exceeds 4096 commands");
        if (command.kind == "learn") {
            ensure(command.endpoint > endpoint && command.endpoint <= limit,
                   "Learning must advance within the admitted curriculum");
            endpoint = command.endpoint;
        } else if (command.kind == "ask") {
            ensure(answers.insert(label_key(command.label)).second, "Duplicate answer label");
        } else if (command.kind == "save") {
            ensure(saves < 64, "Conversation exceeds 64 explicit snapshots");
            ensure(snapshots.insert(label_key(command.label)).second, "Duplicate snapshot label");
            ++saves;
        } else if (command.kind == "quit")
            finished = true;
        ++count;
    }
};
inline void read_magic(std::istream &input) {
    std::string magic;
    line(input, magic, 32);
    if (!magic.empty() && magic.back() == '\r')
        magic.pop_back();
    ensure(magic == "SGCONVERSATION1", "Expected SGCONVERSATION1 header");
}
inline std::vector<Command> read(std::istream &input, uint64_t start, uint64_t end) {
    read_magic(input);
    Sequence sequence(start, end);
    std::vector<Command> commands;
    std::string source;
    size_t bytes = 0;
    while (line(input, source)) {
        bytes += source.size() + 1;
        ensure(bytes <= 1024 * 1024, "Conversation script exceeds one MiB");
        auto command = parse(source);
        sequence.accept(command);
        if (!command.kind.empty())
            commands.push_back(std::move(command));
    }
    ensure(!input.bad() && !commands.empty(), "Empty or unreadable conversation script");
    return commands;
}
inline std::string prompt(const std::string &question) {
    return "Question: " + question + "\nAnswer: ";
}
} // namespace conversation
