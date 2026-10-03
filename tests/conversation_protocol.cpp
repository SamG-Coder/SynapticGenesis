// Pure C++ protocol checks; this target does not link to CUDA.
#include "../experiments/conversation_protocol.hpp"
#include <iostream>
#include <functional>

int main() {
    try {
        using namespace conversation;
        size_t rejected = 0;
        auto refuse = [&](const std::function<void()> &operation) {
            bool failed = false;
            try { operation(); } catch (const std::exception &) { failed = true; }
            ensure(failed, "Accepted a malformed host fixture");
            ++rejected;
        };
        auto session = [](const std::string &text) {
            std::istringstream input(text);
            return read(input, 11, 29);
        };
        auto commands = session("SGCONVERSATION1\r\n# comment\nask before \"What is water?\"\n"
                                "learn 17\nsave middle\nask after \"What is water?\"\nlearn 29\nquit\n");
        ensure(commands.size() == 6 && commands[0].question == "What is water?" &&
                   commands[1].endpoint == 17 && commands[4].endpoint == 29,
               "Valid conversation changed during parsing");
        ensure(prompt("Hello!") == "Question: Hello!\nAnswer: ", "Question framing changed");
        for (const auto &bad : std::vector<std::string>{
                 "ask a \"\"", "ask a \"   \"", "ask ../x \"Hello\"", "ask a \"unclosed",
                 "ask a \"Hello\" trailing", "save /outside", "save x trailing", "save", "quit now",
                 "learn -1", "learn 0", "learn 100000001", "learn 99999999999999999999",
                 "learn 17x", "learn 1 2", "erase everything"})
            refuse([&] { parse(bad); });
        refuse([&] { parse(std::string("ask a \"x") + '\0' + "y\""); });
        refuse([&] { parse("ask a \"" + std::string(8193, 'x') + "\""); });
        for (const auto &bad : std::vector<std::string>{
                 "BAD\nquit\n", "SGCONVERSATION1\n", "SGCONVERSATION1\nlearn 11\n",
                 "SGCONVERSATION1\nlearn 30\n", "SGCONVERSATION1\nlearn 17\nlearn 16\n",
                 "SGCONVERSATION1\nask A \"x\"\nask a \"y\"\n",
                 "SGCONVERSATION1\nsave A\nsave a\n", "SGCONVERSATION1\nquit\nlearn 12\n"})
            refuse([&] { session(bad); });
        refuse([&] { session(std::string(33, 'x')); });
        refuse([&] { session("SGCONVERSATION1\n" + std::string(16385, 'x')); });
        Sequence state(11, 29);
        state.accept(parse("learn 17"));
        refuse([&] { state.accept(parse("learn 16")); });
        ensure(state.endpoint == 17 && state.count == 1, "Rejected endpoint altered session state");
        state.accept(parse("ask a \"Hello!\""));
        refuse([&] { state.accept(parse("ask A \"Hello!\"")); });
        ensure(state.count == 2 && state.answers.size() == 1, "Rejected label altered session state");
        Sequence snapshots(11, 29);
        for (int i = 0; i < 64; ++i)
            snapshots.accept(parse("save n" + std::to_string(i)));
        refuse([&] { snapshots.accept(parse("save overflow")); });
        std::cout << "{\"passed\":true,\"rejections\":" << rejected
                  << ",\"native_model_commands\":0,\"cuda_linked\":false}\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "FAIL " << error.what() << '\n';
        return 1;
    }
}
