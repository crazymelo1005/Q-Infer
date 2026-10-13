// Protocol-level test double for the Strata engine. It speaks the line shapes recorded in
// tests/data/strata_transcript_gen.txt, so EngineDriver tests need no GPU and no weights.
// The scenario is argv[1]. Every line is flushed: stdout is a pipe, and full buffering would
// deadlock the driver waiting for a line that is still sitting in the mock's buffer.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

void emit(const std::string& s) {
    std::fputs(s.c_str(), stdout);
    std::fputc('\n', stdout);
    std::fflush(stdout);
}

std::vector<std::string> split_ws(const std::string& s) {
    std::vector<std::string> out;
    std::size_t i = 0;
    while (i < s.size()) {
        while (i < s.size() && (s[i] == ' ' || s[i] == '\t')) ++i;
        std::size_t j = i;
        while (j < s.size() && !(s[j] == ' ' || s[j] == '\t')) ++j;
        if (j > i) out.push_back(s.substr(i, j - i));
        i = j;
    }
    return out;
}

const int kScript[4] = {101, 102, 103, 104};

// 15 fields, matching the recorded transcript (generated prompt_tokens prompt_ms decode_ms
// finish drafts_accepted drafts_offered reused hits lookups ram_blobs file_blobs file_mb
// prompt_read offloaded).
void emit_done(int64_t generated, const char* finish) {
    char buf[160];
    std::snprintf(buf, sizeof(buf), "DONE %lld 3 1 1 %s 0 0 0 0 0 0 0 0 3 0",
                  static_cast<long long>(generated), finish);
    emit(buf);
}

void do_gen(int64_t max_new) {
    emit("PP 1 3 1 100.0");
    const int k = static_cast<int>(std::min<int64_t>(max_new, 4));
    for (int i = 0; i < k; ++i) emit("T " + std::to_string(kScript[i]));
    emit_done(k, "length");
}

void do_bgen(int slot, int64_t max_new) {
    emit("BADM " + std::to_string(slot) + " 1");
    const int k = static_cast<int>(std::min<int64_t>(max_new, 4));
    for (int i = 0; i < k; ++i) {
        emit("BT " + std::to_string(slot) + " " + std::to_string(kScript[i]));
    }
    emit("BDONE " + std::to_string(slot) + " " + std::to_string(k) + " stop 1");
}

}  // namespace

int main(int argc, char** argv) {
    const std::string scenario = argc > 1 ? argv[1] : "basic";

    if (scenario == "never_ready") {
        emit("INFO engine=mock");
        std::string line;
        while (std::getline(std::cin, line)) {}  // swallow forever
        return 0;
    }

    emit("INFO engine=mock context=65536");
    emit("READY 65536 stop");

    bool first_gen = true;
    std::string line;
    while (std::getline(std::cin, line)) {
        const std::vector<std::string> w = split_ws(line);
        if (w.empty()) continue;
        const std::string& cmd = w[0];

        if (cmd == "QUIT") return 0;

        if (cmd == "GEN") {
            if (scenario == "crash" && first_gen) return 1;  // die with no DONE
            if (scenario == "err") { emit("ERR boom"); continue; }
            if (scenario == "cancel") {
                // Two tokens, then wait for the driver's STOP and finish as cancelled.
                emit("PP 1 3 1 100.0");
                emit("T 101");
                emit("T 102");
                std::string s2;
                if (std::getline(std::cin, s2)) {
                    const std::vector<std::string> w2 = split_ws(s2);
                    if (!w2.empty() && w2[0] == "STOP") emit_done(1, "cancel");
                }
                first_gen = false;
                continue;
            }
            do_gen(w.size() > 1 ? std::atoll(w[1].c_str()) : 0);
            first_gen = false;
            continue;
        }

        if (cmd == "BGEN") {
            if (scenario == "refuse") { emit("ERR no free slot"); continue; }
            if (scenario == "err") { emit("ERR boom"); continue; }
            const int slot = w.size() > 1 ? std::atoi(w[1].c_str()) : 0;
            const int64_t n = w.size() > 2 ? std::atoll(w[2].c_str()) : 0;
            if (scenario == "cancel") {
                // One token, then wait for the driver's BSTOP and finish as cancelled.
                emit("BADM " + std::to_string(slot) + " 1");
                emit("BT " + std::to_string(slot) + " 101");
                std::string s2;
                if (std::getline(std::cin, s2)) {
                    const std::vector<std::string> w2 = split_ws(s2);
                    if (!w2.empty() && w2[0] == "BSTOP") {
                        emit("BDONE " + std::to_string(slot) + " 0 cancel 1");
                    }
                }
                continue;
            }
            do_bgen(slot, n);
            continue;
        }

        if (cmd == "BSTOP") {
            const int slot = w.size() > 1 ? std::atoi(w[1].c_str()) : 0;
            emit("BDONE " + std::to_string(slot) + " 0 cancel 1");
            continue;
        }

        if (scenario == "err") { emit("ERR boom"); continue; }
        // Any other command is ignored; the mock models the driver, not the engine.
    }
    return 0;
}
