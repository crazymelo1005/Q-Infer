// Command builders for the vendored Strata engine's stdin line protocol.
// Literals are copied verbatim from tests/data/strata_transcript_gen.txt (Task 3): ids are
// comma-joined with no spaces, and each sampling key carries its own leading space.
#include "engine/engine_protocol.hpp"

#include <fstream>
#include <string>
#include <vector>

#include "check.hpp"

using namespace qinfer::engine;

// Sweep the recorded transcript: every line must parse, and the driver-critical shapes must be
// present. Run only when a path is given (the second ctest entry passes one); skipped otherwise.
static void sweep(const char* path) {
    std::ifstream in(path);
    CHECK(in.good());
    int ready = 0, tokens = 0, done = 0, errors = 0;
    uint64_t done_generated = 0;
    std::string line;
    while (std::getline(in, line)) {
        const EngineEvent e = parse_event(line);
        if (e.kind == EventKind::kReady) ++ready;
        if (e.kind == EventKind::kToken) ++tokens;
        if (e.kind == EventKind::kDone) { ++done; done_generated = e.done.generated; }
        if (e.kind == EventKind::kErr || e.kind == EventKind::kFatal) ++errors;
    }
    CHECK(ready >= 1);
    CHECK(tokens >= 1);
    CHECK(done == 1);
    CHECK(errors == 0);
    CHECK(done_generated == static_cast<uint64_t>(tokens));
}

int main(int argc, char** argv) {
    // A present float key prints with no trailing zeroes and a single leading space.
    SamplingParams p;
    p.temperature = 0.8f;
    p.top_p = 0.95f;
    CHECK(build_gen(128, p, {1, 2, 3}) == "GEN 128 temperature=0.8 top_p=0.95 1,2,3");

    // Negative fields are omitted entirely, not sent as -1.
    SamplingParams d;
    CHECK(build_gen(1, d, {7}) == "GEN 1 7");

    // seed is an integer and survives round-trip; only set when >= 0.
    SamplingParams s;
    s.seed = 42;
    CHECK(build_gen(4, s, {5, 6}) == "GEN 4 seed=42 5,6");

    CHECK(build_bgen(3, 16, d, {9, 10}) == "BGEN 3 16 9,10");
    CHECK(build_bstop(3) == "BSTOP 3");
    CHECK(build_stop() == "STOP");
    CHECK(build_quit() == "QUIT");
    CHECK(build_vram() == "VRAM");

    // READY: u is the max_context, raw is the trailing word.
    const auto ready = parse_event("READY 65536 stop");
    CHECK(ready.kind == EventKind::kReady && ready.u == 65536 && ready.raw == "stop");

    const auto tok = parse_event("T 1234");
    CHECK(tok.kind == EventKind::kToken && tok.token == 1234);

    // PP <position> <prompt_tokens> <ms> <tok/s>
    const auto pp = parse_event("PP 4096 93160 512 91.3");
    CHECK(pp.kind == EventKind::kPrefill && pp.u == 4096 && pp.prompt_tokens == 93160 &&
          pp.d == 512 && pp.tok_s == 91.3);

    // DONE: 15 fields, finish is the 5th.
    const auto done = parse_event("DONE 6 53 34620 2666 length 0 0 0 0 0 0 0 0 53 0");
    CHECK(done.kind == EventKind::kDone && done.done.generated == 6 &&
          done.done.prompt_tokens == 53 && done.done.finish == "length" &&
          done.done.offloaded == 0);

    const auto canc = parse_event("DONE 3 53 100 900 cancel 0 0 0 0 0 0 0 0 53 0");
    CHECK(canc.done.finish == "cancel");

    // A short DONE (fewer trailing fields than the engine emits) still parses.
    const auto short_done = parse_event("DONE 2 3 1 2 stop");
    CHECK(short_done.kind == EventKind::kDone && short_done.done.generated == 2 &&
          short_done.done.finish == "stop" && short_done.done.offloaded == 0);

    const auto badm = parse_event("BADM 2 1");
    CHECK(badm.kind == EventKind::kBadm && badm.slot == 2 && badm.admitted);

    const auto bt = parse_event("BT 4 777");
    CHECK(bt.kind == EventKind::kBatchToken && bt.slot == 4 && bt.token == 777);

    const auto bd = parse_event("BDONE 4 12 stop 133.5");
    CHECK(bd.kind == EventKind::kBatchDone && bd.slot == 4 && bd.done.generated == 12 &&
          bd.done.finish == "stop" && bd.d == 133.5);

    const auto inf = parse_event("INFO engine=0.1.43.1 context=65536 batch_slots=8");
    CHECK(inf.kind == EventKind::kInfo && inf.fields.at("engine") == "0.1.43.1" &&
          inf.fields.at("batch_slots") == "8");

    const auto er = parse_event("ERR no free slot");
    CHECK(er.kind == EventKind::kErr && er.raw == "no free slot");
    CHECK(parse_event("FATAL boom").kind == EventKind::kFatal);
    CHECK(parse_event("SERR io 1 disk gone").kind == EventKind::kSerr);
    // An unrecognized line is kOther, never a failure: the protocol grows.
    CHECK(parse_event("SAVED").kind == EventKind::kOther);

    if (argc > 1) sweep(argv[1]);
    return 0;
}

