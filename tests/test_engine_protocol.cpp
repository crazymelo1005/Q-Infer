// Command builders for the vendored Strata engine's stdin line protocol.
// Literals are copied verbatim from tests/data/strata_transcript_gen.txt (Task 3): ids are
// comma-joined with no spaces, and each sampling key carries its own leading space.
#include "engine/engine_protocol.hpp"

#include <string>
#include <vector>

#include "check.hpp"

using namespace qinfer::engine;

int main() {
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
    return 0;
}
