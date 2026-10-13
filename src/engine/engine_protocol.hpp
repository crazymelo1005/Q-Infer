// Command builders for the vendored Strata engine's stdin line protocol.
//
// Wire format (verified against tests/data/strata_transcript_gen.txt, engine 0.1.43.1):
//   GEN  <max_new>[ <key>=<value> ...] <id>,<id>,...
//   BGEN <slot> <max_new>[ ...same keys...] <id>,<id>,...
// Every sampling key carries its own leading space; ids are comma-joined with no spaces.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace qinfer::engine {

// Sampling knobs for GEN/BGEN. A field left at its negative default is omitted from the
// line entirely, so the engine's own default for that key applies (never sent as -1).
struct SamplingParams {
    float temperature = -1.0f;
    float top_p = -1.0f;
    float min_p = -1.0f;
    int top_k = -1;
    long long seed = -1;
    int penalty_repeat = -1;
    int penalty_freq = -1;
    int penalty_present = -1;
    int penalty_last_n = -1;
};

// " temperature=0.8 top_p=0.95" — one leading space per present key, in the engine's order.
std::string sampling_keys(const SamplingParams&);

std::string build_gen(int64_t max_new, const SamplingParams&, const std::vector<int32_t>& ids);
std::string build_bgen(int slot, int64_t max_new, const SamplingParams&, const std::vector<int32_t>& ids);
std::string build_bstop(int slot);   // "BSTOP <slot>"
std::string build_stop();            // "STOP"
std::string build_quit();            // "QUIT"
std::string build_vram();            // "VRAM"

}  // namespace qinfer::engine
