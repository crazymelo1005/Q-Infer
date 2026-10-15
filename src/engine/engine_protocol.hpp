// Command builders for the vendored Strata engine's stdin line protocol.
//
// Wire format (verified against tests/data/strata_transcript_gen.txt, engine 0.1.43.1):
//   GEN  <max_new>[ <key>=<value> ...] <id>,<id>,...
//   BGEN <slot> <max_new>[ ...same keys...] <id>,<id>,...
// Every sampling key carries its own leading space; ids are comma-joined with no spaces.
#pragma once

#include <cstdint>
#include <map>
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
    float penalty_repeat = -1.0f;
    float penalty_freq = -1.0f;
    float penalty_present = -1.0f;
    int penalty_last_n = -1;
};

// " temperature=0.8 top_p=0.95" — one leading space per present key, in the engine's order.
std::string sampling_keys(const SamplingParams&);

std::string build_gen(int64_t max_new, const SamplingParams&, const std::vector<int32_t>& ids);
std::string build_bgen(int slot, int64_t max_new, const SamplingParams&, const std::vector<int32_t>& ids);
// GENI/BGENI: vision embedding injection. Same sampling-key shape as GEN/BGEN, but the token
// list is an embedding file path followed by space-separated image token ids (not comma-joined).
// Wire format: "GENI <max_new>[ <key>=<value> ...] <emb_file> <id>[ <id> ...]"
std::string build_geni(int64_t max_new, const SamplingParams&,
                       const std::string& emb_file, const std::vector<int32_t>& ids);
std::string build_bgeni(int slot, int64_t max_new, const SamplingParams&,
                        const std::string& emb_file, const std::vector<int32_t>& ids);
std::string build_bstop(int slot);   // "BSTOP <slot>"
std::string build_stop();            // "STOP"
std::string build_quit();            // "QUIT"
std::string build_vram();            // "VRAM"

// ---- parsing the engine's stdout ----------------------------------------------------------

// One parsed stdout line. An unrecognized command is kOther, never a failure: the protocol
// grows (0.1.43.1 already emits RESUME and REUSED, which the driver does not act on).
enum class EventKind {
    kReady, kInfo, kVram, kToken, kPrefill, kResume, kDone,
    kBadm, kBatchToken, kBatchDone, kYielded, kSession, kSwait,
    kSerr, kFatal, kErr, kOther
};

// The DONE line, in field order. A line with fewer fields than a later engine version emits
// still parses: missing trailing fields stay at their zero default (never index past present).
struct DoneInfo {
    uint64_t generated = 0, prompt_tokens = 0;
    double prompt_ms = 0, decode_ms = 0;
    std::string finish;              // stop | length | cancel | nonfinite
    uint64_t drafts_accepted = 0, drafts_offered = 0, reused = 0,
             hits = 0, lookups = 0, ram_blobs = 0, file_blobs = 0,
             prompt_read = 0, offloaded = 0;
    double file_mb = 0;
};

struct EngineEvent {
    EventKind kind = EventKind::kOther;
    std::string raw;                 // unparsed remainder / message text
    uint64_t u = 0;                  // READY ctx, PP position, RESUME n, BADM/YIELDED count
    int slot = -1;
    int32_t token = 0;               // T / BT
    double d = 0;                    // PP ms / BDONE ms
    double tok_s = 0;                // PP tok/s
    uint64_t prompt_tokens = 0;      // PP
    bool admitted = false;           // BADM <slot> 1|0
    DoneInfo done;
    std::map<std::string, std::string> fields;   // INFO / VRAM key=value pairs
};

EngineEvent parse_event(const std::string& line);

}  // namespace qinfer::engine
