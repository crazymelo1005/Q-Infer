#include "engine/engine_protocol.hpp"

#include <charconv>
#include <cstddef>

namespace qinfer::engine {
namespace {

// Shortest round-tripping decimal for a float: "0.8" / "0.95" / "0".
// (printf("%.10g", (double)0.95f) would widen to 0.9499999881; to_chars(float) does not.)
std::string fmt_float(float v) {
    char buf[32];
    const std::to_chars_result r = std::to_chars(buf, buf + sizeof(buf), v);
    return std::string(buf, r.ptr);
}

void append_key(std::string& out, const char* name, const std::string& value) {
    out += ' ';
    out += name;
    out += '=';
    out += value;
}

std::string join_ids(const std::vector<int32_t>& ids) {
    std::string out;
    for (std::size_t i = 0; i < ids.size(); ++i) {
        if (i != 0) out += ',';
        out += std::to_string(ids[i]);
    }
    return out;
}

}  // namespace

std::string sampling_keys(const SamplingParams& p) {
    std::string out;
    if (p.temperature >= 0.0f) append_key(out, "temperature", fmt_float(p.temperature));
    if (p.top_p >= 0.0f) append_key(out, "top_p", fmt_float(p.top_p));
    if (p.top_k >= 0) append_key(out, "top_k", std::to_string(p.top_k));
    if (p.min_p >= 0.0f) append_key(out, "min_p", fmt_float(p.min_p));
    if (p.penalty_repeat >= 0) append_key(out, "penalty_repeat", std::to_string(p.penalty_repeat));
    if (p.penalty_freq >= 0) append_key(out, "penalty_freq", std::to_string(p.penalty_freq));
    if (p.penalty_present >= 0) append_key(out, "penalty_present", std::to_string(p.penalty_present));
    if (p.penalty_last_n >= 0) append_key(out, "penalty_last_n", std::to_string(p.penalty_last_n));
    if (p.seed >= 0) append_key(out, "seed", std::to_string(p.seed));
    return out;
}

std::string build_gen(int64_t max_new, const SamplingParams& p, const std::vector<int32_t>& ids) {
    return "GEN " + std::to_string(max_new) + sampling_keys(p) + " " + join_ids(ids);
}

std::string build_bgen(int slot, int64_t max_new, const SamplingParams& p,
                       const std::vector<int32_t>& ids) {
    return "BGEN " + std::to_string(slot) + " " + std::to_string(max_new) + sampling_keys(p) +
           " " + join_ids(ids);
}

std::string build_bstop(int slot) { return "BSTOP " + std::to_string(slot); }
std::string build_stop() { return "STOP"; }
std::string build_quit() { return "QUIT"; }
std::string build_vram() { return "VRAM"; }

}  // namespace qinfer::engine
