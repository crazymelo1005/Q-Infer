#include "engine/engine_protocol.hpp"

#include <cctype>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cstdlib>

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

std::vector<std::string> split_ws(const std::string& s) {
    std::vector<std::string> out;
    std::size_t i = 0;
    while (i < s.size()) {
        while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i]))) ++i;
        std::size_t j = i;
        while (j < s.size() && !std::isspace(static_cast<unsigned char>(s[j]))) ++j;
        if (j > i) out.emplace_back(s.substr(i, j - i));
        i = j;
    }
    return out;
}

// These never throw on malformed input: strtoull/strtod return 0 on no-conversion, which is
// the same default the field already carries, so a short or odd line degrades rather than aborts.
uint64_t to_u64(const std::string& s) { return std::strtoull(s.c_str(), nullptr, 10); }
double to_f64(const std::string& s) { return std::strtod(s.c_str(), nullptr); }
int to_int(const std::string& s) { return static_cast<int>(std::strtoll(s.c_str(), nullptr, 10)); }
int32_t to_i32(const std::string& s) { return static_cast<int32_t>(std::strtoll(s.c_str(), nullptr, 10)); }

// Join tokens [from, end) back with single spaces (the message text of READY/ERR/...).
std::string tail(const std::vector<std::string>& t, std::size_t from) {
    std::string out;
    for (std::size_t i = from; i < t.size(); ++i) {
        if (i > from) out += ' ';
        out += t[i];
    }
    return out;
}

void read_fields(const std::vector<std::string>& t, std::map<std::string, std::string>& out) {
    for (std::size_t i = 1; i < t.size(); ++i) {
        const std::size_t eq = t[i].find('=');
        if (eq != std::string::npos) out[t[i].substr(0, eq)] = t[i].substr(eq + 1);
    }
}

}  // namespace

std::string sampling_keys(const SamplingParams& p) {
    std::string out;
    if (p.temperature >= 0.0f) append_key(out, "temperature", fmt_float(p.temperature));
    if (p.top_p >= 0.0f) append_key(out, "top_p", fmt_float(p.top_p));
    if (p.top_k >= 0) append_key(out, "top_k", std::to_string(p.top_k));
    if (p.min_p >= 0.0f) append_key(out, "min_p", fmt_float(p.min_p));
    if (p.penalty_repeat >= 0.0f) append_key(out, "penalty_repeat", fmt_float(p.penalty_repeat));
    if (p.penalty_freq >= 0.0f) append_key(out, "penalty_freq", fmt_float(p.penalty_freq));
    if (p.penalty_present >= 0.0f) append_key(out, "penalty_present", fmt_float(p.penalty_present));
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

// GENI/BGENI: vision embedding injection. Unlike GEN, the ids are space-separated (not
// comma-joined), and an embedding file path precedes them.
namespace {
std::string join_ids_spaced(const std::vector<int32_t>& ids) {
    std::string out;
    for (std::size_t i = 0; i < ids.size(); ++i) {
        if (i != 0) out += ' ';
        out += std::to_string(ids[i]);
    }
    return out;
}
}  // namespace

std::string build_geni(int64_t max_new, const SamplingParams& p,
                       const std::string& emb_file, const std::vector<int32_t>& ids) {
    return "GENI " + std::to_string(max_new) + sampling_keys(p) + " " + emb_file + " " +
           join_ids_spaced(ids);
}

std::string build_bgeni(int slot, int64_t max_new, const SamplingParams& p,
                        const std::string& emb_file, const std::vector<int32_t>& ids) {
    return "BGENI " + std::to_string(slot) + " " + std::to_string(max_new) + sampling_keys(p) +
           " " + emb_file + " " + join_ids_spaced(ids);
}

std::string build_bstop(int slot) { return "BSTOP " + std::to_string(slot); }
std::string build_stop() { return "STOP"; }
std::string build_quit() { return "QUIT"; }
std::string build_vram() { return "VRAM"; }

EngineEvent parse_event(const std::string& line) {
    EngineEvent e;
    const std::vector<std::string> t = split_ws(line);
    if (t.empty()) return e;
    const std::string& cmd = t[0];

    if (cmd == "READY") {
        e.kind = EventKind::kReady;
        if (t.size() > 1) e.u = to_u64(t[1]);
        e.raw = tail(t, 2);
    } else if (cmd == "INFO") {
        e.kind = EventKind::kInfo;
        read_fields(t, e.fields);
    } else if (cmd == "VRAM") {
        e.kind = EventKind::kVram;
        read_fields(t, e.fields);
    } else if (cmd == "T") {
        e.kind = EventKind::kToken;
        if (t.size() > 1) e.token = to_i32(t[1]);
    } else if (cmd == "PP") {
        e.kind = EventKind::kPrefill;
        if (t.size() > 1) e.u = to_u64(t[1]);
        if (t.size() > 2) e.prompt_tokens = to_u64(t[2]);
        if (t.size() > 3) e.d = to_f64(t[3]);
        if (t.size() > 4) e.tok_s = to_f64(t[4]);
    } else if (cmd == "RESUME") {
        e.kind = EventKind::kResume;
        if (t.size() > 1) e.u = to_u64(t[1]);
    } else if (cmd == "DONE") {
        e.kind = EventKind::kDone;
        const std::size_t n = t.size() - 1;  // field count after the command word
        auto u_at = [&](std::size_t k, uint64_t& out) { if (k < n) out = to_u64(t[k + 1]); };
        auto f_at = [&](std::size_t k, double& out) { if (k < n) out = to_f64(t[k + 1]); };
        u_at(0, e.done.generated);
        u_at(1, e.done.prompt_tokens);
        f_at(2, e.done.prompt_ms);
        f_at(3, e.done.decode_ms);
        if (n > 4) e.done.finish = t[5];
        u_at(5, e.done.drafts_accepted);
        u_at(6, e.done.drafts_offered);
        u_at(7, e.done.reused);
        u_at(8, e.done.hits);
        u_at(9, e.done.lookups);
        u_at(10, e.done.ram_blobs);
        u_at(11, e.done.file_blobs);
        f_at(12, e.done.file_mb);
        u_at(13, e.done.prompt_read);
        u_at(14, e.done.offloaded);
    } else if (cmd == "BADM") {
        e.kind = EventKind::kBadm;
        if (t.size() > 1) e.slot = to_int(t[1]);
        if (t.size() > 2) e.admitted = (t[2] == "1");
    } else if (cmd == "BT") {
        e.kind = EventKind::kBatchToken;
        if (t.size() > 1) e.slot = to_int(t[1]);
        if (t.size() > 2) e.token = to_i32(t[2]);
    } else if (cmd == "BDONE") {
        e.kind = EventKind::kBatchDone;
        if (t.size() > 1) e.slot = to_int(t[1]);
        if (t.size() > 2) e.done.generated = to_u64(t[2]);
        if (t.size() > 3) e.done.finish = t[3];
        if (t.size() > 4) e.d = to_f64(t[4]);
    } else if (cmd == "YIELDED" || cmd == "BYIELDED") {
        e.kind = EventKind::kYielded;
        if (t.size() > 1) e.u = to_u64(t[1]);
    } else if (cmd == "SESSION") {
        e.kind = EventKind::kSession;
        e.raw = tail(t, 1);
    } else if (cmd == "SWAIT") {
        e.kind = EventKind::kSwait;
        e.raw = tail(t, 1);
    } else if (cmd == "SERR") {
        e.kind = EventKind::kSerr;
        e.raw = tail(t, 1);
    } else if (cmd == "FATAL") {
        e.kind = EventKind::kFatal;
        e.raw = tail(t, 1);
    } else if (cmd == "ERR") {
        e.kind = EventKind::kErr;
        e.raw = tail(t, 1);
    } else {
        // Any other word (INFO-adjacent lines, SAVED/RESTORED, RESUME's sibling RESUED, ...) is
        // not a failure: the protocol grows, so unknown lines are kOther with their text in raw.
        e.kind = EventKind::kOther;
        e.raw = tail(t, 1);
    }
    return e;
}

}  // namespace qinfer::engine
