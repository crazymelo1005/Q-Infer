// High-level driver over the engine's line protocol: one child process, one request at a time
// (v0 serializes callers, as the service layer already does). Token ids cross this boundary,
// not text: detokenization stays in the service layer, so the driver needs no tokenizer.
#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include "engine/engine_process.hpp"
#include "engine/engine_protocol.hpp"

namespace qinfer::engine {

struct DriverConfig {
    std::string exe;
    std::vector<std::string> engine_args;   // everything after "--serve" (which start() adds)
    std::string stderr_path;                // empty = /dev/null
    int ready_timeout_ms = 120000;          // real engines take seconds; the mock is instant
};

struct GenRequest {
    std::vector<int32_t> prompt;
    SamplingParams sampling;
    int64_t max_new = 512;
};

struct GenSummary {
    std::string finish;      // stop | length | cancel | nonfinite
    uint64_t generated = 0, prompt_tokens = 0, reused = 0;
    double decode_ms = 0, prompt_ms = 0;
    bool cancelled = false;
};

class EngineDriver {
public:
    EngineDriver() = default;
    ~EngineDriver();                        // QUIT + reap (SIGKILL if it ignores QUIT)
    EngineDriver(EngineDriver&&) noexcept;
    EngineDriver& operator=(EngineDriver&&) noexcept;
    EngineDriver(const EngineDriver&) = delete;
    EngineDriver& operator=(const EngineDriver&) = delete;

    // Spawn the engine and read until READY (bounded by ready_timeout_ms); the INFO fields and
    // the READY max_context are kept. Valid only after a successful start().
    static bool start(const DriverConfig&, EngineDriver& out, std::string& err);

    uint64_t max_context() const { return max_context_; }
    bool alive() const { return proc_.alive(); }
    const std::map<std::string, std::string>& info_fields() const { return info_fields_; }

    // on_token gets each generated id; returning false cancels: the driver writes STOP, keeps
    // reading to DONE, and returns true with summary.cancelled set.
    bool generate(const GenRequest&, const std::function<bool(int32_t)>& on_token,
                  GenSummary& out, std::string& err);

private:
    void shutdown();
    EngineProcess proc_;
    uint64_t max_context_ = 0;
    std::map<std::string, std::string> info_fields_;
};

}  // namespace qinfer::engine
