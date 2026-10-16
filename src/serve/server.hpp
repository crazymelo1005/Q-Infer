// HTTP wiring layer: httplib routing, auth, worker pool. Requests are parsed and
// validated on HTTP threads, then handed to the scheduler's bounded queue; slot
// workers drive the facade, and increments are written back to the HTTP thread
// through the Job event buffer (non-streaming collects them all, SSE per frame).
//
// A key v0 concurrency constraint: artifact::GgufFile reads go through a single
// mutable ifstream (seekg+read in gguf_table.cpp's read_at) and are not thread
// safe; the PLE GgufTable is the same. So one process-wide engine_mu_ serializes
// all mutable Engine access (generate and tokenizer().encode — encode writes the
// piece_cache_ shared by workers, see tokenizer.cpp:261/275) — the queue/slot/
// event-buffer structures are already in place for M2; the lock is the first
// throughput cut (per-thread file handles or pread come later). The lock lives in
// the service layer, not the facade: Engine::generate itself is lock-free and the
// service layer is the only parallel user of the engine.
#pragma once

#include <atomic>
#include <ctime>
#include <deque>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "serve/facade.hpp"
#include "serve/json.hpp"
#include "serve/prompt_cache.hpp"
#include "serve/prometheus.hpp"
#include "serve/records.hpp"
#include "serve/scheduler.hpp"
#include "serve/telemetry.hpp"
#include "serve/toolcall.hpp"
#include "engine/vision_process.hpp"

namespace httplib {
class Server;
struct Request;
struct Response;
}

namespace qinfer::serve {

// The 503 body is the single literal shared by the whole service (spec §4:
// the flat `{error: "queue full"}` shape).
inline constexpr const char* kQueueFullBody = "{\"error\":\"queue full\"}";

struct ServeConfig {
    std::string model_path;
    std::string host = "127.0.0.1";
    int port = 8080;
    std::string api_key;                 // empty = no auth (local default)
    std::size_t slots = 8;
    std::size_t queue_capacity = 0;      // 0 = 2 x slots (spec §4 item 1)
    int threads = 0;                     // httplib HTTP pool; 0 = 2 x slots + 2
    int compute_threads = 0;             // engine inference pool (scheduling::set_pool_threads); 0 = default
                                         // (half the physical cores, capped at 8 — conservative by requirement); 1 or negative = off
    // --gpu: enable the CUDA device side. auto = every visible GPU; otherwise a
    // comma-separated device index list (nvidia-smi numbering). Empty + non-auto
    // = pure CPU (current behavior). Only QINFER_CUDA_BUILD builds may use it.
    bool gpu_auto = false;               // --gpu auto (use every visible CUDA GPU at startup)
    std::vector<int> gpu_devices;        // --gpu 0,1 explicit list; empty = pure CPU
    std::string lang = "en";             // --lang en|zh: server-side output language
                                         // (startup logs; error messages remain English by API
                                         // convention with the detail string carried through)
    std::size_t conversation_cache = 64; // token-prefix cache entries; 0 = off (/metrics reads it live)
    std::uint64_t kv_resident = 0;       // --kv-resident N: per-layer device KV residency cap in
                                         // tokens (reversed semantics: 0 = all-resident; above N
                                         // the oldest pages stream back to the host store)
    std::uint64_t reject_after = 0;      // 0 = off; >0 = 503 every request from the Nth (integration tests; see main)
    // Compute backend (P2, ADR-017): "strata" = the vendored engine driven over its line protocol
    // (the default now that P2's end-to-end A/B is green), "inhouse" = the in-house Engine,
    // "native" = the self-written NativeBackend (Phase 1 native engine).
    // strata needs engine_exe; engine_args are strata-only.
    std::string backend = "strata";
    std::string engine_exe;                  // --engine-exe: path to the vendored engine binary
    std::vector<std::string> engine_args;    // --engine-arg ARG, repeatable (order preserved)
    std::vector<int> strata_gpu_devices;     // parsed from --engine-arg --gpu for strata backend;
                                             // empty = monitor all visible GPUs (engine default)
    // Facade geometry passthrough (defaults = real-model derived values; integration tests override).
    FacadeConfig facade;
    // Vision (M5b): optional image encoder. When mmproj_path is non-empty, the server starts
    // a VisionProcess and accepts image_url parts in chat requests.
    std::string mmproj_path;           // --mmproj: path to mmproj.gguf
    std::string vision_binary;         // --vision-binary: path to qinfer-vision (empty = the
                                       // per-arm default below; the iGPU arm is build-vision-vk/)
    std::string media_root;            // --media-root: allowed prefix for file:// URLs
    // --vision-gpu on|off: pass --gpu to the encoder child. With the ggml-vulkan build
    // (tools/build_vision.sh STRATA_VISION_VULKAN=ON) that runs the encoder on Vulkan device 0,
    // which on this box is the Intel iGPU: 192 ms against 1375 ms on 8 CPU threads, in system
    // memory, so it costs the engine no VRAM. ON by default (2026-10-15): the A/B at ADR-009's
    // distribution level (measure/vision_quality.py, 5 images x 220-token generations) found
    // the encoder's fp16 divergence below the pipeline's own noise floor -- token-by-token
    // greedy text is not self-repeatable on either arm, so the arms are indistinguishable at
    // end-task level. --vision-gpu off selects the CPU encoder for machines without the
    // Vulkan build; if the iGPU binary fails to start, the server falls back to the CPU one.
    bool vision_gpu = true;
    // --preset auto|en-code|cjk|off: the use-case bundle below (see apply_preset). Operator flags
    // always win; the preset only fills in what a measurement says is better for this use.
    std::string preset = "auto";
    std::string draft_vocab;           // --draft-vocab FILE: explicit MTP draft-token subset
    std::string preset_note;           // filled by apply_preset; main prints it as one startup line
};

// The encoder binary that --vision-gpu selects by default: the ggml-vulkan build when the
// encoder runs on the iGPU, the plain build otherwise. An explicit --vision-binary wins.
inline const char* default_vision_binary(bool gpu) {
    return gpu ? "build-vision-vk/bin/qinfer-vision" : "build-vision/bin/qinfer-vision";
}

// ---- the two defaults added 2026-10-16 (both are overridable, and both are what the
// ---- deployment on this box was measured with; see docs ADR-021's correction).

// Whether the operator already named an expert tier in --engine-arg. --resident-experts is the
// engine's own spelling for "mmap + pin the complement"; naming any of the three keeps it.
inline bool has_expert_tier(const std::vector<std::string>& engine_args) {
    for (const std::string& s : engine_args)
        if (s == "--mmap-experts" || s == "--resident-experts" || s == "--resident-cpu-experts") return true;
    return false;
}

// Expert tier default: the engine's resident RAM mode. It is --mmap-experts plus
// --resident-cpu-experts --resident-pin --resident-soft: the experts the GPU cache cannot hold
// (16-20 GiB on the sc117 IQ2_XS pairing) are copied into a page-locked buffer, so a miss is a
// DMA instead of a page-cache fault plus a CPU memcpy. Measured on 2x RTX 5060 Ti + sc117
// IQ2_XS, cold prompts: prefill 1,010-1,087 -> 1,585.9 tok/s at 4K (+46..57%), 1,786.9 ->
// 2,368.7 at 21K (+33%), and 3,317.7 tok/s at 131K with 0.0 MB of expert reads. The engine
// sizes the pinned complement from the RAM it can actually get, so this is self-limiting;
// --engine-arg --mmap-experts restores the old tier.
inline void apply_expert_tier_default(const std::string& backend, std::vector<std::string>& engine_args) {
    if (backend == "strata" && !has_expert_tier(engine_args)) engine_args.push_back("--resident-experts");
}

// What --vision asked for. Auto (the default) enables the encoder whenever an mmproj can be
// found, On demands one, Off refuses one.
enum class VisionRequest { Auto, On, Off };

// The mmproj beside the model: a single mmproj*.gguf in --model's directory (lexicographically
// first when a directory holds several), empty when there is none or the directory is unreadable.
inline std::string find_mmproj_beside(const std::string& model_path) {
    std::error_code ec;
    const std::filesystem::path p(model_path);
    const std::filesystem::path dir = p.has_parent_path() ? p.parent_path() : std::filesystem::path(".");
    std::string best;
    for (std::filesystem::directory_iterator it(dir, ec), end; !ec && it != end; it.increment(ec)) {
        if (!it->is_regular_file(ec)) continue;
        const std::string name = it->path().filename().string();
        if (name.size() <= 11 || name.compare(0, 6, "mmproj") != 0) continue;
        if (name.compare(name.size() - 5, 5, ".gguf") != 0) continue;
        const std::string full = it->path().string();
        if (best.empty() || full < best) best = full;
    }
    return best;
}

// Resolve the vision encoder's mmproj for a request: an explicit --mmproj wins, Auto/On look
// beside the model, Off returns empty. Returns false with err set when On cannot find one, or
// when Off was asked for together with an explicit --mmproj (a contradiction, not a preference).
inline bool resolve_vision(VisionRequest req, const std::string& model_path, const std::string& explicit_mmproj,
                           std::string& out_mmproj, std::string& err) {
    if (req == VisionRequest::Off) {
        if (!explicit_mmproj.empty()) { err = "--vision off conflicts with --mmproj"; return false; }
        out_mmproj.clear();
        return true;
    }
    if (!explicit_mmproj.empty()) { out_mmproj = explicit_mmproj; return true; }
    out_mmproj = find_mmproj_beside(model_path);
    if (out_mmproj.empty() && req == VisionRequest::On) {
        err = "--vision on: no --mmproj given and no mmproj*.gguf beside " + model_path;
        return false;
    }
    return true;
}

// ---- the use-case preset (2026-10-16). The vendor's pattern is a table of bitwise speed switches
// ---- keyed on the GPU arch (arch_defaults.cpp) that only its AMD builds consult; this is the same
// ---- idea keyed on what we measured on this hardware, with every entry naming its measurement and
// ---- with the operator's own flags always winning.
//
//   auto     the neutral defaults and nothing else -- the expert tier (apply_expert_tier_default)
//            and the vision encoder on the iGPU (vision_gpu). Every run already gets these.
//   en-code  auto + the English/code draft vocabulary (--mtp-draft-vocab). Measured on a 300-token
//            English generation: decode 91.2 tok/s against 85.9 and 84.5 with the engine's shipped
//            CJK-inclusive default (+7.4%; draft acceptance 77.8% against 72.0/72.4%), and the
//            draft head drops 137.9 -> 52.6 MiB, which the expert cache takes back (+60 slots).
//   cjk      auto only, said out loud: the engine's own default draft vocabulary covers every
//            Chinese, Japanese and Korean token (106,299 ids), and en-code costs Chinese answers
//            about 25% (62.9 against 83.9/83.1 tok/s, acceptance 34.7%). Pick this when Chinese
//            dominates.
//   off      no preset at all; whatever the operator passed.
inline bool apply_preset(ServeConfig& cfg, std::string& note, std::string& err) {
    if (cfg.preset.empty() || cfg.preset == "off") return true;
    if (cfg.preset != "auto" && cfg.preset != "en-code" && cfg.preset != "cjk") {
        err = "--preset needs auto, en-code, cjk, or off, got \"" + cfg.preset + "\"";
        return false;
    }
    note = cfg.preset;
    if (cfg.preset != "en-code") return true;
    std::string vocab = cfg.draft_vocab;
    if (vocab.empty()) {
        const std::filesystem::path p(cfg.model_path);
        const std::filesystem::path dir = p.has_parent_path() ? p.parent_path() : std::filesystem::path(".");
        const std::string cand = (dir / "draft_vocab_en.bin").string();
        std::error_code ec;
        if (std::filesystem::is_regular_file(cand, ec)) vocab = cand;
    }
    if (vocab.empty()) {
        note += ": draft_vocab_en.bin not found beside --model and no --draft-vocab given; keeping the "
                "engine's CJK-inclusive default (copy the file next to the model, or pass --draft-vocab)";
        return true;
    }
    cfg.engine_args.push_back("--mtp-draft-vocab");
    cfg.engine_args.push_back(vocab);
    note += ": mtp-draft-vocab " + vocab;
    return true;
}

class Serve {
public:
    Serve(ServeConfig cfg, std::shared_ptr<Backend> eng);
    ~Serve();

    bool listen(std::string& err);   // register routes and bind; blocks in the HTTP pool; err on failure
    void stop();                     // thread-safe; callable from a signal handler

private:
    void register_routes();
    void worker_main();
    void handle_chat(const httplib::Request& req, httplib::Response& res, bool stream);
    void handle_completions(const httplib::Request& req, httplib::Response& res);
    void handle_messages(const httplib::Request& req, httplib::Response& res);
    void handle_count_tokens(const httplib::Request& req, httplib::Response& res);
    void serve_sse(const std::shared_ptr<Job>& job, bool include_usage,
                   const std::vector<Json>& tools, httplib::Response& res);
    void serve_anthropic_sse(const std::shared_ptr<Job>& job, const std::vector<Json>& tools,
                             httplib::Response& res);
    std::shared_ptr<Job> launch(const std::string& path, std::string prompt,
                                const GenOptions& opts, bool stream);
    void record_request(const std::shared_ptr<Job>& job, int http_status, bool disconnected);   // T12: last_timings; T13: gen totals; T14: records ring
    TelemetryExtra telemetry_extra() const;   // T13: engine-side metrics per telemetry sample (accumulated in record_request)

    ServeConfig cfg_;
    std::shared_ptr<Backend> eng_;   // the engine seam (ADR-017): in-house Engine, StrataBackend, or NativeBackend
    Scheduler sched_;
    PromptCache prompt_cache_;   // token-prefix cache: hits skip engine_mu_ (source of /metrics conversation_cache)
    std::mutex engine_mu_;             // see file header: v0 serializes engine access (encode + generate)
    std::vector<std::thread> workers_;
    std::atomic<std::uint64_t> req_id_{0};
    std::unique_ptr<httplib::Server> svr_;
    std::string web_index_;   // static page loaded at startup
    // Timing of the most recently completed request (shared by /v1/status last_timings and /metrics live).
    struct LastTimings {
        bool have = false;
        double queue_s = 0.0, first_token_s = 0.0, wallclock_s = 0.0, decode_tok_s = 0.0;
        std::uint64_t prompt_tokens = 0, output_tokens = 0;
        std::string finish;
    };
    mutable std::mutex last_mu_;     // telemetry_extra() is const, so the lock needs mutable (like EventBuffer::mu_)
    LastTimings last_;
    std::time_t started_;              // server start time (/v1/status uptime_s)
    // Telemetry: started in the constructor since T13 (launch only runs on requests;
    // sampling must happen even when idle); /v1/status machine and /metrics read it.
    std::unique_ptr<Telemetry> tel_;
    // tok_s_mean accumulation: the latest request's decode rate is v0 tok_s; the
    // all-time mean is tracked by these two fields.
    std::uint64_t gen_done_ = 0;   // total generated tokens (successful requests only)
    std::uint64_t gen_us_ = 0;     // total decode time (us)
    std::atomic<std::uint64_t> reused_{0};   // prompt tokens served from cache (totals.reused; written by workers)
    // Request records ring (source of /metrics requests/totals and /api/requests; filled by record_request since T14).
    Records records_;
    // Three latency histograms (ttft/itl/e2e families on the /metrics Prometheus
    // text port; observed by record_request at completion, internally locked,
    // snapshotted on HTTP threads — unrelated to the last_ critical section).
    Latencies lat_;
    // Vision encoder (M5b): null when --mmproj is not configured. Shared pointer so the
    // HTTP threads can access it without locking (VisionProcess has its own internal mutex).
    std::shared_ptr<engine::VisionProcess> vision_;
    // User feedback (thumbs) from the chat UI: a small bounded in-memory list, newest
    // first, served back on GET /api/feedback. Bounded so a chatty client cannot grow it.
    std::mutex feedback_mu_;
    std::deque<Json> feedback_;   // cap kFeedbackCap (see server.cpp)
    // /metrics totals (updated in the same critical section as last_ —
    // record_request is the only writer from HTTP threads; /metrics holds last_mu_
    // when reading, so no race).
    struct Totals {
        std::uint64_t prompt_tokens = 0, completion_tokens = 0, requests = 0;
        double prompt_ms = 0.0, decode_ms = 0.0;
    };
    Totals totals_;
};

// Used by main.cpp: command-line parsing (hand-rolled key/value scan, zero deps).
ServeConfig parse_flags(int argc, char** argv, std::string& err);

}  // namespace qinfer::serve