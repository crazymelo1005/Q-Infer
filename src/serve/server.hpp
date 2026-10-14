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
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "serve/facade.hpp"
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
    std::string vision_binary;         // --vision-binary: path to qinfer-vision (default: build-vision/bin/qinfer-vision)
    std::string media_root;            // --media-root: allowed prefix for file:// URLs
};

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
    std::string web_index_, web_monitor_;   // static pages loaded at startup
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