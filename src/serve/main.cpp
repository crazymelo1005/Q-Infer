// Entry point: command-line parsing (hand-rolled key/value scan, zero deps) ->
// load the facade -> run Serve.
#include "httplib.h"
#include "scheduling/thread_pool.hpp"
#include "serve/facade.hpp"
#include "serve/server.hpp"
#ifdef QINFER_CUDA_BUILD
#include "dense/dense_exec_cuda.hpp"
#endif

#include <algorithm>
#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <thread>
#include <vector>

using namespace qinfer::serve;

namespace {

Serve* g_serve = nullptr;
volatile std::sig_atomic_t g_signaled = 0;

void on_signal(int) {
    if (g_serve) g_serve->stop();
    g_signaled = 1;
}

// Numeric args (ruling 6): the whole string must parse as a number (endptr at the
// end, no ERANGE overflow); garbage -> err, main exits 2.
std::int64_t arg_int(const std::map<std::string, std::string>& a, const char* key, std::int64_t dflt,
                     std::string& err) {
    const auto it = a.find(key);
    if (it == a.end()) return dflt;
    const std::string v = it->second;
    errno = 0;
    char* end = nullptr;
    const long long r = std::strtoll(v.c_str(), &end, 10);
    if (end == v.c_str() || *end != '\0' || errno == ERANGE) {
        err = "argument " + std::string(key) + " needs an integer, got \"" + v + "\"";
        return dflt;
    }
    return static_cast<std::int64_t>(r);
}

// --gpu parsing: auto (all visible GPUs) or a comma-separated list of non-negative
// device indices (nvidia-smi numbering). Invalid -> err.
bool parse_gpu(const std::map<std::string, std::string>& a, bool& gpu_auto,
               std::vector<int>& devices, std::string& err) {
    const auto it = a.find("--gpu");
    if (it == a.end()) return true;
    const std::string v = it->second;
    if (v == "auto") {
        gpu_auto = true;
        return true;
    }
    std::size_t start = 0;
    while (start <= v.size()) {
        const std::size_t comma = v.find(',', start);
        const std::string tok = v.substr(start, comma == std::string::npos ? std::string::npos
                                                                          : comma - start);
        if (tok.empty()) {
            err = "--gpu needs \"auto\" or a comma-separated device list, got \"" + v + "\" (empty segment)";
            return false;
        }
        errno = 0;
        char* end = nullptr;
        const long long r = std::strtoll(tok.c_str(), &end, 10);
        if (end == tok.c_str() || *end != '\0' || errno == ERANGE || r < 0) {
            err = "--gpu needs non-negative device indices, segment \"" + tok + "\" invalid (got " + v + ")";
            return false;
        }
        devices.push_back(static_cast<int>(r));
        if (comma == std::string::npos) break;
        start = comma + 1;
    }
    return true;
}

double arg_double(const std::map<std::string, std::string>& a, const char* key, double dflt,
                  std::string& err) {
    const auto it = a.find(key);
    if (it == a.end()) return dflt;
    const std::string v = it->second;
    errno = 0;
    char* end = nullptr;
    const double r = std::strtod(v.c_str(), &end);
    if (end == v.c_str() || *end != '\0' || errno == ERANGE) {
        err = "argument " + std::string(key) + " needs a number, got \"" + v + "\"";
        return dflt;
    }
    return r;
}

}  // namespace

// server.hpp declares qinfer::serve::parse_flags; the definition lives in that
// namespace here (main itself stays in the global scope).
namespace qinfer::serve {

ServeConfig parse_flags(int argc, char** argv, std::string& err) {
    std::map<std::string, std::string> a;
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        if (s.rfind("--", 0) != 0) { err = "unexpected positional argument: " + s; return {}; }
        const auto eq = s.find('=');
        const std::string k = eq == std::string::npos ? s : s.substr(0, eq);
        const std::string v = eq == std::string::npos ? (i + 1 < argc ? argv[++i] : std::string{}) : s.substr(eq + 1);
        if (v.empty()) { err = "argument " + k + " needs a value"; return {}; }
        a[k] = v;
    }
    ServeConfig c;
    const auto it = a.find("--model");
    if (it == a.end()) { err = "missing --model (path to the main-case GGUF shard)"; return {}; }
    c.model_path = it->second;
    if (a.count("--host")) c.host = a["--host"];
    c.port = static_cast<int>(arg_int(a, "--port", 8080, err));
    if (a.count("--api-key")) c.api_key = a["--api-key"];
    c.slots = static_cast<std::size_t>(arg_int(a, "--slots", 8, err));
    c.queue_capacity = static_cast<std::size_t>(arg_int(a, "--queue", 0, err));
    c.threads = static_cast<int>(arg_int(a, "--threads", 0, err));
    // Inference thread pool: default = half the physical cores, capped at 8
// (conservative by requirement; unrelated to the HTTP pool).
    {
        unsigned hw = std::thread::hardware_concurrency();
        const int dflt = hw > 0 ? static_cast<int>(std::min<unsigned>(hw / 2, 8)) : 2;
        c.compute_threads = static_cast<int>(arg_int(a, "--compute-threads", dflt, err));
    }
    // --lang en|zh: server-side output language (startup logs). Default en; CLI
// errors and usage text stay English.
    if (a.count("--lang")) {
        const std::string v = a["--lang"];
        if (v != "en" && v != "zh") { err = "--lang needs en or zh, got \"" + v + "\""; return {}; }
        c.lang = v;
    }
    c.conversation_cache = static_cast<std::size_t>(arg_int(a, "--conversation-cache", 64, err));
    c.kv_resident = static_cast<std::uint64_t>(arg_int(a, "--kv-resident", 0, err));
    if (!parse_gpu(a, c.gpu_auto, c.gpu_devices, err)) return {};
    c.reject_after = static_cast<std::uint64_t>(arg_int(a, "--reject-after", 0, err));
    FacadeConfig& f = c.facade;
    f.model_path = c.model_path;
    f.n_embd = static_cast<std::uint64_t>(arg_int(a, "--n-embd", 2560, err));
    f.hc = static_cast<std::uint64_t>(arg_int(a, "--hc", 4, err));
    f.hc_lr = static_cast<std::uint64_t>(arg_int(a, "--hc-lr", 320, err));
    f.n_layers = static_cast<int>(arg_int(a, "--layers", 48, err));
    f.top_k = static_cast<int>(arg_int(a, "--top-k", 10, err));
    f.gdn_head_dim = static_cast<std::uint64_t>(arg_int(a, "--gdn-head-dim", 128, err));
    f.qsa_head_dim = static_cast<std::uint64_t>(arg_int(a, "--qsa-head-dim", 256, err));
    f.n_rot = static_cast<std::uint64_t>(arg_int(a, "--n-rot", 64, err));
    f.idx_dim = static_cast<std::uint64_t>(arg_int(a, "--idx-dim", 128, err));
    f.idx_top_k = static_cast<std::uint64_t>(arg_int(a, "--idx-top-k", 2048, err));
    f.idx_block = static_cast<std::uint64_t>(arg_int(a, "--idx-block", 4, err));
    f.max_context = static_cast<std::uint64_t>(arg_int(a, "--max-context", 262144, err));
    f.cache_slots = static_cast<std::size_t>(arg_int(a, "--cache-slots", 0, err));
    f.cache_byte_budget = static_cast<std::uint64_t>(arg_int(a, "--cache-byte-budget", 0, err));
    f.ple_budget_bytes = static_cast<std::uint64_t>(arg_int(a, "--ple-budget-bytes", 0, err));
    if (a.count("--ple-table")) f.ple_table = a["--ple-table"];
    f.kv_slots = static_cast<std::uint64_t>(arg_int(a, "--kv-slots", 0, err));
    f.kv_resident = c.kv_resident;
#ifdef QINFER_CUDA_BUILD
    // --gpu activates dense device execution (stage A wiring): the CUDA whole-layer
    // entries are handed to the facade, so the step loop runs *_cuda per layer.
    // Without --gpu the executors stay empty -> pure host path (host-build behavior).
    if (c.gpu_auto || !c.gpu_devices.empty()) {
        c.facade.dense_exec = dense::dense_exec_cuda();
    }
#endif
    return c;
}

}  // namespace qinfer::serve

int main(int argc, char** argv) {
    std::string err;
    const ServeConfig cfg = parse_flags(argc, argv, err);
    if (!err.empty()) {
        std::fprintf(stderr, "%s\n", err.c_str());
        std::fprintf(stderr, "Usage: qinfer-server --model <gguf> [--host H] [--port P] [--api-key K] "
                             "[--slots N] [--queue N] [--threads N] [--compute-threads N] [--lang en|zh] "
                             "[--gpu auto|0,1] [--reject-after N] [--n-embd N] [--hc N] [--hc-lr N] "
                             "[--layers N] [--top-k N] [--gdn-head-dim N] [--qsa-head-dim N] [--n-rot N] "
                             "[--idx-dim N] [--idx-top-k N] [--idx-block N] [--max-context N] "
                             "[--cache-slots N] [--cache-byte-budget N] [--ple-table P] "
                             "[--ple-budget-bytes N] [--kv-slots N] [--kv-resident N] "
                             "[--conversation-cache N]\n"
                             "        --threads N is the HTTP thread pool; --compute-threads N is the "
                             "inference thread pool (default: half the physical cores, capped at 8; "
                             "1 or negative = single-threaded); --lang selects the startup-log language "
                             "(en default, zh optional); --gpu selects CUDA devices (auto = all, "
                             "0,1 = explicit; requires a CUDA-enabled build)\n");
        return 2;
    }
    // --reject-after: integration-test-only knob for deterministic 503s (reject every
    // request from the Nth one on); not for production. Enforced in the handler
    // before launch() against req_id_.
#ifndef QINFER_CUDA_BUILD
    if (cfg.gpu_auto || !cfg.gpu_devices.empty()) {
        std::fprintf(stderr,
                     "--gpu requires a CUDA-enabled build (cmake -B build/cuda -DQINFER_ENABLE_CUDA=ON); "
                     "this build has CUDA disabled\n");
        return 1;
    }
#endif
    std::shared_ptr<Engine> eng = std::make_shared<Engine>();
    if (!Engine::open(cfg.facade, *eng, err)) {
        std::fprintf(stderr, "Load failed: %s\n", err.c_str());
        return 1;
    }
    const bool zh = cfg.lang == "zh";
    if (zh) {
        std::printf("模型 %s（词表 %llu、n_embd %llu、%d 层）就绪\n", eng->model_name().c_str(),
                    (unsigned long long) eng->tokenizer().vocab_size(),
                    (unsigned long long) cfg.facade.n_embd, cfg.facade.n_layers);
    } else {
        std::printf("Model %s ready (vocab %llu, n_embd %llu, %d layers)\n",
                    eng->model_name().c_str(), (unsigned long long) eng->tokenizer().vocab_size(),
                    (unsigned long long) cfg.facade.n_embd, cfg.facade.n_layers);
    }
    // Inference thread pool: explicit --compute-threads N wins; default = half the
    // physical cores, capped at 8 (conservative by requirement). 1 or negative =
    // pool off (single-threaded, matching engine §6's set_pool_threads). The HTTP
    // pool is managed separately by cfg.threads (2*slots+2); they do not interact.
    qinfer::scheduling::set_pool_threads(cfg.compute_threads);
    if (zh) {
        std::printf("推理线程池：%d 线程（--compute-threads）\n", cfg.compute_threads);
    } else {
        std::printf("Inference thread pool: %d threads (--compute-threads)\n", cfg.compute_threads);
    }
    // Startup lines go to stdout, which is block-buffered when redirected to a
    // file: without an explicit flush the whole log stays invisible until exit.
    std::fflush(stdout);
    Serve serve(cfg, eng);
    g_serve = &serve;
    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);
    if (!serve.listen(err)) {
        std::fprintf(stderr, "%s\n", err.c_str());
        return 1;
    }
    if (zh) {
        std::printf("服务退出（信号 %d）\n", static_cast<int>(g_signaled));
    } else {
        std::printf("Server exiting (signal %d)\n", static_cast<int>(g_signaled));
    }
    return 0;
}