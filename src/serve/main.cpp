// Entry point: command-line parsing (hand-rolled key/value scan, zero deps) ->
// load the facade -> run Serve.
#include "httplib.h"
#include "artifact/gguf_table.hpp"
#include "scheduling/thread_pool.hpp"
#include "serve/backend_strata.hpp"
#include "serve/facade.hpp"
#include "serve/native_backend.hpp"
#include "serve/prompt.hpp"
#include "serve/server.hpp"
#include "text/tokenizer.hpp"
#ifdef QINFER_CUDA_BUILD
#include "dense/dense_exec_cuda.hpp"
#endif

#include <algorithm>
#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <map>
#include <sstream>
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
    std::vector<std::string> engine_args;  // --engine-arg is repeatable; the map keeps only one value
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        if (s.rfind("--", 0) != 0) { err = "unexpected positional argument: " + s; return {}; }
        const auto eq = s.find('=');
        const std::string k = eq == std::string::npos ? s : s.substr(0, eq);
        const std::string v = eq == std::string::npos ? (i + 1 < argc ? argv[++i] : std::string{}) : s.substr(eq + 1);
        if (v.empty()) { err = "argument " + k + " needs a value"; return {}; }
        if (k == "--engine-arg") { engine_args.push_back(v); continue; }
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
    // --backend strata|inhouse|native (P2): strata drives the vendored engine over its line protocol;
    // native uses the self-written NativeBackend (Phase 1 native engine).
    if (a.count("--backend")) {
        const std::string v = a["--backend"];
        if (v != "inhouse" && v != "strata" && v != "native") {
            err = "--backend needs inhouse, strata, or native, got \"" + v + "\"";
            return {};
        }
        c.backend = v;
    }
    if (a.count("--engine-exe")) c.engine_exe = a["--engine-exe"];
    // M5b: vision encoder (optional; when --mmproj is given, the server starts qinfer-vision
    // and accepts image_url parts in chat requests).
    if (a.count("--mmproj")) c.mmproj_path = a["--mmproj"];
    if (a.count("--vision-binary")) c.vision_binary = a["--vision-binary"];
    if (a.count("--media-root")) c.media_root = a["--media-root"];
    // --igpu off|dense|pool: the vendored engine reads its iGPU switches from the
    // environment, and the engine child inherits ours. Default dense: with the four-format
    // Vulkan dense shaders and the engine's STRATA_IGPU_MIN_FLOPS size gate (so only large
    // dense projections move), iGPU dense is a small net win on the sc117 IQ2_XS pairing
    // (2xRTX 5060 Ti, 2026-10-15): a 300-token prompt went 178.8 -> 187.5 tok/s prefill
    // (+4.9%) and 14.7 -> 15.1 tok/s decode (+2.6%); a 3.3K prompt is a wash (339 -> 342).
    // 'dense' no-ops when no Vulkan device is present. 'pool' stays off (0.36 tok/s,
    // ADR-020); --igpu off / --igpu pool opt out.
    if (a.count("--igpu")) {
        const std::string v = a["--igpu"];
        if (v != "off" && v != "dense" && v != "pool") {
            err = "--igpu needs off, dense, or pool, got \"" + v + "\"";
            return {};
        }
        ::setenv("STRATA_IGPU_DENSE", v == "dense" ? "1" : "0", 1);
        ::setenv("STRATA_IGPU_POOL", v == "pool" ? "1" : "0", 1);
    } else {
        if (::getenv("STRATA_IGPU_DENSE") == nullptr) ::setenv("STRATA_IGPU_DENSE", "1", 1);
        if (::getenv("STRATA_IGPU_POOL") == nullptr) ::setenv("STRATA_IGPU_POOL", "0", 1);
    }
    // Fused expert prefill path (engine env, inherited by the child): measured on the
    // sc117 IQ2_XS pairing at +16% cold prefill (820 -> 952 tok/s on a 3.3K prompt,
    // and 810 -> 891 on another). On by default; set STRATA_PF_FUSED=0 to opt out.
    if (::getenv("STRATA_PF_FUSED") == nullptr) ::setenv("STRATA_PF_FUSED", "1", 1);
    // Hand the CPU expert pool a share of the prefill (engine 'auto': it only applies
    // to chunks under its own ~3K cap). Measured +7..13% on top of the fused path
    // (891 -> 1003 tok/s at 3.3K). On by default; STRATA_PREFILL_CPU_SHARE=0 opts out.
    if (::getenv("STRATA_PREFILL_CPU_SHARE") == nullptr) ::setenv("STRATA_PREFILL_CPU_SHARE", "auto", 1);
    c.engine_args = std::move(engine_args);
    // For the Strata backend, parse --gpu from engine_args so telemetry can monitor only the GPUs
    // the engine actually uses (matching Strata's own monitoring convention). Without this, telemetry
    // would show all visible GPUs regardless of what the engine subprocess is using.
    if (c.backend == "strata") {
        for (std::size_t i = 0; i + 1 < c.engine_args.size(); ++i) {
            if (c.engine_args[i] == "--gpu") {
                const std::string& v = c.engine_args[i + 1];
                if (v != "auto") {
                    std::istringstream ss(v);
                    for (std::string tok; std::getline(ss, tok, ',');) {
                        if (!tok.empty()) c.strata_gpu_devices.push_back(std::atoi(tok.c_str()));
                    }
                }
                break;
            }
        }
    }
    if (c.backend == "strata" && c.engine_exe.empty()) {
        err = "--backend defaults to strata, which needs --engine-exe <path to the vendored "
              "engine binary>; pass --backend inhouse or --backend native to use a built-in engine";
        return {};
    }
    if (c.backend != "strata" && !c.engine_args.empty()) {
        err = "--engine-arg is only used with --backend strata";
        return {};
    }
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
                             "[--conversation-cache N] [--backend inhouse|strata|native] "
                             "[--engine-exe P] [--engine-arg A]...\n"
                             "        --threads N is the HTTP thread pool; --compute-threads N is the "
                             "inference thread pool (default: half the physical cores, capped at 8; "
                             "1 or negative = single-threaded); --lang selects the startup-log language "
                             "(en default, zh optional); --gpu selects CUDA devices (auto = all, "
                             "0,1 = explicit; requires a CUDA-enabled build); --backend defaults to "
                             "strata and then needs --engine-exe (pass --backend inhouse to run the "
                             "in-house engine, or --backend native for the self-written native engine)\n");
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
    if (cfg.backend == "strata" && (cfg.gpu_auto || !cfg.gpu_devices.empty())) {
        std::fprintf(stderr,
                     "--gpu applies to the in-house and native backends only; pass --backend inhouse "
                     "or --backend native, or give the strata engine its devices through --engine-arg\n");
        return 1;
    }
    std::shared_ptr<Backend> eng;
    if (cfg.backend == "strata") {
        // The vendored engine owns the weights; the GGUF here is read for metadata only
        // (tokenizer + chat template + a display name) -- no tensors are touched.
        qinfer::artifact::GgufFile g;
        if (!g.open(cfg.facade.model_path, err)) {
            std::fprintf(stderr, "Load failed: %s\n", err.c_str());
            return 1;
        }
        StrataBackendConfig sc;
        sc.facade = cfg.facade;
        sc.driver.exe = cfg.engine_exe;
        sc.driver.engine_args = cfg.engine_args;
        sc.driver.ready_timeout_ms = 180000;  // a real engine loads weights for seconds
        // The engine's stderr carries its load diagnostics and warnings; without this it goes to
        // /dev/null and a startup death (or a WARN) is invisible to the operator.
        sc.driver.stderr_path = "/dev/stderr";
        if (!qinfer::text::Tokenizer::from_gguf(g, sc.tokenizer, err) ||
            !chat_template_from_gguf(g, sc.chat_template, err)) {
            std::fprintf(stderr, "Load failed: %s\n", err.c_str());
            return 1;
        }
        const std::string* nm = g.meta_str("general.name");
        sc.model_name = (nm != nullptr && !nm->empty())
                            ? *nm
                            : std::filesystem::path(cfg.facade.model_path).stem().string();
        auto b = std::make_shared<StrataBackend>();
        if (!StrataBackend::open(std::move(sc), *b, err)) {
            std::fprintf(stderr, "Engine start failed: %s\n", err.c_str());
            return 1;
        }
        eng = b;
    } else if (cfg.backend == "native") {
        // NativeBackend: self-written backend using NativeSession (Phase 1 native engine).
        // Loads model, tokenizer, and chat template from GGUF; supports --gpu for CUDA offload.
        auto nb = std::make_shared<NativeBackend>();
        if (!NativeBackend::open(cfg.facade, *nb, err)) {
            std::fprintf(stderr, "Native backend load failed: %s\n", err.c_str());
            return 1;
        }
        eng = nb;
    } else {
        auto e = std::make_shared<Engine>();
        if (!Engine::open(cfg.facade, *e, err)) {
            std::fprintf(stderr, "Load failed: %s\n", err.c_str());
            return 1;
        }
        eng = e;
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