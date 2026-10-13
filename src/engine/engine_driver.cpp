#include "engine/engine_driver.hpp"

#include <chrono>
#include <utility>

namespace qinfer::engine {

EngineDriver::~EngineDriver() { shutdown(); }

EngineDriver::EngineDriver(EngineDriver&& o) noexcept
    : proc_(std::move(o.proc_)), max_context_(o.max_context_),
      info_fields_(std::move(o.info_fields_)) {}

EngineDriver& EngineDriver::operator=(EngineDriver&& o) noexcept {
    if (this != &o) {
        shutdown();
        proc_ = std::move(o.proc_);
        max_context_ = o.max_context_;
        info_fields_ = std::move(o.info_fields_);
    }
    return *this;
}

bool EngineDriver::start(const DriverConfig& cfg, EngineDriver& out, std::string& err) {
    err.clear();

    ProcessConfig pc;
    pc.exe = cfg.exe;
    pc.args.push_back("--serve");  // the engine's serve mode; matches Popen([exe, "--serve", *args])
    for (const std::string& a : cfg.engine_args) pc.args.push_back(a);
    pc.stderr_path = cfg.stderr_path;

    if (!EngineProcess::start(pc, out.proc_, err)) return false;

    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::milliseconds(cfg.ready_timeout_ms);
    std::string line;
    for (;;) {
        const int remaining = static_cast<int>(
            std::chrono::duration_cast<std::chrono::milliseconds>(
                deadline - std::chrono::steady_clock::now()).count());
        if (remaining <= 0) {
            err = "engine did not become ready in time";
            out.proc_.kill();
            return false;
        }
        if (!out.proc_.read_line_timeout(line, err, remaining)) {
            if (err.empty()) err = "engine closed stdout before READY";
            out.proc_.kill();
            return false;
        }
        const EngineEvent e = parse_event(line);
        if (e.kind == EventKind::kInfo) {
            out.info_fields_ = e.fields;
            continue;
        }
        if (e.kind == EventKind::kReady) {
            out.max_context_ = e.u;
            return true;
        }
        if (e.kind == EventKind::kErr || e.kind == EventKind::kFatal) {
            err = "engine " + line;
            out.proc_.kill();
            return false;
        }
        // Anything else before READY is ignored (the engine may chatter before it is ready).
    }
}

bool EngineDriver::generate(const GenRequest& req, const std::function<bool(int32_t)>& on_token,
                            GenSummary& out, std::string& err) {
    out = GenSummary{};
    err.clear();

    if (!proc_.write_line(build_gen(req.max_new, req.sampling, req.prompt), err)) return false;

    bool cancelled = false;
    std::string line;
    for (;;) {
        if (!proc_.read_line(line, err)) {
            if (err.empty()) err = "engine closed stdout during generation";
            return false;
        }
        const EngineEvent e = parse_event(line);
        switch (e.kind) {
            case EventKind::kToken:
                if (on_token && !on_token(e.token) && !cancelled) {
                    if (!proc_.write_line(build_stop(), err)) return false;
                    cancelled = true;
                }
                break;
            case EventKind::kDone:
                out.finish = e.done.finish;
                out.generated = e.done.generated;
                out.prompt_tokens = e.done.prompt_tokens;
                out.reused = e.done.reused;
                out.decode_ms = e.done.decode_ms;
                out.prompt_ms = e.done.prompt_ms;
                out.cancelled = cancelled || e.done.finish == "cancel";
                return true;
            case EventKind::kErr:
            case EventKind::kFatal:
                err = "engine " + line;
                return false;
            default:
                break;  // PP/RESUME/INFO (and anything else) are recorded and ignored here
        }
    }
}

void EngineDriver::shutdown() {
    if (!proc_.alive()) return;
    std::string err;
    proc_.write_line(build_quit(), err);
    if (proc_.wait_timeout(2000) < 0) proc_.kill();  // ignored QUIT: SIGKILL
}

}  // namespace qinfer::engine
