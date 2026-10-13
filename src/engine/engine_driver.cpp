#include "engine/engine_driver.hpp"

#include <chrono>
#include <cstdlib>
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
            out.cfg_ = cfg;
            out.init_slots_();
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

void EngineDriver::init_slots_() {
    // Conservative default (do not assume 8): grow only when the engine advertises batch_slots.
    int n = 2;
    const auto it = info_fields_.find("batch_slots");
    if (it != info_fields_.end()) {
        const int v = std::atoi(it->second.c_str());
        if (v > 0) n = v;
    }
    slots_.assign(static_cast<std::size_t>(n), SlotState::kFree);
    slot_cb_.assign(static_cast<std::size_t>(n), nullptr);
}

int EngineDriver::acquire_slot() {
    for (std::size_t i = 0; i < slots_.size(); ++i) {
        if (slots_[i] == SlotState::kFree) {
            slots_[i] = SlotState::kBusy;
            return static_cast<int>(i);
        }
    }
    return -1;
}

bool EngineDriver::generate_batch(int slot, const GenRequest& req,
                                  const std::function<bool(int32_t)>& on_token, GenSummary& out,
                                  std::string& err) {
    out = GenSummary{};
    err.clear();
    if (slot < 0 || slot >= static_cast<int>(slots_.size()) || slots_[slot] != SlotState::kBusy) {
        err = "generate_batch: slot is not acquired";
        return false;
    }
    slot_cb_[slot] = on_token;
    auto release = [&]() {
        slots_[slot] = SlotState::kFree;
        slot_cb_[slot] = nullptr;
    };

    if (!proc_.write_line(build_bgen(slot, req.max_new, req.sampling, req.prompt), err)) {
        release();
        return false;
    }

    std::string line;
    for (;;) {
        if (!proc_.read_line(line, err)) {
            if (err.empty()) err = "engine closed stdout during generation";
            release();
            return false;
        }
        const EngineEvent e = parse_event(line);
        switch (e.kind) {
            case EventKind::kBatchToken:
                if (e.slot == slot) {
                    if (slot_cb_[slot] && !slot_cb_[slot](e.token) &&
                        slots_[slot] != SlotState::kCancelling) {
                        if (!proc_.write_line(build_bstop(slot), err)) {
                            release();
                            return false;
                        }
                        slots_[slot] = SlotState::kCancelling;
                    }
                } else if (e.slot >= 0 && e.slot < static_cast<int>(slot_cb_.size()) &&
                           slot_cb_[e.slot]) {
                    slot_cb_[e.slot](e.token);  // re-dispatch to the owning slot's callback
                } else {
                    err = "BT for an unowned slot " + std::to_string(e.slot);
                    release();
                    return false;
                }
                break;
            case EventKind::kBatchDone:
                if (e.slot == slot) {
                    out.finish = e.done.finish;
                    out.generated = e.done.generated;
                    out.decode_ms = e.d;
                    out.cancelled =
                        e.done.finish == "cancel" || slots_[slot] == SlotState::kCancelling;
                    release();
                    return true;
                }
                break;  // another slot's BDONE belongs to its own call
            case EventKind::kErr:
            case EventKind::kFatal:
                // A refusal arrives as ERR with no BDONE: free the slot and report.
                err = "engine " + line;
                release();
                return false;
            default:
                break;
        }
    }
}

void EngineDriver::cancel_slot(int slot) {
    if (slot < 0 || slot >= static_cast<int>(slots_.size()) || slots_[slot] != SlotState::kBusy) {
        return;
    }
    std::string err;
    proc_.write_line(build_bstop(slot), err);
    slots_[slot] = SlotState::kCancelling;  // its BDONE ... cancel ends the owning call
}

bool EngineDriver::stop_and_recover(std::string& err) {
    err.clear();
    proc_.kill();  // make sure the old child is gone before respawning
    EngineDriver fresh;
    if (!EngineDriver::start(cfg_, fresh, err)) return false;
    *this = std::move(fresh);  // proc_, info fields and a fresh (all-free) slot table
    return true;
}

}  // namespace qinfer::engine
