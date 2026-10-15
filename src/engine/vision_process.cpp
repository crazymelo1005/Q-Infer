#include "engine/vision_process.hpp"

#include <charconv>
#include <cstdlib>
#include <sstream>
#include <utility>

namespace qinfer::engine {

VisionProcess::~VisionProcess() { kill(); }

VisionProcess::VisionProcess(VisionProcess&& o) noexcept
    : proc_(std::move(o.proc_)) {}

VisionProcess& VisionProcess::operator=(VisionProcess&& o) noexcept {
    if (this != &o) {
        kill();
        proc_ = std::move(o.proc_);
    }
    return *this;
}

bool VisionProcess::start(const ProcessConfig& cfg, VisionProcess& out, std::string& err) {
    err.clear();
    if (!EngineProcess::start(cfg, out.proc_, err)) return false;

    // Wait for the READY handshake. 30s is plenty for the CPU encoder (it reaches READY in ~0.75s
    // on a warm cache), but a ggml-vulkan encoder needs far longer to come up: measured on the
    // iGPU, ~49s to READY with only the Intel ICD visible, ~80s when the two RTX cards are also
    // enumerated (ggml-vulkan probes every device). The encoder is resident, so this is paid once
    // per process, not per image -- but the handshake has to wait it out or vision silently
    // disables itself. Env knobs that help: VK_ICD_FILENAMES=<intel_icd.json> skips the RTX probe.
    std::string line;
    if (!out.proc_.read_line_timeout(line, err, 240000)) {
        out.proc_.kill();
        if (err.empty()) err = "vision encoder: EOF before READY";
        return false;
    }
    if (line.rfind("READY", 0) != 0) {
        out.proc_.kill();
        err = "vision encoder: expected READY, got: " + line;
        return false;
    }
    return true;
}

bool VisionProcess::encode(const std::string& image_path, const std::string& out_path,
                           std::string& err) {
    EncodeResult r;
    encode(image_path, out_path, err, r);
    return r.ok;
}

EncodeResult VisionProcess::encode(const std::string& image_path, const std::string& out_path,
                                   std::string& err, EncodeResult& result) {
    result = EncodeResult{};
    err.clear();

    std::lock_guard<std::mutex> lock(mu_);

    // Send the ENC command.
    const std::string cmd = "ENC " + image_path + " " + out_path;
    if (!proc_.write_line(cmd, err)) {
        result.err = "vision write failed: " + err;
        return result;
    }

    // Read the response line.
    std::string line;
    if (!proc_.read_line(line, err)) {
        result.err = "vision read failed: " + err;
        return result;
    }

    // Parse "OK <n_tokens> <nx> <ny> <ms>" or "ERR <msg>".
    if (line.rfind("OK", 0) == 0) {
        result.ok = true;
        // Tokenize the remainder.
        std::istringstream iss(line.substr(3));
        iss >> result.n_tokens >> result.nx >> result.ny >> result.ms;
    } else if (line.rfind("ERR", 0) == 0) {
        result.ok = false;
        result.err = line.substr(4);
    } else {
        result.ok = false;
        result.err = "vision encoder: unexpected response: " + line;
    }
    err = result.err;
    return result;
}

bool VisionProcess::alive() const {
    std::lock_guard<std::mutex> lock(mu_);
    return proc_.alive();
}

void VisionProcess::kill() {
    std::lock_guard<std::mutex> lock(mu_);
    proc_.kill();
}

}  // namespace qinfer::engine
