// Spawn a qinfer-vision child process and drive its stdin/stdout line protocol for image
// encoding. The protocol: on startup the encoder prints "READY <ctx>", then for each
// "ENC <image> <out>" command it prints either "OK <n_tokens> <nx> <ny> <ms>" or "ERR <msg>".
//
// Thread-safe: encode() serializes with an internal mutex so multiple HTTP threads can share
// one encoder process. POSIX-only (same as EngineProcess).
#pragma once

#include <mutex>
#include <string>

#include "engine/engine_process.hpp"

namespace qinfer::engine {

struct EncodeResult {
    bool ok = false;
    int n_tokens = 0;
    int nx = 0;
    int ny = 0;
    double ms = 0.0;
    std::string err;
};

class VisionProcess {
public:
    VisionProcess() = default;
    ~VisionProcess();
    VisionProcess(VisionProcess&&) noexcept;
    VisionProcess& operator=(VisionProcess&&) noexcept;
    VisionProcess(const VisionProcess&) = delete;
    VisionProcess& operator=(const VisionProcess&) = delete;

    // Fork+exec the encoder binary; waits for the READY handshake. Returns false on exec
    // failure or READY timeout.
    static bool start(const ProcessConfig&, VisionProcess& out, std::string& err);

    // Send an ENC command and wait for the OK/ERR response. Thread-safe (serialized).
    bool encode(const std::string& image_path, const std::string& out_path, std::string& err);
    // Overload returning full result struct.
    EncodeResult encode(const std::string& image_path, const std::string& out_path,
                        std::string& err, EncodeResult& result);

    bool alive() const;
    void kill();

private:
    EngineProcess proc_;
    mutable std::mutex mu_;   // serializes encode() calls
};

}  // namespace qinfer::engine
