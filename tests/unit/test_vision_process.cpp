// VisionProcess spawns the vision encoder and drives its stdin/stdout line protocol.
// Uses a shell echo script as a mock encoder (same approach as test_engine_process.cpp),
// so this runs anywhere without requiring the actual qinfer-vision binary or model weights.
#include "engine/vision_process.hpp"

#include <cstdlib>
#include <fstream>
#include <string>

#include "check.hpp"

using namespace qinfer::engine;

int main() {
    // Mock encoder: writes READY on startup, then for each ENC line responds with
    // "OK <n_tokens> <nx> <ny> <ms>" to simulate a successful encoding.
    // This validates the handshake and protocol parsing without needing real weights.
    ProcessConfig c;
    c.exe = "/bin/sh";
    c.args = {"-c", "echo READY 65536; while read -r line; do echo \"OK 256 32 32 12.5\"; done"};

    VisionProcess vp;
    std::string err;
    CHECK(VisionProcess::start(c, vp, err));
    CHECK(err.empty());

    // Encode should succeed with the mock returning OK.
    EncodeResult result;
    vp.encode("/path/to/image.png", "/path/to/out.sve1", err, result);
    CHECK(result.ok);
    CHECK(result.n_tokens == 256);
    CHECK(result.nx == 32);
    CHECK(result.ny == 32);

    // Process should still be alive.
    CHECK(vp.alive());
    vp.kill();
    CHECK(!vp.alive());

    // A nonexistent exe fails to start with a message, never aborts.
    ProcessConfig bad;
    bad.exe = "/nonexistent/qinfer-vision";
    VisionProcess vp2;
    err.clear();
    CHECK(!VisionProcess::start(bad, vp2, err));
    CHECK(!err.empty());

    // Mock encoder that returns ERR: encode should report failure.
    ProcessConfig err_cfg;
    err_cfg.exe = "/bin/sh";
    err_cfg.args = {"-c", "echo READY 65536; while read -r line; do echo \"ERR out of memory\"; done"};
    VisionProcess vp3;
    CHECK(VisionProcess::start(err_cfg, vp3, err));
    EncodeResult err_result;
    vp3.encode("/img.png", "/out.sve1", err, err_result);
    CHECK(!err_result.ok);
    CHECK(err_result.err.find("out of memory") != std::string::npos);
    vp3.kill();

    return 0;
}
