// Task 4: Token-by-token parity validation.
// NativeBackend must produce the SAME tokens as the existing Engine
// (which itself matches Strata subprocess). This is the correctness gate
// for the entire native engine effort.
//
// Test design:
//   1. Load the same model into both Engine and NativeBackend
//   2. Run the same prompt with the same params (greedy, seed=0, 20 tokens)
//   3. Compare token-by-token -- must match exactly
//
// Skips (returns 0) when QINFER_TEST_MODEL is not set.

#include "serve/facade.hpp"
#include "serve/native_backend.hpp"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

int main() {
    const char* model_path = std::getenv("QINFER_TEST_MODEL");
    if (!model_path) {
        std::fprintf(stderr, "QINFER_TEST_MODEL not set, skipping\n");
        return 0;
    }

    // Build both backends with identical config
    qinfer::serve::FacadeConfig cfg;
    cfg.model_path = model_path;
    cfg.n_embd = 2560;
    cfg.hc = 4;
    cfg.hc_lr = 320;
    cfg.n_layers = 48;
    cfg.top_k = 10;
    cfg.gdn_head_dim = 128;
    cfg.qsa_head_dim = 256;
    cfg.n_rot = 64;
    cfg.idx_dim = 128;
    cfg.idx_top_k = 2048;
    cfg.idx_block = 4;

    // Reference: existing Engine
    qinfer::serve::Engine engine;
    std::string err;
    if (!qinfer::serve::Engine::open(cfg, engine, err)) {
        std::fprintf(stderr, "Engine::open failed: %s\n", err.c_str());
        return 1;
    }

    // Subject: NativeBackend
    qinfer::serve::NativeBackend native;
    if (!qinfer::serve::NativeBackend::open(cfg, native, err)) {
        std::fprintf(stderr, "NativeBackend::open failed: %s\n", err.c_str());
        return 1;
    }

    // Same prompt, same params
    std::vector<std::int32_t> prompt = {1, 23, 456, 789};
    qinfer::serve::GenOptions opt;
    opt.max_tokens = 20;
    opt.temperature = 0.0f;  // greedy for determinism
    opt.seed = 0;

    qinfer::serve::GenResult engine_res, native_res;
    if (!engine.generate(prompt, opt, nullptr, engine_res, err)) {
        std::fprintf(stderr, "Engine generate failed: %s\n", err.c_str());
        return 1;
    }
    if (!native.generate(prompt, opt, nullptr, native_res, err)) {
        std::fprintf(stderr, "NativeBackend generate failed: %s\n", err.c_str());
        return 1;
    }

    // Compare token-by-token
    std::printf("Engine:  %zu tokens\n", engine_res.tokens.size());
    std::printf("Native:  %zu tokens\n", native_res.tokens.size());

    std::size_t n = std::min(engine_res.tokens.size(), native_res.tokens.size());
    bool match = true;
    for (std::size_t i = 0; i < n; ++i) {
        if (engine_res.tokens[i] != native_res.tokens[i]) {
            std::printf("MISMATCH at position %zu: engine=%d native=%d\n",
                        i, engine_res.tokens[i], native_res.tokens[i]);
            match = false;
            break;
        }
    }
    if (engine_res.tokens.size() != native_res.tokens.size()) {
        std::printf("LENGTH MISMATCH: engine=%zu native=%zu\n",
                    engine_res.tokens.size(), native_res.tokens.size());
        match = false;
    }

    if (match) {
        std::printf("PASS: token-by-token parity confirmed (%zu tokens)\n", n);
        return 0;
    } else {
        std::printf("FAIL: parity broken\n");
        return 1;
    }
}
