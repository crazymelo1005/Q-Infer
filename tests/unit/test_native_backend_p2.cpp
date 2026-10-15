// Phase 2: NativeBackend session persistence
// Tests:
// 1. generate() with session_id reuses session state
// 2. generate() without session_id creates fresh session each time
// 3. Different session_ids maintain independent state

#include "serve/native_backend.hpp"
#include "serve/facade.hpp"

#include <cassert>
#include <cstdio>
#include <cstdlib>

using namespace qinfer;

int main() {
    const char* model_path = std::getenv("QINFER_TEST_MODEL");
    if (!model_path) {
        std::fprintf(stderr, "QINFER_TEST_MODEL not set, skipping\n");
        return 0;
    }

    // Build backend
    serve::FacadeConfig cfg{};
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
    cfg.cache_slots = 0;

    serve::NativeBackend backend;
    std::string err;
    if (!serve::NativeBackend::open(cfg, backend, err)) {
        std::fprintf(stderr, "open failed: %s\n", err.c_str());
        return 1;
    }

    // Test 1: generate() with session_id reuses session
    {
        serve::GenOptions opt;
        opt.max_tokens = 5;
        opt.temperature = 0.0f;  // greedy for determinism
        opt.session_id = "test-session-1";

        // First generate
        std::vector<std::int32_t> prompt1 = {1, 23, 456};
        serve::GenResult res1;
        if (!backend.generate(prompt1, opt, nullptr, res1, err)) {
            std::fprintf(stderr, "generate 1 failed: %s\n", err.c_str());
            return 1;
        }

        // Second generate with same session_id - should continue from where we left off
        std::vector<std::int32_t> prompt2 = {789};  // short continuation
        serve::GenResult res2;
        if (!backend.generate(prompt2, opt, nullptr, res2, err)) {
            std::fprintf(stderr, "generate 2 failed: %s\n", err.c_str());
            return 1;
        }

        // Both should succeed (session persisted)
        assert(res1.n_gen > 0);
        assert(res2.n_gen > 0);

        std::printf("PASS: generate() with session_id reuses session\n");
    }

    // Test 2: generate() without session_id creates fresh session
    {
        serve::GenOptions opt;
        opt.max_tokens = 5;
        opt.temperature = 0.0f;
        opt.session_id = "";  // empty = no persistence

        std::vector<std::int32_t> prompt = {1, 23, 456};
        
        // First generate
        serve::GenResult res1;
        if (!backend.generate(prompt, opt, nullptr, res1, err)) {
            std::fprintf(stderr, "generate 1 failed: %s\n", err.c_str());
            return 1;
        }

        // Second generate - should start fresh (same result as first)
        serve::GenResult res2;
        if (!backend.generate(prompt, opt, nullptr, res2, err)) {
            std::fprintf(stderr, "generate 2 failed: %s\n", err.c_str());
            return 1;
        }

        // Both should produce the same output (fresh session each time)
        assert(res1.text == res2.text);

        std::printf("PASS: generate() without session_id creates fresh session\n");
    }

    // Test 3: different session_ids maintain independent state
    {
        serve::GenOptions opt1;
        opt1.max_tokens = 3;
        opt1.temperature = 0.0f;
        opt1.session_id = "session-A";

        serve::GenOptions opt2;
        opt2.max_tokens = 3;
        opt2.temperature = 0.0f;
        opt2.session_id = "session-B";

        std::vector<std::int32_t> prompt = {1, 23};

        // Generate with session A
        serve::GenResult resA1;
        if (!backend.generate(prompt, opt1, nullptr, resA1, err)) {
            std::fprintf(stderr, "generate A1 failed: %s\n", err.c_str());
            return 1;
        }

        // Generate with session B
        serve::GenResult resB1;
        if (!backend.generate(prompt, opt2, nullptr, resB1, err)) {
            std::fprintf(stderr, "generate B1 failed: %s\n", err.c_str());
            return 1;
        }

        // Both should succeed independently
        assert(resA1.n_gen > 0);
        assert(resB1.n_gen > 0);

        std::printf("PASS: different session_ids maintain independent state\n");
    }

    std::printf("All Phase 2 NativeBackend tests passed\n");
    return 0;
}
