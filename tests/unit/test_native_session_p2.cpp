// Phase 2: NativeSession enhancements — reset() and chunked prefill
// Tests:
// 1. reset() clears session state for new conversation
// 2. prefill_chunked() processes tokens in chunks with same result as single-pass
// 3. Session state persists across multiple run_token() calls

#include "serve/native_session.hpp"
#include "artifact/gguf_table.hpp"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace qinfer;

int main() {
    const char* model_path = std::getenv("QINFER_TEST_MODEL");
    if (!model_path) {
        std::fprintf(stderr, "QINFER_TEST_MODEL not set, skipping\n");
        return 0;
    }

    artifact::GgufFile gguf;
    std::string err;
    if (!gguf.open(model_path, err)) {
        std::fprintf(stderr, "Failed to open model: %s\n", err.c_str());
        return 1;
    }

    // Build session
    serve::NativeSession session;
    dense::StepConfig cfg{};
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

    if (!session.open_model(gguf, cfg, err)) {
        std::fprintf(stderr, "open_model failed: %s\n", err.c_str());
        return 1;
    }

    // Test 1: reset()
    {
        dense::TokenEmbed te;
        if (!dense::build_token_embed(gguf, 2560, te, err)) {
            std::fprintf(stderr, "build_token_embed failed: %s\n", err.c_str());
            return 1;
        }

        // Prepare and run a token
        if (!session.prepare(64, 0, 0, err)) {
            std::fprintf(stderr, "prepare failed: %s\n", err.c_str());
            return 1;
        }
        if (!session.run_token(gguf, te, 1, err)) {
            std::fprintf(stderr, "run_token failed: %s\n", err.c_str());
            return 1;
        }
        assert(session.position() == 1);

        // Reset should clear position
        session.reset();
        assert(session.position() == 0);

        std::printf("PASS: reset() clears session state\n");
    }

    // Test 2: chunked prefill produces same result as single-pass
    {
        dense::TokenEmbed te;
        if (!dense::build_token_embed(gguf, 2560, te, err)) {
            std::fprintf(stderr, "build_token_embed failed: %s\n", err.c_str());
            return 1;
        }

        std::vector<int32_t> prompt = {1, 23, 456, 789, 1011, 1213};

        // Single-pass
        session.reset();
        if (!session.prepare(64, 0, 0, err)) {
            std::fprintf(stderr, "prepare failed: %s\n", err.c_str());
            return 1;
        }
        for (int32_t tok : prompt) {
            if (!session.run_token(gguf, te, tok, err)) {
                std::fprintf(stderr, "run_token failed: %s\n", err.c_str());
                return 1;
            }
        }
        auto logits_single = session.logits();
        auto pos_single = session.position();

        // Chunked (chunk_size=2)
        session.reset();
        if (!session.prepare(64, 0, 0, err)) {
            std::fprintf(stderr, "prepare failed: %s\n", err.c_str());
            return 1;
        }
        std::size_t chunks = 0;
        auto on_progress = [&chunks](std::size_t) { ++chunks; };
        if (!session.prefill_chunked(gguf, te, prompt, 2, on_progress, err)) {
            std::fprintf(stderr, "prefill_chunked failed: %s\n", err.c_str());
            return 1;
        }
        auto logits_chunked = session.logits();
        auto pos_chunked = session.position();

        // Verify same result
        assert(pos_single == pos_chunked);
        assert(logits_single.size() == logits_chunked.size());
        for (std::size_t i = 0; i < logits_single.size(); ++i) {
            assert(std::abs(logits_single[i] - logits_chunked[i]) < 1e-5f);
        }
        assert(chunks == 3);  // 6 tokens / chunk_size 2 = 3 chunks

        std::printf("PASS: chunked prefill matches single-pass (%zu chunks)\n", chunks);
    }

    // Test 3: session state persists across run_token() calls
    {
        dense::TokenEmbed te;
        if (!dense::build_token_embed(gguf, 2560, te, err)) {
            std::fprintf(stderr, "build_token_embed failed: %s\n", err.c_str());
            return 1;
        }

        session.reset();
        if (!session.prepare(64, 0, 0, err)) {
            std::fprintf(stderr, "prepare failed: %s\n", err.c_str());
            return 1;
        }

        // Run multiple tokens, position should increment
        for (int i = 0; i < 5; ++i) {
            if (!session.run_token(gguf, te, 1, err)) {
                std::fprintf(stderr, "run_token failed: %s\n", err.c_str());
                return 1;
            }
            assert(session.position() == static_cast<std::uint64_t>(i + 1));
        }

        std::printf("PASS: session state persists across run_token() calls\n");
    }

    std::printf("All Phase 2 NativeSession tests passed\n");
    return 0;
}
