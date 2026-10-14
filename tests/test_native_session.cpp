#include "serve/native_session.hpp"
#include "artifact/gguf_table.hpp"
#include "dense/step_loop.hpp"
#include "dense/token_embed.hpp"

#include <cassert>
#include <cstdio>
#include <string>

int main() {
    // Load model
    qinfer::artifact::GgufFile gguf;
    std::string err;
    const char* model_path = std::getenv("QINFER_TEST_MODEL");
    if (!model_path) {
        std::fprintf(stderr, "QINFER_TEST_MODEL not set, skipping\n");
        return 0;  // allow test to pass without model
    }
    if (!gguf.open(model_path, err)) {
        std::fprintf(stderr, "open failed: %s\n", err.c_str());
        return 1;
    }

    // Build session
    qinfer::serve::NativeSession session;
    qinfer::dense::StepConfig cfg{};
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

    // Prepare for inference
    if (!session.prepare(/*n_pages=*/64, /*pos_base=*/0,
                         /*kv_slots=*/0, err)) {
        std::fprintf(stderr, "prepare failed: %s\n", err.c_str());
        return 1;
    }

    // Build token embed
    qinfer::dense::TokenEmbed te;
    if (!qinfer::dense::build_token_embed(gguf, 2560, te, err)) {
        std::fprintf(stderr, "build_token_embed failed: %s\n", err.c_str());
        return 1;
    }

    // Run one token (token id = 1, typically BOS)
    if (!session.run_token(gguf, te, /*token=*/1, err)) {
        std::fprintf(stderr, "run_token failed: %s\n", err.c_str());
        return 1;
    }

    // Verify position advanced
    assert(session.position() == 1);

    // Verify logits are non-empty and finite
    const auto& logits = session.logits();
    assert(!logits.empty());
    for (float v : logits) {
        assert(std::isfinite(v));
    }

    std::printf("PASS: native_session produced %zu finite logits\n", logits.size());
    return 0;
}
