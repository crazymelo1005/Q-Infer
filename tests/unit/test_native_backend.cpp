#include "serve/native_backend.hpp"
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <string>

int main() {
    const char* model_path = std::getenv("QINFER_TEST_MODEL");
    if (!model_path) {
        std::fprintf(stderr, "QINFER_TEST_MODEL not set, skipping\n");
        return 0;
    }

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

    qinfer::serve::NativeBackend backend;
    std::string err;
    if (!qinfer::serve::NativeBackend::open(cfg, backend, err)) {
        std::fprintf(stderr, "open failed: %s\n", err.c_str());
        return 1;
    }

    // Verify backend interface
    assert(!backend.model_name().empty());
    assert(backend.config().n_layers == 48);

    // Generate a few tokens
    qinfer::serve::GenOptions opt;
    opt.max_tokens = 5;
    opt.temperature = 0.0f;  // greedy
    opt.seed = 42;
    qinfer::serve::GenResult res;
    std::vector<std::int32_t> prompt = {1};  // BOS

    if (!backend.generate(prompt, opt, nullptr, res, err)) {
        std::fprintf(stderr, "generate failed: %s\n", err.c_str());
        return 1;
    }

    assert(res.n_gen == 5);
    assert(!res.tokens.empty());
    std::printf("PASS: native_backend generated %zu tokens\n", res.tokens.size());
    return 0;
}
