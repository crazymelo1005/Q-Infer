// Per-inference session state container for the native inference engine.
// Wraps existing dense infrastructure: StepModel/StepScratch (weights + scratch buffers),
// build/prepare/run functions (model construction + per-layer computation), and KV cache management.
//
// The session loop iterates all 48 layers, dispatching dense computation to GPU (via DenseExec
// function pointers in StepConfig) and MoE to CPU. Phase 1 runs sequentially; the loop structure
// mirrors Strata's session_loop for future doorbell insertion in Phase 3.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "artifact/gguf_table.hpp"
#include "dense/step_loop.hpp"
#include "dense/token_embed.hpp"
#include "dense/block_layer.hpp"
#include "dense/model_head.hpp"
#include "dense/kv_window.hpp"
#include "dense/dense_exec.hpp"

namespace qinfer::serve {

struct FacadeConfig;  // forward decl

// Per-inference session: owns all mutable state for one generate() call.
// Modeled after Strata's SessionState but self-written.
class NativeSession {
public:
    NativeSession() = default;
    ~NativeSession();
    NativeSession(NativeSession&&) noexcept;
    NativeSession& operator=(NativeSession&&) noexcept;
    NativeSession(const NativeSession&) = delete;
    NativeSession& operator=(const NativeSession&) = delete;

    // Load model weights and build step model. Called once per backend lifetime.
    bool open_model(const artifact::GgufFile& gguf,
                    const dense::StepConfig& cfg,
                    std::string& err);

    // Prepare per-inference state: scratch buffers, KV windows, sampler.
    // Called once per generate() call.
    bool prepare(std::uint64_t n_pages, std::uint64_t pos_base,
                 std::uint64_t kv_slots, std::string& err);

    // Run one token through all 48 layers. Updates R (residual stack) and logits.
    // This is the new session loop — iterates layers, dispatches dense to GPU
    // (via DenseExec) and MoE to CPU.
    bool run_token(const artifact::GgufFile& gguf,
                   const dense::TokenEmbed& te,
                   std::uint64_t token, std::string& err);

    // Access logits after run_token
    const std::vector<float>& logits() const { return scratch_.head.logits; }

    // Position tracking
    std::uint64_t position() const { return pos_; }

private:
    dense::StepModel model_;
    dense::StepScratch scratch_;
    dense::StepConfig step_cfg_;
    std::vector<dense::KvWindow> wins_;
    std::vector<dense::KvHostStore> stores_;
    std::uint64_t pos_ = 0;
    bool model_ready_ = false;

    void attach_kv_windows(std::uint64_t logical_blocks);
};

}  // namespace qinfer::serve
