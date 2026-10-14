#include "serve/native_session.hpp"

#include <algorithm>

namespace qinfer::serve {

NativeSession::~NativeSession() = default;
NativeSession::NativeSession(NativeSession&&) noexcept = default;
NativeSession& NativeSession::operator=(NativeSession&&) noexcept = default;

bool NativeSession::open_model(const artifact::GgufFile& gguf,
                                const dense::StepConfig& cfg,
                                std::string& err) {
    step_cfg_ = cfg;
    if (!dense::build_step_model(gguf, step_cfg_, model_, err)) return false;
    model_ready_ = true;
    return true;
}

bool NativeSession::prepare(std::uint64_t n_pages, std::uint64_t pos_base,
                             std::uint64_t kv_slots, std::string& err) {
    if (!model_ready_) { err = "model not loaded"; return false; }
    if (!dense::prepare_step_scratch(model_, n_pages, pos_base, scratch_, err))
        return false;
    pos_ = 0;
    // Attach KV windows if slots > 0
    if (kv_slots > 0) {
        step_cfg_.cache_slots = kv_slots;
        attach_kv_windows(n_pages);
    }
    return true;
}

void NativeSession::attach_kv_windows(std::uint64_t logical_blocks) {
    wins_.clear();
    stores_.clear();
    std::size_t n_qsa = 0;
    for (const auto& row : model_.map)
        if (row.kind == dense::LayerKind::kQsa) ++n_qsa;
    wins_.reserve(n_qsa);
    stores_.reserve(n_qsa);
    for (std::size_t l = 0; l < model_.map.size(); ++l) {
        if (model_.map[l].kind != dense::LayerKind::kQsa) continue;
        dense::QsaScratch& q = scratch_.blocks[l].qsa;
        q.kv.n_pages = step_cfg_.cache_slots;
        q.dev_kv_pages = logical_blocks;
        q.kv.reset();
        stores_.emplace_back();
        stores_.back().init(q.kv.page_size, q.kv.n_head_kv,
                            q.kv.head_dim, logical_blocks);
        wins_.emplace_back();
        dense::KvWindow& w = wins_.back();
        w.page_bytes = 2 * q.kv.n_head_kv * q.kv.page_size *
                       q.kv.head_dim * 2;
        w.quota_bytes = step_cfg_.cache_slots * w.page_bytes;
        w.store = &stores_.back();
        dense::attach_kv_window(q, w, logical_blocks);
    }
}

bool NativeSession::run_token(const artifact::GgufFile& gguf,
                               const dense::TokenEmbed& te,
                               std::uint64_t token, std::string& err) {
    // New session loop: iterate all 48 layers, dispatching dense to GPU
    // (via DenseExec in StepConfig) and MoE to CPU.
    //
    // Phase 1: sequential execution. The loop structure mirrors Strata's
    // session_loop for future doorbell insertion:
    //   for each layer: block_layer (GPU dense + CPU MoE)
    // Phase 3 will split block_layer into pre/post with doorbell between.

    err.clear();
    scratch_.embd.assign(te.n_embd, 0.0f);
    if (!dense::read_token_embedding(gguf, te, token,
                                      scratch_.embd.data(), err))
        return false;

    // Initialize residual stack from embedding
    dense::hc_broadcast_init(scratch_.embd.data(), te.n_embd,
                              step_cfg_.hc, scratch_.R.data());

    // Layer loop
    for (std::size_t l = 0; l < model_.blocks.size(); ++l) {
        if (!dense::run_block_layer(gguf, model_.blocks[l],
                                     scratch_.R.data(),
                                     scratch_.blocks[l], err))
            return false;
    }

    // LM head: gr_read + output projection → logits
    if (!dense::run_model_head(gguf, model_.head, scratch_.R.data(),
                                scratch_.head, err))
        return false;

    ++pos_;
    return true;
}

void NativeSession::reset() {
    pos_ = 0;
    // Note: scratch_ and KV state are reset by prepare() at the start of each generate() call.
    // reset() just clears the position counter for session reuse.
}

bool NativeSession::prefill_chunked(const artifact::GgufFile& gguf,
                                     const dense::TokenEmbed& te,
                                     const std::vector<std::int32_t>& tokens,
                                     std::size_t chunk_size,
                                     const std::function<void(std::size_t)>& on_progress,
                                     std::string& err) {
    if (chunk_size == 0) {
        err = "chunk_size must be > 0";
        return false;
    }

    const std::size_t n = tokens.size();
    for (std::size_t i = 0; i < n; i += chunk_size) {
        const std::size_t end = std::min(i + chunk_size, n);
        for (std::size_t j = i; j < end; ++j) {
            if (!run_token(gguf, te, static_cast<std::uint64_t>(tokens[j]), err)) {
                return false;
            }
        }
        if (on_progress) {
            on_progress(end);
        }
    }
    return true;
}

}  // namespace qinfer::serve
