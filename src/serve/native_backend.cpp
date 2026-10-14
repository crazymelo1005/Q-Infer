#include "serve/native_backend.hpp"

#include <algorithm>
#include <filesystem>

#include "dense/step_loop.hpp"
#include "serve/prompt.hpp"
#include "serve/sampler.hpp"

namespace qinfer::serve {

NativeBackend::~NativeBackend() = default;

// build_step_config removed: dense_exec wiring is handled by main.cpp via
// FacadeConfig (same as Engine). NativeBackend::open() copies cfg into cfg_,
// so cfg_.dense_exec already reflects --gpu (or stays empty for CPU-only).

bool NativeBackend::open(const FacadeConfig& cfg, NativeBackend& out,
                          std::string& err) {
    out.gguf_.close();
    out.cfg_ = cfg;
    if (cfg.model_path.empty()) { err = "empty model path"; return false; }
    if (!out.gguf_.open(cfg.model_path, err)) return false;
    if (!text::Tokenizer::from_gguf(out.gguf_, out.tokenizer_, err)) return false;
    if (!chat_template_from_gguf(out.gguf_, out.chat_template_, err)) return false;

    // EOS token
    std::uint64_t eos = 0;
    if (out.gguf_.meta_u64("tokenizer.ggml.eos_token_id", eos)) {
        out.eos_ = static_cast<std::int32_t>(eos);
    } else {
        const std::int32_t sp = out.tokenizer_.special_id("<|im_end|>");
        if (sp < 0) { err = "no eos token"; return false; }
        out.eos_ = sp;
    }

    // Token embed
    if (!dense::build_token_embed(out.gguf_, cfg.n_embd, out.te_, err)) return false;

    // Load model (dense_exec already populated by main.cpp via cfg if --gpu)
    dense::StepConfig sc{};
    sc.n_embd = cfg.n_embd;
    sc.hc = cfg.hc;
    sc.hc_lr = cfg.hc_lr;
    sc.n_layers = cfg.n_layers;
    sc.top_k = cfg.top_k;
    sc.gdn_head_dim = cfg.gdn_head_dim;
    sc.qsa_head_dim = cfg.qsa_head_dim;
    sc.n_rot = cfg.n_rot;
    sc.idx_dim = cfg.idx_dim;
    sc.idx_top_k = cfg.idx_top_k;
    sc.idx_block = cfg.idx_block;
    sc.cache_slots = cfg.cache_slots;
    sc.cache_byte_budget = cfg.cache_byte_budget;
    sc.kv_resident_tokens = cfg.kv_resident;
    if (out.cfg_.dense_exec.run_gdn || out.cfg_.dense_exec.run_qsa)
        sc.dense_exec = &out.cfg_.dense_exec;

    if (!out.session_.open_model(out.gguf_, sc, err)) return false;

    // Model name
    const std::string* nm = out.gguf_.meta_str("general.name");
    out.model_name_ = (nm && !nm->empty()) ? *nm
        : std::filesystem::path(cfg.model_path).stem().string();
    return true;
}

bool NativeBackend::generate(const std::vector<std::int32_t>& prompt,
                              const GenOptions& opt,
                              const std::function<bool(const std::string&)>& on_token,
                              GenResult& res, std::string& err) {
    if (opt.max_tokens == 0) { err = "max_tokens must be positive"; return false; }
    SamplerParams sp;
    sp.temperature = opt.temperature;
    sp.top_p = opt.top_p;
    sp.seed = opt.seed;
    if (!sp.valid()) { err = "invalid sampler params"; return false; }

    // Compute page count
    const std::uint64_t n_pages =
        (static_cast<std::uint64_t>(prompt.size()) + opt.max_tokens + 8 +
         cfg_.idx_block - 1) / cfg_.idx_block;

    // Reuse session (model already loaded in open()). Just re-prepare scratch.
    if (!session_.prepare(n_pages, 0, cfg_.kv_slots, err)) return false;

    Sampler sampler(opt.seed);
    res = GenResult{};
    res.n_prompt = prompt.size();

    // Prefill: run all prompt tokens
    for (std::size_t i = 0; i < prompt.size(); ++i) {
        if (!session_.run_token(gguf_, te_, static_cast<std::uint64_t>(prompt[i]), err))
            return false;
    }

    // Generate loop
    std::size_t max_win = 0;
    for (const std::string& st : opt.stop)
        max_win = std::max(max_win, st.size());
    std::string text;

    for (;;) {
        const auto& logits = session_.logits();
        if (logits.empty()) { err = "empty logits"; return false; }
        const std::uint32_t id = sampler.sample(logits.data(), logits.size(), sp);
        if (static_cast<std::int32_t>(id) == eos_) {
            res.finish = Finish::kStop;
            res.text = text;
            break;
        }
        std::string piece;
        if (!tokenizer_.token_bytes(static_cast<std::int32_t>(id), piece))
            piece = "\xEF\xBF\xBD";
        res.tokens.push_back(static_cast<std::int32_t>(id));
        ++res.n_gen;
        const std::size_t old = text.size();
        text += piece;
        // Stop string detection
        std::size_t cut = std::string::npos;
        if (max_win > 0 && !piece.empty()) {
            const std::size_t win = old >= max_win ? old + 1 - max_win : 0;
            for (const std::string& st : opt.stop) {
                if (st.empty()) continue;
                std::size_t p = win;
                while ((p = text.find(st, p)) != std::string::npos) {
                    const std::size_t end = p + st.size();
                    if (end > old && (cut == std::string::npos || end < cut))
                        cut = end;
                    ++p;
                }
            }
        }
        if (cut != std::string::npos) {
            text.resize(cut);
            const std::string keep = piece.substr(0, cut > old ? cut - old : 0);
            if (!keep.empty() && on_token && !on_token(keep)) {
                err = "client stop"; return false;
            }
            res.finish = Finish::kStopString;
            res.text = text;
            break;
        }
        if (on_token && !on_token(piece)) { err = "client abort"; return false; }
        if (res.n_gen >= opt.max_tokens) {
            res.finish = Finish::kLength;
            res.text = text;
            break;
        }
        if (!session_.run_token(gguf_, te_, id, err)) return false;
    }
    return true;
}

std::uint64_t NativeBackend::kv_misses() const {
    // NativeSession does not expose KV window stats yet (Phase 1).
    // Return 0 for now; Phase 3 will add doorbell-based stats.
    return 0;
}

std::uint64_t NativeBackend::kv_evictions() const {
    return 0;
}

std::uint64_t NativeBackend::kv_bytes_moved() const {
    return 0;
}

}  // namespace qinfer::serve
