#include "serve/facade.hpp"

#include <algorithm>
#include <cmath>

namespace qinfer::serve {

bool Engine::open(const FacadeConfig& cfg, Engine& out, std::string& err) {
    // `out = Engine{}` 写不动：文件句柄（GgufFile/GgufTable）不可复制也不可移动。逐成员复位等价：
    // 其余成员在下面每一步都会被覆盖，能永久持有的只有这两个文件句柄与 KV 窗口（它们随 generate 重装）。
    out.g_.close();
    out.table_.close();
    out.wins_.clear();
    out.stores_.clear();
    out.cfg_ = cfg;
    if (cfg.model_path.empty()) { err = "模型路径为空"; return false; }
    if (!out.g_.open(cfg.model_path, err)) return false;
    if (!text::Tokenizer::from_gguf(out.g_, out.tokenizer_, err)) return false;
    if (!chat_template_from_gguf(out.g_, out.chat_template_, err)) return false;
    std::uint64_t eos = 0;
    if (out.g_.meta_u64("tokenizer.ggml.eos_token_id", eos)) {
        out.eos_ = static_cast<std::int32_t>(eos);
    } else {
        const std::int32_t sp = out.tokenizer_.special_id("<|im_end|>");
        if (sp < 0) { err = "找不到 eos token id（元数据与 special 词表都没有）"; return false; }
        out.eos_ = sp;
    }
    if (!build_token_embed(out.g_, cfg.n_embd, out.te_, err)) return false;
    out.step_cfg_ = dense::StepConfig{};
    out.step_cfg_.n_embd = cfg.n_embd;
    out.step_cfg_.hc = cfg.hc;
    out.step_cfg_.hc_lr = cfg.hc_lr;
    out.step_cfg_.n_layers = cfg.n_layers;
    out.step_cfg_.top_k = cfg.top_k;
    out.step_cfg_.gdn_head_dim = cfg.gdn_head_dim;
    out.step_cfg_.qsa_head_dim = cfg.qsa_head_dim;
    out.step_cfg_.n_rot = cfg.n_rot;
    out.step_cfg_.idx_dim = cfg.idx_dim;
    out.step_cfg_.idx_top_k = cfg.idx_top_k;
    out.step_cfg_.idx_block = cfg.idx_block;
    out.step_cfg_.cache_slots = cfg.cache_slots;
    out.step_cfg_.cache_byte_budget = cfg.cache_byte_budget;
    if (!cfg.ple_table.empty()) {
        if (!out.table_.open(cfg.ple_table, err)) {
            err = "PLE 表打不开：" + err;
            return false;
        }
        out.rows_ = storage::RowCache(4096);
        out.pspec_ = ple::PleSpec{};
        out.pw_ = ple::PleBuffers{};
        if (!ple::read_ple_spec(out.g_, /*layer=*/1, cfg.n_embd, cfg.hc, out.pspec_, err) ||
            !out.pspec_.present || !ple::load_ple_weights(out.g_, out.pspec_, out.pw_, err)) {
            err = "读第 1 层的 PLE 登记/权重失败：" + err;
            return false;
        }
        out.run_.layer = 1;
        out.run_.spec = out.pspec_;
        out.run_.w = out.pw_;
        out.run_.table = &out.src_;
        out.run_.cache = &out.rows_;
        out.run_.budget_bytes = cfg.ple_budget_bytes;
        out.step_cfg_.ple = &out.run_;
    }
    if (!build_step_model(out.g_, out.step_cfg_, out.m_, err)) return false;
    const std::string* nm = out.g_.meta_str("general.name");
    out.model_name_ =
        (nm != nullptr && !nm->empty()) ? *nm : std::filesystem::path(cfg.model_path).stem().string();
    return true;
}

// 每次 generate 一份：为每个 QSA 层建权威副本 + 常驻窗口，配额 = cfg_.kv_slots × 一页字节。
// 先 reserve 再落位 stores_/wins_（KvWindow.store 是指针，vector 扩容会让指针悬空）。
void Engine::attach_kv_windows(dense::StepScratch& s, std::uint64_t logical_blocks) {
    wins_.clear();
    stores_.clear();
    std::size_t n_qsa = 0;
    for (const auto& row : m_.map)
        if (row.kind == dense::LayerKind::kQsa) ++n_qsa;
    wins_.reserve(n_qsa);
    stores_.reserve(n_qsa);
    for (std::size_t l = 0; l < m_.map.size(); ++l) {
        if (m_.map[l].kind != dense::LayerKind::kQsa) continue;
        dense::QsaScratch& q = s.blocks[l].qsa;
        q.kv.n_pages = cfg_.kv_slots;      // 常驻槽位只给这么多（逻辑容量在副本里）
        q.kv.reset();
        stores_.emplace_back();
        stores_.back().init(q.kv.page_size, q.kv.n_head_kv, q.kv.head_dim, logical_blocks);
        wins_.emplace_back();              // 先落位再接线：窗口对象常驻在 wins_，scratch 挂的指针不悬空
        dense::KvWindow& w = wins_.back();
        w.page_bytes = 2 * q.kv.n_head_kv * q.kv.page_size * q.kv.head_dim * 2;
        w.quota_bytes = cfg_.kv_slots * w.page_bytes;
        w.store = &stores_.back();
        dense::attach_kv_window(q, w, logical_blocks);
    }
}

bool Engine::generate(const std::vector<std::int32_t>& prompt, const GenOptions& opt,
                      const std::function<bool(const std::string&)>& on_token, GenResult& res,
                      std::string& err) {
    if (opt.max_tokens == 0) { err = "max_tokens 必须为正"; return false; }
    SamplerParams sp;
    sp.temperature = opt.temperature;
    sp.top_p = opt.top_p;
    sp.seed = opt.seed;
    if (!sp.valid()) { err = "采样参数非法：temperature 非负、top_p 在 (0,1]"; return false; }
    // 一次 generate 一份全新会话：scratch、KV 窗口都在这次调用里装配。页数按 idx_block 个 cell 一页估，
    // 尾留 8 个 cell 余量（prefill 不重跑，故只此一次分配）。
    const std::uint64_t n_pages =
        (static_cast<std::uint64_t>(prompt.size()) + opt.max_tokens + 8 +
         step_cfg_.idx_block - 1) / step_cfg_.idx_block;
    dense::StepScratch s;
    if (!prepare_step_scratch(m_, n_pages, /*pos_base=*/0, s, err)) return false;
    if (cfg_.kv_slots > 0) attach_kv_windows(s, n_pages);
    Sampler sampler(opt.seed);
    res = GenResult{};
    res.n_prompt = prompt.size();
    for (std::size_t i = 0; i < prompt.size(); ++i) {
        if (!run_token_step(g_, m_, te_, static_cast<std::uint64_t>(prompt[i]), s, err)) return false;
    }
    std::size_t max_win = 0;
    for (const std::string& st : opt.stop) max_win = std::max(max_win, st.size());
    std::string text;
    for (;;) {
        if (s.head.logits.empty()) { err = "logits 为空，无法采样"; return false; }
        const std::uint32_t id = sampler.sample(s.head.logits.data(), s.head.logits.size(), sp);
        if (static_cast<std::int32_t>(id) == eos_) {
            res.finish = Finish::kStop;
            res.text = text;
            break;
        }
        std::string piece;
        if (!tokenizer_.token_bytes(static_cast<std::int32_t>(id), piece)) {
            piece = "\xEF\xBF\xBD";   // 防御：词表外 id 到不了这里，给 U+FFFD
        }
        res.tokens.push_back(static_cast<std::int32_t>(id));
        ++res.n_gen;
        const std::size_t old = text.size();
        text += piece;
        // stop 匹配：只看「结束点落在新 piece 内」的出现（跨 piece 边界也算），取最早结束的那次。
        std::size_t cut = std::string::npos;
        if (max_win > 0 && !piece.empty()) {
            const std::size_t win = old >= max_win ? old + 1 - max_win : 0;
            for (const std::string& st : opt.stop) {
                if (st.empty()) continue;
                std::size_t p = win;
                while ((p = text.find(st, p)) != std::string::npos) {
                    const std::size_t end = p + st.size();
                    if (end > old && (cut == std::string::npos || end < cut)) cut = end;
                    ++p;
                }
            }
        }
        if (cut != std::string::npos) {
            text.resize(cut);
            const std::string keep = piece.substr(0, cut > old ? cut - old : 0);
            if (!keep.empty() && on_token && !on_token(keep)) { err = "客户端中断"; return false; }
            res.finish = Finish::kStopString;
            res.text = text;
            break;
        }
        if (on_token && !on_token(piece)) { err = "客户端中断"; return false; }
        if (res.n_gen >= opt.max_tokens) {
            res.finish = Finish::kLength;
            res.text = text;
            break;
        }
        if (!run_token_step(g_, m_, te_, id, s, err)) return false;
    }
    return true;
}

std::uint64_t Engine::kv_misses() const {
    std::uint64_t v = 0;
    for (const auto& w : wins_) v += w.misses;
    return v;
}
std::uint64_t Engine::kv_evictions() const {
    std::uint64_t v = 0;
    for (const auto& w : wins_) v += w.evictions;
    return v;
}
std::uint64_t Engine::kv_bytes_moved() const {
    std::uint64_t v = 0;
    for (const auto& st : stores_) v += st.bytes_moved;
    return v;
}

}  // namespace qinfer::serve