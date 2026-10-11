#include "dense/step_loop.hpp"

#include "dense/rope.hpp"

namespace qinfer::dense {

bool build_step_model(const artifact::GgufFile& gguf, const StepConfig& cfg, StepModel& out,
                      std::string& err) {
    err.clear();
    out = StepModel{};
    out.cfg = cfg;
    if (cfg.n_layers <= 0) {
        err = "层数必须为正";
        return false;
    }
    if (cfg.n_embd == 0 || cfg.hc == 0 || cfg.hc_lr == 0) {
        err = "n_embd / hc / hc_lr 有零";
        return false;
    }

    if (!scan_layers(gguf, cfg.n_layers, out.map, err)) return false;

    out.blocks.resize(static_cast<std::size_t>(cfg.n_layers));
    for (int l = 0; l < cfg.n_layers; ++l) {
        const LayerKind kind = out.map[static_cast<std::size_t>(l)].kind;
        if (kind == LayerKind::kUnknown) {
            err = "层 " + std::to_string(l) + " 判不出类型";
            return false;
        }
        const std::uint64_t head_dim = kind == LayerKind::kGdn ? cfg.gdn_head_dim : cfg.qsa_head_dim;
        if (head_dim == 0) {
            err = "层 " + std::to_string(l) + " 的头宽为 0（GDN/QSA 各自必给）";
            return false;
        }
        if (!build_block_layer(gguf, l, cfg.n_embd, cfg.hc, cfg.hc_lr, cfg.top_k, head_dim,
                               cfg.n_rot, cfg.idx_dim, cfg.idx_top_k, cfg.idx_block,
                               out.blocks[static_cast<std::size_t>(l)], err)) {
            err = "层 " + std::to_string(l) + " 建不起来：" + err;
            return false;
        }
    }
    return build_model_head(gguf, cfg.n_embd, cfg.hc, cfg.hc_lr, out.head, err);
}

bool prepare_step_scratch(const StepModel& model, std::uint64_t n_pages, std::uint64_t pos_base,
                          StepScratch& out, std::string& err) {
    err.clear();
    out = StepScratch{};
    if (n_pages == 0) {
        err = "页数必须为正";
        return false;
    }
    const std::size_t n = static_cast<std::size_t>(model.cfg.n_layers);
    out.R.assign(static_cast<std::size_t>(model.cfg.n_embd * model.cfg.hc), 0.0f);
    out.blocks.resize(n);
    for (std::size_t l = 0; l < n; ++l) {
        const BlockLayer& bl = model.blocks[l];
        if (bl.kind != LayerKind::kQsa) continue;
        const QsaLayer& ql = bl.qsa;
        QsaScratch& s = out.blocks[l].qsa;
        s.kv.page_size = 4;
        s.kv.n_head_kv = ql.sh.n_head_kv;
        s.kv.head_dim = ql.sh.head_dim;
        s.kv.n_pages = n_pages;
        s.kv.reset();
        s.kv.table.resize(static_cast<std::size_t>(n_pages));
        for (std::uint64_t p = 0; p < n_pages; ++p) {
            s.kv.table[static_cast<std::size_t>(p)] = static_cast<std::int32_t>(p);
        }
        s.pos_base = pos_base;
        const std::uint64_t max_pos = pos_base + n_pages * s.kv.page_size + 1;
        if (!build_rope_table(ql.n_rot, 1.0e7, max_pos, s.cos_tab, s.sin_tab, err)) return false;
    }
    out.ready = true;
    return true;
}

bool run_step_model(const artifact::GgufFile& gguf, const StepModel& model, const float* x_embd,
                    StepScratch& s, std::string& err) {
    err.clear();
    if (x_embd == nullptr) {
        err = "run_step_model 的 embedding 为空";
        return false;
    }
    if (!s.ready) {
        err = "StepScratch 未预备（先调 prepare_step_scratch）";
        return false;
    }
    const std::size_t n = static_cast<std::size_t>(model.cfg.n_layers);
    if (s.blocks.size() != n) {
        err = "scratch 的层数与模型不一致";
        return false;
    }

    // 残差栈的初值：同一个向量广播到 hc 条流（[S-61]）。
    hc_broadcast_init(x_embd, model.cfg.n_embd, model.cfg.hc, s.R.data());
    for (std::size_t l = 0; l < n; ++l) {
        if (!run_block_layer(gguf, model.blocks[l], s.R.data(), s.blocks[l], err)) {
            err = "层 " + std::to_string(l) + " 跑不动：" + err;
            return false;
        }
    }
    return run_model_head(gguf, model.head, s.R.data(), s.head, err);
}

}  // namespace qinfer::dense
