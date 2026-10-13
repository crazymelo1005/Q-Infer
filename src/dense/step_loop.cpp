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
        // 专家缓存的配置逐层套用（engine §5）；`cache_slots == 0` 时保持按需直读。
        experts::MoeLayer& ml = out.blocks[static_cast<std::size_t>(l)].moe;
        ml.cache_slots = cfg.cache_slots;
        ml.cache_ways = cfg.cache_ways;
        ml.cache_byte_budget = cfg.cache_byte_budget;
        ml.cache_policy = cfg.cache_policy;
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
    out.phases.assign(n, LayerPhases{});  // 每步的相位耗时（interfaces §7 的「每步耗时」按层拆）
    // 交出（interfaces §8）：把调用方按层建好的那一条挂到该层的 scratch 上，跑每一步时逐层生效。
    if (model.cfg.offload_layers != nullptr) {
        if (model.cfg.offload_layers->size() != n) {
            err = "交出表给的层数与模型的层数不一致";
            return false;
        }
        for (std::size_t l = 0; l < n; ++l) {
            out.blocks[l].offload = &(*model.cfg.offload_layers)[l];
        }
    }
    // dense 设备执行器（阶段 A dense 接线）：非空则逐层生效（GDN/QSA 整层走 *_cuda）。
    for (std::size_t l = 0; l < n; ++l) {
        out.blocks[l].dense = model.cfg.dense_exec;
        // Device KV (stage B): QSA layers get the exec pointer; the pool itself
        // is reserved lazily on first entry (see run_qsa_layer_cuda).
        out.blocks[l].qsa.dev_kv =
            model.cfg.dense_exec != nullptr ? &model.cfg.dense_exec->kv : nullptr;
    }
    // PLE（[S-63]）：配置只挂到它自己那一层，别的层保持为空——于是「挂错层」不可能悄悄发生。
    if (model.cfg.ple != nullptr) {
        const PleRun& pr = *model.cfg.ple;
        if (!pr.ready()) {
            err = "PLE 的配置不完整（层号 / 六张张力 / 表行来源 / 行缓存）";
            return false;
        }
        if (pr.layer >= static_cast<int>(n)) {
            err = "PLE 挂在第 " + std::to_string(pr.layer) + " 层，但模型只有 " + std::to_string(n) +
                  " 层";
            return false;
        }
        const std::uint64_t n_embd = model.cfg.n_embd, hc = model.cfg.hc;
        if (pr.w.key_bf16.size() != static_cast<std::size_t>(n_embd * n_embd * hc) ||
            pr.w.value_bf16.size() != static_cast<std::size_t>(n_embd * n_embd) ||
            pr.w.norm_key.size() != static_cast<std::size_t>(n_embd * hc)) {
            err = "PLE 的权重与模型的几何（n_embd × hc）对不上";
            return false;
        }
        out.ple.hist.assign(static_cast<std::size_t>(ple::kHistRows) * n_embd * hc, 0.0f);
        out.ple.emb.assign(static_cast<std::size_t>(n_embd), 0.0f);
        out.ple.prev[0] = kernels::kTokenNull;
        out.ple.prev[1] = kernels::kTokenNull;
        out.ple.has_token = false;
        out.blocks[static_cast<std::size_t>(pr.layer)].ple = model.cfg.ple;
        out.blocks[static_cast<std::size_t>(pr.layer)].ple_state = &out.ple;
    }
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

LayerPhases sum_phases(const std::vector<LayerPhases>& phases) {
    LayerPhases t;
    for (const LayerPhases& p : phases) {
        t.load_ms += p.load_ms;
        t.gr_read_ms += p.gr_read_ms;
        t.gr_split.norm_ms += p.gr_split.norm_ms;
        t.gr_split.lo_ms += p.gr_split.lo_ms;
        t.gr_split.mixed_ms += p.gr_split.mixed_ms;
        t.gr_split.inject_ms += p.gr_split.inject_ms;
        t.attn_ms += p.attn_ms;
        t.attn.proj_ms += p.attn.proj_ms;
        t.attn.prep_ms += p.attn.prep_ms;
        t.attn.rec_ms += p.attn.rec_ms;
        t.attn.close_ms += p.attn.close_ms;
        t.attn.kv_ms += p.attn.kv_ms;
        t.attn.idx_ms += p.attn.idx_ms;
        t.attn.attend_ms += p.attn.attend_ms;
        t.moe_ms += p.moe_ms;
        t.moe_io_ms += p.moe_io_ms;
        t.moe_route_ms += p.moe_route_ms;
        t.moe_expert_ms += p.moe_expert_ms;
        t.moe_shared_ms += p.moe_shared_ms;
        t.moe_off_fetch_ms += p.moe_off_fetch_ms;
        t.moe_off_worker_ms += p.moe_off_worker_ms;
        t.moe_off_merge_ms += p.moe_off_merge_ms;
        t.gr_write_ms += p.gr_write_ms;
    }
    return t;
}

bool run_step_model(const artifact::GgufFile& gguf, const StepModel& model, const float* x_embd,
                    StepScratch& s, std::string& err, const scheduling::StepDemand* demand) {
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

    // 相位计时清零：run_block_layer 是累加的，清零才得到「本步」的数。
    std::fill(s.phases.begin(), s.phases.end(), LayerPhases{});

    // 残差栈的初值：同一个向量广播到 hc 条流（[S-61]）。
    hc_broadcast_init(x_embd, model.cfg.n_embd, model.cfg.hc, s.R.data());
    experts::StepRecord rec;
    rec.step = demand != nullptr ? demand->step : 0;
    for (std::size_t l = 0; l < n; ++l) {
        // 缓存侧的每步增量：按层取「本步前后」的快照差（首次调用时源还没建，前快照记全 0，
        // 于是这一层的首次填充就正确地记成这一步的未命中）。
        const experts::MoeScratch& ms = s.blocks[l].moe;
        const experts::CacheSnapshot prev = ms.source != nullptr
                                                ? experts::CacheSnapshot::of(*ms.source)
                                                : experts::CacheSnapshot{};
        LayerPhases* ph =
            l < s.phases.size() ? &s.phases[l] : nullptr;  // 每步覆写，故是「本步」的数
        if (!run_block_layer(gguf, model.blocks[l], s.R.data(), s.blocks[l], err, ph)) {
            err = "层 " + std::to_string(l) + " 跑不动：" + err;
            return false;
        }
        if (ms.source != nullptr) {
            const experts::CacheSnapshot now = experts::CacheSnapshot::of(*ms.source);
            const auto inc = [](std::uint64_t a, std::uint64_t b) { return a > b ? a - b : 0; };
            rec.cache_hits += inc(now.hits, prev.hits);
            rec.cache_misses += inc(now.misses, prev.misses);
            rec.cache_evictions += inc(now.replaced, prev.replaced);
            rec.cache_blocked += inc(now.blocked, prev.blocked);
            rec.cache_transient += inc(now.transient_reads, prev.transient_reads);
            rec.residents += now.residents;
            rec.bytes_resident += now.bytes_resident;
            rec.page_entries += now.page_entries;
        }
    }
    // 预算侧：给了需求且画像里有带宽才走仲裁器；否则那几项留 0（表示这一步没做仲裁）。
    if (demand != nullptr && model.cfg.profile.pcie_h2d_gbps > 0.0) {
        const std::uint64_t budget = scheduling::step_budget_bytes(model.cfg.profile);
        const scheduling::Result r = scheduling::arbitrate(budget, scheduling::build_requests(*demand));
        const scheduling::StepObservation ob = scheduling::observe(r, budget, demand->step);
        rec.budget_bytes = ob.budget_bytes;
        rec.admitted_bytes = ob.admitted_bytes;
        rec.overspend_bytes = ob.overspend_bytes;
        rec.late_requests = ob.late_requests;
        rec.late_bytes = ob.late_bytes;
        rec.budget_used = ob.budget_used;
    }
    const std::uint64_t accesses = rec.cache_hits + rec.cache_misses;
    rec.hit_rate = accesses > 0 ? static_cast<double>(rec.cache_hits) / static_cast<double>(accesses) : 0.0;
    s.last_record = rec;
    s.has_record = true;

    return run_model_head(gguf, model.head, s.R.data(), s.head, err);
}

bool run_token_step(const artifact::GgufFile& gguf, const StepModel& model, const TokenEmbed& e,
                    std::uint64_t token, StepScratch& s, std::string& err) {
    err.clear();
    if (e.n_embd != model.cfg.n_embd) {
        err = "TokenEmbed 的行宽 " + std::to_string(e.n_embd) + " 与模型的 n_embd " +
              std::to_string(model.cfg.n_embd) + " 不一致";
        return false;
    }
    s.embd.assign(static_cast<std::size_t>(e.n_embd), 0.0f);
    if (!read_token_embedding(gguf, e, token, s.embd.data(), err)) return false;

    // PLE 的表行向量：用**当前 token 与它前面两个**（会话状态里的 `prev`，最老在前；越界处记 NULL，
    // 与整段重算那条的 `build_prev` 同一口径）。行缓存与字节预算沿用配置里那一份。
    // 表行向量固定 2560 宽（16 头 × 160，见 [S-63] 的几何），故这一条只在 n_embd 与它相等时可用
    // ——真模型正是 2560；不相等就明确失败，不截断也不补零。
    const bool with_ple = model.cfg.ple != nullptr;
    if (with_ple && model.cfg.ple->can_stage()) {
        if (e.n_embd != static_cast<std::uint64_t>(ple::kVectorDim)) {
            err = "PLE 的表行向量固定 " + std::to_string(ple::kVectorDim) + " 宽，与这个模型的 n_embd " +
                  std::to_string(e.n_embd) + " 不一致";
            return false;
        }
        ple::PathStats stats;
        if (!ple::ple_vector_with_prev(*model.cfg.ple->table, static_cast<std::int32_t>(token),
                                       s.ple.prev[0], s.ple.prev[1], *model.cfg.ple->cache,
                                       model.cfg.ple->budget_bytes, s.ple.emb.data(), stats, err)) {
            err = "取 PLE 表行失败：" + err;
            return false;
        }
        s.ple.has_token = true;
    }
    if (!run_step_model(gguf, model, s.embd.data(), s, err)) return false;
    // prev 在这一步跑完之后才推进：本步的哈希用的必须是上一步结束时的那两个。
    if (with_ple && model.cfg.ple->can_stage()) {
        s.ple.prev[0] = s.ple.prev[1];
        s.ple.prev[1] = static_cast<std::int32_t>(token);
    }
    return true;
}

}  // namespace qinfer::dense
