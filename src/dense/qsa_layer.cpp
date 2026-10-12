#include "dense/qsa_layer.hpp"

#include "dense/kv_window.hpp"
#include "dense/norm.hpp"
#include "dense/qsa_attend.hpp"
#include "dense/qsa_index.hpp"
#include "dense/qsa_indexer.hpp"
#include "dense/qsa_qk_prep.hpp"
#include "experts/ffn.hpp"
#include "kernels/bf16.hpp"
#include "kernels/fp16.hpp"

#include <chrono>

#include <cstring>

namespace qinfer::dense {

namespace {

const artifact::GgufTensorInfo* need(const artifact::GgufFile& g, const std::string& name,
                                     std::string& err) {
    const artifact::GgufTensorInfo* t = g.find(name);
    if (t == nullptr) err = name + " 不存在";
    return t;
}

// 一张 2 维张力的规格：档取它自己的类型码，几何取 [cols, rows]（GGML 的 ne0 是输入维）。
// `allow_dense_bf16`：索引器的两个投影在这份 artifact 里是 **BF16**（[S-54]），而 BF16 不是「量化权重」
// ——它不在这份按档分派的分派表里（路由器的 bf16 点积走的是另一条路），故这两个张力只核几何、不查内核。
bool spec_of(const artifact::GgufTensorInfo& t, const char* what, experts::MatrixSpec& out,
             std::string& err, bool allow_dense_bf16 = false) {
    if (t.dims.size() != 2) {
        err = std::string(what) + " 不是 2 维（QSA 的投影没有专家维）";
        return false;
    }
    out.format = experts::format_from_type_code(t.type);
    out.experts = 1;
    out.cols = t.dims[0];
    out.rows = t.dims[1];
    if (!out.geometry_ok()) {
        err = std::string(what) + " 的几何不成立：cols " + std::to_string(out.cols) + " 不是块元素数 " +
              std::to_string(experts::block_elems(out.format)) + " 的整数倍";
        return false;
    }
    const bool dense_bf16 = allow_dense_bf16 && out.format == experts::Format::kBf16;
    if (!dense_bf16 && !experts::has_kernel(out.format)) {
        err = std::string(what) + " 的档 " + experts::format_name(out.format) + " 没有内核";
        return false;
    }
    return true;
}

// 一维 F32 的 norm gamma：长度必须等于期望值。
bool norm_of(const artifact::GgufTensorInfo& t, const char* what, std::uint64_t want,
             std::string& err) {
    if (t.dims.size() != 1 || t.elements() != want) {
        err = std::string(what) + " 的形状不是 [" + std::to_string(want) + "]";
        return false;
    }
    if (t.type != 0) {  // F32
        err = std::string(what) + " 的类型码不是 F32（QSA 的 norm gamma 是 F32）";
        return false;
    }
    return true;
}

}  // namespace

bool build_qsa_layer(const artifact::GgufFile& gguf, int layer, std::uint64_t n_embd,
                     std::uint64_t head_dim, std::uint64_t n_rot, std::uint64_t idx_dim,
                     std::uint64_t idx_top_k, std::uint64_t idx_block, QsaLayer& out,
                     std::string& err) {
    err.clear();
    out = QsaLayer{};
    out.layer = layer;
    const std::string b = "blk." + std::to_string(layer) + ".";

    // GDN 层有 attn_qkv、QSA 层没有——用它把两种层分开（报错要能直接看出是哪一种层）。
    if (gguf.find(b + "attn_qkv.weight") != nullptr) {
        err = "层 " + std::to_string(layer) + " 不是 QSA 层（它有 attn_qkv.weight，是 GDN 层）";
        return false;
    }
    out.t_q = need(gguf, b + "attn_q.weight", err);
    if (out.t_q == nullptr) return false;
    out.t_k = need(gguf, b + "attn_k.weight", err);
    if (out.t_k == nullptr) return false;
    out.t_v = need(gguf, b + "attn_v.weight", err);
    if (out.t_v == nullptr) return false;
    out.t_out = need(gguf, b + "attn_output.weight", err);
    if (out.t_out == nullptr) return false;
    out.t_q_norm = need(gguf, b + "attn_q_norm.weight", err);
    if (out.t_q_norm == nullptr) return false;
    out.t_k_norm = need(gguf, b + "attn_k_norm.weight", err);
    if (out.t_k_norm == nullptr) return false;
    out.t_idx_q = need(gguf, b + "indexer.q_proj.weight", err);
    if (out.t_idx_q == nullptr) return false;
    out.t_idx_k = need(gguf, b + "indexer.k_proj.weight", err);
    if (out.t_idx_k == nullptr) return false;
    out.t_idx_q_norm = need(gguf, b + "indexer.q_norm.weight", err);
    if (out.t_idx_q_norm == nullptr) return false;
    out.t_idx_k_norm = need(gguf, b + "indexer.k_norm.weight", err);
    if (out.t_idx_k_norm == nullptr) return false;

    if (!spec_of(*out.t_q, "attn_q", out.q, err)) return false;
    if (!spec_of(*out.t_k, "attn_k", out.k, err)) return false;
    if (!spec_of(*out.t_v, "attn_v", out.v, err)) return false;
    if (!spec_of(*out.t_out, "attn_output", out.out, err)) return false;
    if (!spec_of(*out.t_idx_q, "indexer.q_proj", out.idx_q, err, /*allow_dense_bf16=*/true))
        return false;
    if (!spec_of(*out.t_idx_k, "indexer.k_proj", out.idx_k, err, /*allow_dense_bf16=*/true))
        return false;

    if (head_dim == 0 || n_rot == 0 || n_rot > head_dim || n_rot % 2 != 0 || idx_dim == 0 ||
        n_rot > idx_dim) {
        err = "head_dim / n_rot / idx_dim 不合法（n_rot 同时受 head_dim 与 idx_dim 约束）";
        return false;
    }
    // 四个投影的输入维都是 n_embd。
    if (out.q.cols != n_embd || out.k.cols != n_embd || out.v.cols != n_embd ||
        out.idx_q.cols != n_embd || out.idx_k.cols != n_embd) {
        err = "某个投影的输入维不是 n_embd";
        return false;
    }
    // attn_q 的输出是 2·n_head·head_dim（q 与门融合）；attn_output 的输入是 n_head·head_dim。
    if (out.q.rows % (2 * head_dim) != 0) {
        err = "attn_q 的输出维 " + std::to_string(out.q.rows) + " 不是 2·head_dim 的整数倍";
        return false;
    }
    const std::uint64_t n_head = out.q.rows / (2 * head_dim);
    if (out.out.cols != n_head * head_dim || out.out.rows != n_embd) {
        err = "attn_output 的几何与 attn_q 推出的 n_head 不自洽";
        return false;
    }
    if (out.v.rows % head_dim != 0) {
        err = "attn_v 的输出维不是 head_dim 的整数倍";
        return false;
    }
    const std::uint64_t n_head_kv = out.v.rows / head_dim;
    if (n_head_kv == 0 || n_head % n_head_kv != 0) {
        err = "n_head 不是 n_head_kv 的整数倍";
        return false;
    }
    // 两个索引器投影：q 的输出 = idx_n_head·idx_dim、k 的输出 = idx_dim。
    if (out.idx_q.rows % idx_dim != 0 || out.idx_k.rows != idx_dim) {
        err = "索引器投影的输出维与 idx_dim 对不上";
        return false;
    }
    const std::uint64_t idx_n_head = out.idx_q.rows / idx_dim;
    // 四个 norm 的长度。
    if (!norm_of(*out.t_q_norm, "attn_q_norm", head_dim, err)) return false;
    if (!norm_of(*out.t_k_norm, "attn_k_norm", head_dim, err)) return false;
    if (!norm_of(*out.t_idx_q_norm, "indexer.q_norm", idx_dim, err)) return false;
    if (!norm_of(*out.t_idx_k_norm, "indexer.k_norm", idx_dim, err)) return false;

    out.sh = QsaShapes{idx_top_k, idx_block, head_dim, n_head, n_head_kv};
    out.n_rot = n_rot;
    out.idx_dim = idx_dim;
    out.idx_n_head = idx_n_head;
    out.idx_dense_bf16 = out.idx_q.format == experts::Format::kBf16;
    return true;
}

namespace {

// 读一张 F32 一维张力（norm gamma）到 float。
bool load_f32_1d(const artifact::GgufFile& g, const artifact::GgufTensorInfo& t,
                 std::vector<float>& dst, std::string& err) {
    dst.resize(static_cast<std::size_t>(t.elements()));
    if (!g.read_at(t.offset, reinterpret_cast<std::uint8_t*>(dst.data()), dst.size() * 4)) {
        err = t.name + " 读失败";
        return false;
    }
    return true;
}

// 读一张 2 维 BF16 张力到 bf16 位型。
bool load_bf16_2d(const artifact::GgufFile& g, const artifact::GgufTensorInfo& t,
                  std::vector<std::uint16_t>& dst, std::string& err) {
    dst.resize(static_cast<std::size_t>(t.elements()));
    if (!g.read_at(t.offset, reinterpret_cast<std::uint8_t*>(dst.data()), dst.size() * 2)) {
        err = t.name + " 读失败";
        return false;
    }
    return true;
}

// 读一张量化投影的字节，并按它自己的档做一次矩阵乘。
bool project(const artifact::GgufFile& g, const artifact::GgufTensorInfo& t,
             const experts::MatrixSpec& spec, const float* x, std::vector<std::uint8_t>& w,
             bool reload, experts::FfnScratch& sc, float* out, std::string& err) {
    if (reload) {
        w.resize(static_cast<std::size_t>(spec.rows * spec.row_bytes()));
        if (!g.read_at(t.offset, w.data(), w.size())) {
            err = t.name + " 读失败";
            return false;
        }
    }
    if (!experts::quantize_input(spec, x, sc, err)) return false;
    return experts::run_gemv(spec, w.data(), sc, out, err);
}

// 把逻辑页搬入常驻窗口（非空时）。逻辑页表由窗口自己维护（准入写、驱逐写回 -1）；没挂窗口时表由
// 调用方预填（恒等映射），这里是空操作。
bool touch_page(QsaScratch& s, std::uint64_t logical, std::string& err) {
    if (s.window == nullptr) return true;
    std::uint64_t phys = 0;
    return s.window->ensure(logical, phys, err);
}

}  // namespace

bool run_qsa_layer(const artifact::GgufFile& gguf, const QsaLayer& layer, const float* x, float* out,
                   QsaScratch& s, std::string& err, AttnPhases* phases) {
    err.clear();
    if (x == nullptr || out == nullptr) {
        err = "run_qsa_layer 的输入输出指针为空";
        return false;
    }
    // 子步计时（给了 phases 才累加）：proj / prep / kv / idx / attend，见 dense/attn_phases.hpp。
    const auto clk = [] { return std::chrono::steady_clock::now(); };
    const auto ms = [](std::chrono::steady_clock::time_point a,
                       std::chrono::steady_clock::time_point b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };
    double proj_acc = 0.0;
    const std::uint64_t N = layer.q.cols;
    const std::uint64_t hd = layer.sh.head_dim;
    const std::uint64_t idx_dim = layer.idx_dim;

    // 0. 一层一次：把四个 norm 与索引器的两个 BF16 投影读进来（按层号判，见 QsaScratch::w_layer）。
    const bool reload = s.w_layer != layer.layer;
    if (reload) {
        if (!load_f32_1d(gguf, *layer.t_q_norm, s.q_norm_w, err) ||
            !load_f32_1d(gguf, *layer.t_k_norm, s.k_norm_w, err) ||
            !load_f32_1d(gguf, *layer.t_idx_q_norm, s.idx_q_norm_w, err) ||
            !load_f32_1d(gguf, *layer.t_idx_k_norm, s.idx_k_norm_w, err) ||
            !load_bf16_2d(gguf, *layer.t_idx_q, s.idx_q_w, err) ||
            !load_bf16_2d(gguf, *layer.t_idx_k, s.idx_k_w, err)) {
            return false;
        }
        if (s.cos_tab.empty()) {
            err = "QSA 的 rope 表没配（cos_tab/sin_tab 由调用方建好，theta 用 1e7，见 S-57）";
            return false;
        }
    }

    // 1. 四个注意力投影。
    const auto p_proj = clk();
    s.q_full.assign(static_cast<std::size_t>(layer.q.rows), 0.0f);
    s.k_raw.assign(static_cast<std::size_t>(layer.k.rows), 0.0f);
    s.v_raw.assign(static_cast<std::size_t>(layer.v.rows), 0.0f);
    if (!project(gguf, *layer.t_q, layer.q, x, s.w_q, reload, s.gemv, s.q_full.data(), err) ||
        !project(gguf, *layer.t_k, layer.k, x, s.w_k, reload, s.gemv, s.k_raw.data(), err) ||
        !project(gguf, *layer.t_v, layer.v, x, s.w_v, reload, s.gemv, s.v_raw.data(), err)) {
        return false;
    }
    proj_acc += ms(p_proj, clk());

    // 2. q/门拆分 + q/k 按头归一 + 部分 NEOX rope。
    const auto p_prep = clk();
    s.q_ready.assign(static_cast<std::size_t>(layer.sh.n_head * hd), 0.0f);
    s.k_ready.assign(static_cast<std::size_t>(layer.sh.n_head_kv * hd), 0.0f);
    s.gate.assign(s.q_ready.size(), 0.0f);
    const QsaQkWeights qkw{s.q_norm_w.data(), s.k_norm_w.data(), 1e-6f};
    if (!qsa_qk_prep(layer.sh.n_head, layer.sh.n_head_kv, hd, layer.n_rot, s.pos, s.q_full.data(),
                     s.k_raw.data(), qkw, s.cos_tab, s.sin_tab, s.q_ready.data(),
                     s.k_ready.data(), s.gate.data(), err)) {
        return false;
    }

    // 3. KV 写入（fp16）：K 存的是**归一+rope 之后**的那一份（attend 拿它与同样处理过的 q 比），V 原样。
    if (phases != nullptr) phases->prep_ms += ms(p_prep, clk());
    const auto p_kv = clk();
    s.k_f16.assign(s.k_ready.size(), 0);
    s.v_f16.assign(s.v_raw.size(), 0);
    for (std::size_t i = 0; i < s.k_ready.size(); ++i) s.k_f16[i] = kernels::f32_to_f16_bits(s.k_ready[i]);
    for (std::size_t i = 0; i < s.v_raw.size(); ++i) s.v_f16[i] = kernels::f32_to_f16_bits(s.v_raw[i]);
    // 本 cell 所在的逻辑页要常驻，且**在本步内钉住**（刚写进去的 K/V 不能被换出）。
    bool append_pinned = false;
    std::uint64_t append_page = 0;
    if (s.window != nullptr) {
        if (s.kv.page_size == 0) {
            err = "挂了常驻窗口但 page_size 是 0";
            return false;
        }
        append_page = s.pos / s.kv.page_size;
        if (!touch_page(s, append_page, err) || !s.window->pin(append_page, err)) return false;
        append_pinned = true;
    }
    // 权威副本先写（引擎的口径：写永远写主机副本，槽位只在该页常驻时顺带写）——故驱逐不必回写，
    // 且「准入时从副本拷进来」拿到的就是这一份（含本 cell）。
    if (s.store != nullptr &&
        !s.store->write_cell(s.pos, KvCell{s.k_f16.data(), s.v_f16.data()}, err)) {
        err = "写 KV 权威副本失败：" + err;
        return false;
    }
    if (!kv_append(s.kv, s.pos, KvCell{s.k_f16.data(), s.v_f16.data()}, err)) return false;

    // 4. 索引器：q 走 bf16 投影 + 归一 + rope；键走 bf16 投影后**原样**交给池化。
    if (phases != nullptr) phases->kv_ms += ms(p_kv, clk());
    const auto p_idx = clk();
    s.idx_q_out.assign(static_cast<std::size_t>(layer.idx_q.rows), 0.0f);
    s.idx_k_out.assign(static_cast<std::size_t>(layer.idx_k.rows), 0.0f);
    if (!experts::run_bf16_gemv(s.idx_q_w.data(), layer.idx_q.rows, layer.idx_q.cols, x,
                                s.idx_q_out.data(), err) ||
        !experts::run_bf16_gemv(s.idx_k_w.data(), layer.idx_k.rows, layer.idx_k.cols, x,
                                s.idx_k_out.data(), err)) {
        return false;
    }
    s.q_idx.assign(static_cast<std::size_t>(layer.idx_n_head * idx_dim), 0.0f);
    if (!qsa_indexer_prep(layer.idx_n_head, idx_dim, layer.n_rot, s.pos, s.idx_q_out.data(),
                          s.idx_q_norm_w.data(), 1e-6f, s.cos_tab, s.sin_tab, s.q_idx.data(), err)) {
        return false;
    }
    if (s.idx_state.dead.empty()) s.idx_state.reset(layer.sh, idx_dim);
    if (!indexer_key_append(layer.sh, idx_dim, layer.n_rot, s.pos, s.pos_base,
                            s.idx_k_out.data(),
                            s.idx_k_norm_w.data(), 1e-6f, s.cos_tab, s.sin_tab, s.idx_state, err)) {
        return false;
    }

    // 5. 逐 cell 打分 → 选择 → 按页聚集。
    const std::uint64_t n_kv = s.pos + 1;
    s.scores.assign(static_cast<std::size_t>(n_kv), 0.0f);
    if (!qsa_index(s.idx_state.pooled.data(), s.idx_state.n_bid, idx_dim, layer.idx_n_head,
                   s.q_idx.data(), nullptr, layer.sh.idx_block, n_kv, s.scores.data(), err)) {
        return false;
    }
    const std::uint64_t width = qsa_selection_width(n_kv, layer.sh);
    if (!qsa_select(s.scores.data(), n_kv, width, s.ids, err)) return false;
    if (phases != nullptr) phases->idx_ms += ms(p_idx, clk());  // 索引器全链 + 逐 cell 打分 + 选择
    const auto p_gather = clk();
    const std::uint64_t n_ids = s.ids.size();
    // 选中的 cell 所在页都要常驻（按页去重，只搬一次），并在聚集期间钉住。
    std::vector<std::uint64_t> gather_pages;
    if (s.window != nullptr) {
        for (std::uint64_t id : s.ids) {
            const std::uint64_t pg = id / s.kv.page_size;
            bool seen = false;
            for (std::uint64_t q : gather_pages) {
                if (q == pg) seen = true;
            }
            if (seen) continue;
            if (!touch_page(s, pg, err) || !s.window->pin(pg, err)) return false;
            gather_pages.push_back(pg);
        }
    }
    s.k_scratch.assign(static_cast<std::size_t>(n_ids * layer.sh.n_head_kv * hd), 0);
    s.v_scratch.assign(s.k_scratch.size(), 0);
    if (!kv_gather(s.kv, s.ids.data(), n_ids, s.k_scratch.data(), s.v_scratch.data(), err)) return false;
    for (std::uint64_t pg : gather_pages) {
        if (!s.window->unpin(pg, err)) return false;
    }

    // 6. softmax GQA → 乘门 → 输出投影。
    if (phases != nullptr) phases->kv_ms += ms(p_gather, clk());  // 按页聚集（含搬入与钉住）
    const auto p_att = clk();
    s.attn.assign(static_cast<std::size_t>(layer.sh.n_head * hd), 0.0f);
    if (!qsa_attend(s.q_ready.data(), s.k_scratch.data(), s.v_scratch.data(), n_ids, layer.sh,
                    s.attn.data(), nullptr, err)) {
        return false;
    }
    s.gated.assign(s.attn.size(), 0.0f);
    if (!qsa_gate_apply_f32(s.attn.data(), s.q_full.data(), layer.sh, s.gated.data(), err)) {
        return false;
    }
    if (phases != nullptr) phases->attend_ms += ms(p_att, clk());
    const auto p_out = clk();
    if (!project(gguf, *layer.t_out, layer.out, s.gated.data(), s.w_out, reload, s.gemv, out, err)) {
        return false;
    }
    proj_acc += ms(p_out, clk());
    if (append_pinned && !s.window->unpin(append_page, err)) return false;
    if (phases != nullptr) phases->proj_ms += proj_acc;
    s.w_layer = layer.layer;
    ++s.pos;
    return true;
}

}  // namespace qinfer::dense