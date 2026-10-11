#include "dense/qsa_layer.hpp"

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

    if (head_dim == 0 || n_rot == 0 || n_rot > head_dim || n_rot % 2 != 0 || idx_dim == 0) {
        err = "head_dim / n_rot / idx_dim 不合法";
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

}  // namespace qinfer::dense