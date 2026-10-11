#include "dense/block_layer.hpp"

#include "experts/ffn.hpp"
#include "kernels/bf16.hpp"

namespace qinfer::dense {

namespace {

const artifact::GgufTensorInfo* need(const artifact::GgufFile& g, const std::string& name,
                                     std::string& err) {
    const artifact::GgufTensorInfo* t = g.find(name);
    if (t == nullptr) err = name + " 不存在";
    return t;
}

// 一组 hc_* 的四张张力（norm 是 F32、down/up/inject 是 BF16——[S-53] 实测）。
// inject 不能缺：缺了 gr_read 会保留注入为 0，门退化 2·sigmoid(0) = 1，整段就成了普通残差相加。
bool need_gr_group(const artifact::GgufFile& g, const std::string& prefix,
                   const artifact::GgufTensorInfo*& n, const artifact::GgufTensorInfo*& d,
                   const artifact::GgufTensorInfo*& u, const artifact::GgufTensorInfo*& inj,
                   std::string& err) {
    n = need(g, prefix + "norm.weight", err);
    if (n == nullptr) return false;
    d = need(g, prefix + "down.weight", err);
    if (d == nullptr) return false;
    u = need(g, prefix + "up.weight", err);
    if (u == nullptr) return false;
    inj = need(g, prefix + "inject.weight", err);
    if (inj == nullptr) return false;
    if (n->dims.size() != 1 || d->dims.size() != 2 || u->dims.size() != 2 || inj->dims.size() != 2) {
        err = prefix + " 的维度不对（norm 一维、down/up/inject 二维）";
        return false;
    }
    return true;
}

// 把一组 hc_* 读进 scratch（第一次调用时）。
bool load_gr_group(const artifact::GgufFile& g, const artifact::GgufTensorInfo& n,
                   const artifact::GgufTensorInfo& d, const artifact::GgufTensorInfo& u,
                   const artifact::GgufTensorInfo& inj, std::vector<float>& n_f32,
                   std::vector<std::uint16_t>& d_bf16, std::vector<std::uint16_t>& u_bf16,
                   std::vector<std::uint16_t>& inj_bf16, std::string& err) {
    if (!n_f32.empty()) return true;
    if (n.type != 0) {  // F32
        err = n.name + " 不是 F32（hc_*_norm 在这份 GGUF 里是 F32，[S-53]）";
        return false;
    }
    const std::uint32_t bf16 = 30;
    if (d.type != bf16 || u.type != bf16 || inj.type != bf16) {
        err = d.name + " / " + u.name + " / " + inj.name + " 不是 BF16，本路径按 BF16 读";
        return false;
    }
    n_f32.resize(static_cast<std::size_t>(n.elements()));
    d_bf16.resize(static_cast<std::size_t>(d.elements()));
    u_bf16.resize(static_cast<std::size_t>(u.elements()));
    inj_bf16.resize(static_cast<std::size_t>(inj.elements()));
    if (!g.read_at(n.offset, reinterpret_cast<std::uint8_t*>(n_f32.data()), n_f32.size() * 4) ||
        !g.read_at(d.offset, reinterpret_cast<std::uint8_t*>(d_bf16.data()), d_bf16.size() * 2) ||
        !g.read_at(u.offset, reinterpret_cast<std::uint8_t*>(u_bf16.data()), u_bf16.size() * 2) ||
        !g.read_at(inj.offset, reinterpret_cast<std::uint8_t*>(inj_bf16.data()), inj_bf16.size() * 2)) {
        err = "读 hc_* 权重失败";
        return false;
    }
    return true;
}

}  // namespace

bool build_block_layer(const artifact::GgufFile& gguf, int layer, std::uint64_t n_embd,
                       std::uint64_t hc, std::uint64_t hc_lr, int top_k, std::uint64_t head_dim,
                       std::uint64_t n_rot, std::uint64_t idx_dim, std::uint64_t idx_top_k,
                       std::uint64_t idx_block, BlockLayer& out, std::string& err) {
    err.clear();
    out = BlockLayer{};
    out.layer = layer;
    out.n_embd = n_embd;
    out.gr = GrShapes{n_embd, hc, hc_lr};
    if (!out.gr.sane()) {
        err = "hc 的几何不成立（n_embd / hc / hc_lr 有零）";
        return false;
    }

    // 判型（用层表那条口径：[S-54] 的 QSA 有 attn_q、GDN 有 attn_qkv，互斥）。只看**这一层**：
    // MTP 的 pack 里只有 blk.48，走全模型那条 scan_layers 会被 0..47 的缺席挡住。
    if (!judge_layer(gguf, layer, out.kind, err)) {
        err = "层 " + std::to_string(layer) + " 判不出类型：" + err;
        return false;
    }

    const std::string b = "blk." + std::to_string(layer) + ".";
    if (!need_gr_group(gguf, b + "hc_attn_", out.a_norm, out.a_down, out.a_up, out.a_inject, err))
        return false;
    if (!need_gr_group(gguf, b + "hc_ffn_", out.f_norm, out.f_down, out.f_up, out.f_inject, err))
        return false;
    // 两组 hc_* 的形状必须与几何一致。inject 是 (hc, hc·n_embd)：每流一个标量门，行宽是整栈。
    const std::uint64_t hc_dim = n_embd * hc;
    if (out.a_norm->elements() != hc_dim || out.a_down->dims[0] != hc_dim ||
        out.a_down->dims[1] != hc_lr || out.a_up->dims[0] != hc_lr ||
        out.a_up->dims[1] != hc_dim || out.a_inject->dims[0] != hc_dim ||
        out.a_inject->dims[1] != hc || out.f_norm->elements() != hc_dim ||
        out.f_down->dims[0] != hc_dim || out.f_down->dims[1] != hc_lr ||
        out.f_up->dims[0] != hc_lr || out.f_up->dims[1] != hc_dim ||
        out.f_inject->dims[0] != hc_dim || out.f_inject->dims[1] != hc) {
        err = "hc_* 两组的形状与 n_embd/hc/hc_lr 不自洽";
        return false;
    }

    if (out.kind == LayerKind::kGdn) {
        if (!build_gdn_layer(gguf, layer, n_embd, head_dim, /*d_conv=*/4, out.gdn, err)) return false;
    } else {
        if (!build_qsa_layer(gguf, layer, n_embd, head_dim, n_rot, idx_dim, idx_top_k, idx_block,
                             out.qsa, err)) {
            return false;
        }
    }
    if (!build_moe_layer(gguf, layer, top_k, out.moe, err)) return false;
    return true;
}

bool run_block_layer(const artifact::GgufFile& gguf, const BlockLayer& layer, float* R,
                     BlockScratch& s, std::string& err) {
    err.clear();
    if (R == nullptr) {
        err = "run_block_layer 的 R 为空";
        return false;
    }
    const std::uint64_t n_embd = layer.n_embd;
    if (!load_gr_group(gguf, *layer.a_norm, *layer.a_down, *layer.a_up, *layer.a_inject,
                       s.a_norm_f32, s.a_down_bf16, s.a_up_bf16, s.a_inject_bf16, err) ||
        !load_gr_group(gguf, *layer.f_norm, *layer.f_down, *layer.f_up, *layer.f_inject,
                       s.f_norm_f32, s.f_down_bf16, s.f_up_bf16, s.f_inject_bf16, err)) {
        return false;
    }
    s.mixed.assign(static_cast<std::size_t>(n_embd), 0.0f);
    s.inject.assign(static_cast<std::size_t>(layer.gr.hc), 0.0f);
    s.block_out.assign(static_cast<std::size_t>(n_embd), 0.0f);

    // pre：gr_read(hc_attn_*) → 注意力 → gr_write。
    const GrWeights aw{GrTensor{s.a_norm_f32.data(), GrPrecision::kF32},
                       GrTensor{s.a_down_bf16.data(), GrPrecision::kBf16},
                       GrTensor{s.a_up_bf16.data(), GrPrecision::kBf16},
                       GrTensor{s.a_inject_bf16.data(), GrPrecision::kBf16}};
    if (!gr_read(layer.gr, aw, R, 1e-6f, s.mixed.data(), s.inject.data(), s.gr, err)) return false;
    if (layer.kind == LayerKind::kGdn) {
        if (!run_gdn_layer(gguf, layer.gdn, s.mixed.data(), s.block_out.data(), s.gdn, err)) {
            return false;
        }
    } else {
        if (!run_qsa_layer(gguf, layer.qsa, s.mixed.data(), s.block_out.data(), s.qsa, err)) {
            return false;
        }
    }
    if (!gr_write(layer.gr, R, s.block_out.data(), s.inject.data(), R, err)) return false;

    // post：gr_read(hc_ffn_*) → MoE → gr_write。
    const GrWeights fw{GrTensor{s.f_norm_f32.data(), GrPrecision::kF32},
                       GrTensor{s.f_down_bf16.data(), GrPrecision::kBf16},
                       GrTensor{s.f_up_bf16.data(), GrPrecision::kBf16},
                       GrTensor{s.f_inject_bf16.data(), GrPrecision::kBf16}};
    if (!gr_read(layer.gr, fw, R, 1e-6f, s.mixed.data(), s.inject.data(), s.gr, err)) return false;
    if (!run_moe_layer(gguf, layer.moe, s.mixed.data(), s.block_out.data(), s.moe, err)) return false;
    return gr_write(layer.gr, R, s.block_out.data(), s.inject.data(), R, err);
}

}  // namespace qinfer::dense