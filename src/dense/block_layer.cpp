#include "dense/block_layer.hpp"

#include <chrono>

#include "experts/ffn.hpp"
#include "kernels/bf16.hpp"
#include "kernels/q8_0.hpp"
#include "kernels/bf16.hpp"

namespace qinfer::dense {

namespace {

// 把一张 hc_* 投影的字节按**它自己声明的档**解成 F32（ADR-010）：BF16 是 f32 的高半、放大无损；
// F32 直读；Q8_0 逐块反量化。真模型里主模型部署那份是 BF16、社区 MTP pack 是 Q8_0。
bool load_proj_f32(const artifact::GgufFile& g, const artifact::GgufTensorInfo& t,
                   std::vector<float>& out, std::string& err) {
    if (t.dims.empty()) {
        err = t.name + " 没有维度";
        return false;
    }
    const std::uint64_t cols = t.dims[0];
    const std::uint64_t rows = t.dims.size() >= 2 ? t.dims[1] : 1;
    if (t.type == 0) {  // F32
        out.resize(static_cast<std::size_t>(t.elements()));
        if (!g.read_at(t.offset, reinterpret_cast<std::uint8_t*>(out.data()), out.size() * 4)) {
            err = t.name + " 读失败";
            return false;
        }
        return true;
    }
    if (t.type == 30) {  // BF16 → F32（无损）
        std::vector<std::uint16_t> tmp(static_cast<std::size_t>(t.elements()));
        if (!g.read_at(t.offset, reinterpret_cast<std::uint8_t*>(tmp.data()), tmp.size() * 2)) {
            err = t.name + " 读失败";
            return false;
        }
        out.resize(tmp.size());
        for (std::size_t i = 0; i < tmp.size(); ++i) out[i] = kernels::bf16_bits_to_f32(tmp[i]);
        return true;
    }
    if (t.type == 8) {  // Q8_0：每块 32 值 / 34 字节（2 字节 fp16 尺度 + 32 个 int8 码）
        constexpr std::uint64_t kBlock = 32, kBytes = 34;
        if (cols == 0 || cols % kBlock != 0) {
            err = t.name + " 的列数 " + std::to_string(cols) + " 不是 Q8_0 块元素数的整数倍";
            return false;
        }
        const std::uint64_t nblk = cols / kBlock;
        const std::uint64_t row_bytes = nblk * kBytes;
        std::vector<std::uint8_t> raw(static_cast<std::size_t>(rows * row_bytes));
        if (!g.read_at(t.offset, raw.data(), raw.size())) {
            err = t.name + " 读失败";
            return false;
        }
        out.assign(static_cast<std::size_t>(rows * cols), 0.0f);
        for (std::uint64_t r = 0; r < rows; ++r) {
            for (std::uint64_t b = 0; b < nblk; ++b) {
                kernels::q8_0_dequant_block(
                    *reinterpret_cast<const kernels::Q80Block*>(raw.data() + r * row_bytes +
                                                               b * kBytes),
                    out.data() + r * cols + b * kBlock);
            }
        }
        return true;
    }
    err = t.name + " 的档（码 " + std::to_string(t.type) +
          "）本路径不支持（只收 F32 / BF16 / Q8_0）";
    return false;
}

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
                   std::vector<float>& d_f32, std::vector<float>& u_f32,
                   std::vector<float>& inj_f32, std::string& err) {
    if (!n_f32.empty()) return true;
    if (n.type != 0) {  // F32
        err = n.name + " 不是 F32（hc_*_norm 在这份 GGUF 里是 F32，[S-53]）";
        return false;
    }
    n_f32.resize(static_cast<std::size_t>(n.elements()));
    if (!g.read_at(n.offset, reinterpret_cast<std::uint8_t*>(n_f32.data()), n_f32.size() * 4)) {
        err = "读 " + n.name + " 失败";
        return false;
    }
    return load_proj_f32(g, d, d_f32, err) && load_proj_f32(g, u, u_f32, err) &&
           load_proj_f32(g, inj, inj_f32, err);
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
                     BlockScratch& s, std::string& err, LayerPhases* phases) {
    err.clear();
    if (R == nullptr) {
        err = "run_block_layer 的 R 为空";
        return false;
    }
    // 计时点每层只有八处（纳秒级），故不做条件分支，只在末尾写进 phases（为空就丢掉）。
    using clock = std::chrono::steady_clock;
    const auto ms_between = [](clock::time_point a, clock::time_point b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };
    clock::time_point t0 = clock::now();
    const std::uint64_t n_embd = layer.n_embd;
    if (!load_gr_group(gguf, *layer.a_norm, *layer.a_down, *layer.a_up, *layer.a_inject,
                       s.a_norm_f32, s.a_down_f32, s.a_up_f32, s.a_inject_f32, err) ||
        !load_gr_group(gguf, *layer.f_norm, *layer.f_down, *layer.f_up, *layer.f_inject,
                       s.f_norm_f32, s.f_down_f32, s.f_up_f32, s.f_inject_f32, err)) {
        return false;
    }
    clock::time_point t1 = clock::now();
    s.mixed.assign(static_cast<std::size_t>(n_embd), 0.0f);
    s.inject.assign(static_cast<std::size_t>(layer.gr.hc), 0.0f);
    s.block_out.assign(static_cast<std::size_t>(n_embd), 0.0f);

    // pre：gr_read(hc_attn_*) → 注意力 → gr_write。
    const GrWeights aw{GrTensor{s.a_norm_f32.data(), GrPrecision::kF32},
                       GrTensor{s.a_down_f32.data(), GrPrecision::kF32},
                       GrTensor{s.a_up_f32.data(), GrPrecision::kF32},
                       GrTensor{s.a_inject_f32.data(), GrPrecision::kF32}};
    if (!gr_read(layer.gr, aw, R, 1e-6f, s.mixed.data(), s.inject.data(), s.gr, err)) return false;
    clock::time_point t2 = clock::now();
    if (layer.kind == LayerKind::kGdn) {
        if (!run_gdn_layer(gguf, layer.gdn, s.mixed.data(), s.block_out.data(), s.gdn, err,
                           phases != nullptr ? &phases->attn_proj_ms : nullptr)) {
            return false;
        }
    } else {
        if (!run_qsa_layer(gguf, layer.qsa, s.mixed.data(), s.block_out.data(), s.qsa, err)) {
            return false;
        }
    }
    clock::time_point t3 = clock::now();
    if (!gr_write(layer.gr, R, s.block_out.data(), s.inject.data(), R, err)) return false;
    clock::time_point t4 = clock::now();

    // post：gr_read(hc_ffn_*) → MoE → gr_write。
    const GrWeights fw{GrTensor{s.f_norm_f32.data(), GrPrecision::kF32},
                       GrTensor{s.f_down_f32.data(), GrPrecision::kF32},
                       GrTensor{s.f_up_f32.data(), GrPrecision::kF32},
                       GrTensor{s.f_inject_f32.data(), GrPrecision::kF32}};
    if (!gr_read(layer.gr, fw, R, 1e-6f, s.mixed.data(), s.inject.data(), s.gr, err)) return false;
    clock::time_point t5 = clock::now();
    experts::MoePhases mp;
    if (!run_moe_layer(gguf, layer.moe, s.mixed.data(), s.block_out.data(), s.moe, err, nullptr,
                       nullptr, s.offload, &mp)) {
        return false;
    }
    clock::time_point t6 = clock::now();
    const bool ok = gr_write(layer.gr, R, s.block_out.data(), s.inject.data(), R, err);
    const clock::time_point t7 = clock::now();
    if (phases != nullptr) {
        phases->load_ms += ms_between(t0, t1);
        phases->gr_read_ms += ms_between(t1, t2) + ms_between(t4, t5);
        phases->attn_ms += ms_between(t2, t3);
        phases->gr_write_ms += ms_between(t3, t4) + ms_between(t6, t7);
        phases->moe_ms += ms_between(t5, t6);
        phases->moe_io_ms += mp.io_ms;
        phases->moe_ffn_ms += mp.ffn_ms;
    }
    return ok;
}

}  // namespace qinfer::dense