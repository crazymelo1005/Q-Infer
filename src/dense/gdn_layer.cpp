#include "dense/gdn_layer.hpp"
#include <cstring>

#include "experts/ffn.hpp"
#include "kernels/bf16.hpp"

#include <cmath>

namespace qinfer::dense {

namespace {

const artifact::GgufTensorInfo* need(const artifact::GgufFile& g, const std::string& name,
                                     std::string& err) {
    const artifact::GgufTensorInfo* t = g.find(name);
    if (t == nullptr) err = name + " 不存在";
    return t;
}

// 一张 2 维张力的规格：档取它自己的类型码，几何取 [cols, rows]（GGML 的 ne0 是输入维）。
bool spec_of(const artifact::GgufTensorInfo& t, const char* what, experts::MatrixSpec& out,
             std::string& err) {
    if (t.dims.size() != 2) {
        err = std::string(what) + " 不是 2 维（GDN 的三个投影没有专家维）";
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
    if (!experts::has_kernel(out.format)) {
        err = std::string(what) + " 的档 " + experts::format_name(out.format) + " 没有内核";
        return false;
    }
    return true;
}

}  // namespace

bool build_gdn_layer(const artifact::GgufFile& gguf, int layer, std::uint64_t n_embd,
                     std::uint64_t head_dim, std::uint64_t d_conv, GdnLayer& out,
                     std::string& err) {
    err.clear();
    out = GdnLayer{};
    out.layer = layer;
    const std::string b = "blk." + std::to_string(layer) + ".";

    // QSA 层没有 attn_qkv（它用的是 attn_q/k/v + indexer，见 [S-54]），明确区分开来。
    out.t_qkv = gguf.find(b + "attn_qkv.weight");
    if (out.t_qkv == nullptr) {
        err = "层 " + std::to_string(layer) + " 不是 GDN 层（没有 attn_qkv.weight）";
        return false;
    }
    if (!need(gguf, b + "attn_gate.weight", err)) return false;
    out.t_gate = gguf.find(b + "attn_gate.weight");
    if (!need(gguf, b + "ssm_out.weight", err)) return false;
    out.t_out = gguf.find(b + "ssm_out.weight");
    if (!need(gguf, b + "ssm_conv1d.weight", err)) return false;
    out.t_conv1d = gguf.find(b + "ssm_conv1d.weight");
    if (!need(gguf, b + "ssm_alpha.weight", err)) return false;
    out.t_alpha = gguf.find(b + "ssm_alpha.weight");
    if (!need(gguf, b + "ssm_beta.weight", err)) return false;
    out.t_beta = gguf.find(b + "ssm_beta.weight");
    if (!need(gguf, b + "ssm_a", err)) return false;
    out.t_ssm_a = gguf.find(b + "ssm_a");
    if (!need(gguf, b + "ssm_dt.bias", err)) return false;
    out.t_dt = gguf.find(b + "ssm_dt.bias");
    if (!need(gguf, b + "ssm_norm.weight", err)) return false;
    out.t_norm = gguf.find(b + "ssm_norm.weight");

    if (!spec_of(*out.t_qkv, "attn_qkv", out.qkv, err)) return false;
    if (!spec_of(*out.t_gate, "attn_gate", out.gate, err)) return false;
    if (!spec_of(*out.t_out, "ssm_out", out.out, err)) return false;

    // 几何：q/k 各 h_k 头、v 是 h_v 头、每头 head_dim；10240 = 2·h_k·head_dim + h_v·head_dim。
    if (out.qkv.cols != n_embd) {
        err = "attn_qkv 的输入维 " + std::to_string(out.qkv.cols) + " 与 n_embd " +
              std::to_string(n_embd) + " 不一致";
        return false;
    }
    const std::uint64_t conv_dim = out.qkv.rows;
    const std::uint64_t v_dim = out.gate.rows;
    if (out.gate.cols != n_embd || out.out.cols != v_dim || out.out.rows != n_embd) {
        err = "attn_gate 或 ssm_out 的几何与 attn_qkv 不自洽";
        return false;
    }
    if (conv_dim <= v_dim || (conv_dim - v_dim) % 2 != 0) {
        err = "conv_dim − value_dim 不是两倍的 q/k 维";
        return false;
    }
    const std::uint64_t qk_dim = (conv_dim - v_dim) / 2;
    if (qk_dim % head_dim != 0 || v_dim % head_dim != 0) {
        err = "q/k 维或 v 维不是头宽的整数倍";
        return false;
    }
    out.pre = GdnShapes{n_embd, qk_dim / head_dim, qk_dim / head_dim, v_dim / head_dim, head_dim,
                        d_conv};
    out.rec = GdnRecShapes{head_dim, qk_dim / head_dim, v_dim / head_dim};
    // 其余张力的形状必须与推出来的几何对得上（错了就是接线错，早报比后面算出个可信的错值好）。
    if (out.t_conv1d->dims.size() != 2 || out.t_conv1d->dims[0] != d_conv ||
        out.t_conv1d->dims[1] != conv_dim) {
        err = "ssm_conv1d 的形状不是 [d_conv, conv_dim]";
        return false;
    }
    if (out.t_alpha->dims.size() != 2 || out.t_alpha->dims[0] != n_embd ||
        out.t_alpha->dims[1] != v_dim / head_dim || out.t_beta->dims.size() != 2 ||
        out.t_beta->dims[0] != n_embd || out.t_beta->dims[1] != v_dim / head_dim) {
        err = "ssm_alpha / ssm_beta 的形状不是 [n_embd, h_v]";
        return false;
    }
    if (out.t_ssm_a->elements() != v_dim / head_dim || out.t_dt->elements() != v_dim / head_dim ||
        out.t_norm->elements() != head_dim) {
        err = "ssm_a / ssm_dt / ssm_norm 的长度与 h_v / 头宽对不上";
        return false;
    }
    return true;
}

bool run_gdn_layer(const artifact::GgufFile& gguf, const GdnLayer& layer, const float* x, float* out,
                   GdnScratch& s, std::string& err) {
    err.clear();
    if (x == nullptr || out == nullptr) {
        err = "run_gdn_layer 的输入输出指针为空";
        return false;
    }
    const std::uint64_t N = layer.pre.n_embd;
    const std::uint64_t cd = layer.pre.conv_dim();
    const std::uint64_t vd = layer.pre.v_dim();
    const std::uint64_t hd = layer.pre.head_dim;

    // 1. 三个投影之外的四组权重（F32/BF16）读进来；投影的字节在用到时读。
    auto load_f32 = [&](const artifact::GgufTensorInfo* t, std::vector<float>& dst,
                        std::vector<std::uint8_t>& raw) {
        raw.resize(static_cast<std::size_t>(t->elements() * 4));
        if (!gguf.read_at(t->offset, raw.data(), raw.size())) return false;
        dst.resize(static_cast<std::size_t>(t->elements()));
        std::memcpy(dst.data(), raw.data(), raw.size());
        return true;
    };
    auto load_bf16 = [&](const artifact::GgufTensorInfo* t, std::vector<std::uint8_t>& raw) {
        raw.resize(static_cast<std::size_t>(t->elements() * 2));
        return gguf.read_at(t->offset, raw.data(), raw.size());
    };
    if (!load_f32(layer.t_ssm_a, s.ssm_a, s.w_conv1d) ||        // 先用 w_conv1d 当临时区
        !load_f32(layer.t_dt, s.dt, s.w_conv1d) || !load_f32(layer.t_norm, s.ssm_norm, s.w_conv1d) ||
        !load_bf16(layer.t_alpha, s.w_alpha) || !load_bf16(layer.t_beta, s.w_beta)) {
        err = "读 GDN 的门/卷积/归一权重失败";
        return false;
    }
    {
        std::vector<std::uint8_t> raw;
        if (!load_f32(layer.t_conv1d, s.qkv, raw)) {  // 借 s.qkv 当临时区（下一步会覆盖它）
            err = "读 ssm_conv1d 失败";
            return false;
        }
        s.w_conv1d = std::move(raw);
    }

    // 2. qkv = attn_qkv @ x（按档分派，与专家同一份表）。
    s.w_qkv.resize(static_cast<std::size_t>(layer.qkv.rows * layer.qkv.row_bytes()));
    if (!gguf.read_at(layer.t_qkv->offset, s.w_qkv.data(), s.w_qkv.size())) {
        err = "读 attn_qkv 失败";
        return false;
    }
    s.qkv.assign(static_cast<std::size_t>(cd), 0.0f);
    if (!experts::quantize_input(layer.qkv, x, s.gemv, err) ||
        !experts::run_gemv(layer.qkv, s.w_qkv.data(), s.gemv, s.qkv.data(), err)) {
        return false;
    }

    // 3. 前段：卷积+SiLU、拆 q|k|v、q/k 归一、两个门（门用 x 与 BF16 权重）。
    const GrTensor conv_w{static_cast<const void*>(s.w_conv1d.data()), GrPrecision::kF32};
    const GrTensor alpha_w{static_cast<const void*>(s.w_alpha.data()), GrPrecision::kBf16};
    const GrTensor beta_w{static_cast<const void*>(s.w_beta.data()), GrPrecision::kBf16};
    const GdnPreWeights pw{conv_w, alpha_w, beta_w, s.dt.data(), s.ssm_a.data()};
    const float eps = 1e-6f;
    if (!gdn_preprocess(layer.pre, pw, x, s.qkv.data(), s.conv, eps, s.pre, err)) return false;

    // 4. z = attn_gate @ x（量化投影）——收尾那个 sigmoid 的输入。
    s.w_gate.resize(static_cast<std::size_t>(layer.gate.rows * layer.gate.row_bytes()));
    if (!gguf.read_at(layer.t_gate->offset, s.w_gate.data(), s.w_gate.size())) {
        err = "读 attn_gate 失败";
        return false;
    }
    s.z.assign(static_cast<std::size_t>(vd), 0.0f);
    if (!experts::quantize_input(layer.gate, x, s.gemv, err) ||
        !experts::run_gemv(layer.gate, s.w_gate.data(), s.gemv, s.z.data(), err)) {
        return false;
    }

    // 5. q 乘 1/sqrt(S)——契约明写这一步归调用方（[S-56]）。
    const float qscale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(hd)));
    s.q_scaled.assign(s.pre.q.begin(), s.pre.q.end());
    for (float& v : s.q_scaled) v *= qscale;

    // 6. 递推 + 收尾。
    s.state.resize(layer.rec);
    std::vector<float> beta_gated = s.pre.beta;
    if (!gdn_beta_gate(beta_gated.data(), layer.rec.h_v, err)) return false;
    std::vector<float> o(static_cast<std::size_t>(vd), 0.0f);
    if (!gdn_step(s.state, layer.rec, s.q_scaled.data(), s.pre.k.data(), s.pre.v.data(),
                  s.pre.gate.data(), beta_gated.data(), o.data(), err)) {
        return false;
    }
    s.y.assign(static_cast<std::size_t>(vd), 0.0f);
    if (!gdn_closing_norm(layer.rec, o.data(), s.z.data(), s.ssm_norm.data(), eps, s.y.data(), err)) {
        return false;
    }

    // 7. out = ssm_out @ y（Q8_0 权重）。
    s.w_out.resize(static_cast<std::size_t>(layer.out.rows * layer.out.row_bytes()));
    if (!gguf.read_at(layer.t_out->offset, s.w_out.data(), s.w_out.size())) {
        err = "读 ssm_out 失败";
        return false;
    }
    s.out.assign(static_cast<std::size_t>(N), 0.0f);
    if (!experts::quantize_input(layer.out, s.y.data(), s.gemv, err) ||
        !experts::run_gemv(layer.out, s.w_out.data(), s.gemv, s.out.data(), err)) {
        return false;
    }
    for (std::uint64_t i = 0; i < N; ++i) out[i] = s.out[static_cast<std::size_t>(i)];
    return true;
}

}  // namespace qinfer::dense