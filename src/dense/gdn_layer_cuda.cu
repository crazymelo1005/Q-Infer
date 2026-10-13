// GDN 整层设备执行（见 gdn_layer_cuda.hpp 的口径说明）。
// 串接顺序与主机 run_gdn_layer 一致；四段各调已验收的设备内核：
//   1. qkv 投影（gdn_proj_gemv，IQ4_XS）
//   2. 前段（gdn_preprocess_cuda：卷积+SiLU+拆头归一+双门）
//   3. gate 投影（gdn_proj_gemv，IQ3_S）+ q 乘 1/sqrt(S)
//   4. 递推（gdn_step_cuda）+ 收尾（gdn_closing_norm_cuda）
//   5. out 投影（gdn_proj_gemv，IQ4_XS）
// 权重按层缓存：s.w_* 首次到该层读、换层重读（w_layer 记账，与主机 GdnScratch 同语义）。
#include "dense/gdn_layer_cuda.hpp"

#include <cmath>
#include <cstring>

#include "dense/gdn_preprocess_cuda.hpp"
#include "dense/gdn_recurrence_cuda.hpp"
#include "kernels/cuda/gdn_proj_gemv.hpp"

namespace qinfer::dense {

namespace {

// 张量原始字节：F32/量化档按元素数×4，BF16 按元素数×2（与主机逐张力读同一口径）。
std::size_t tensor_bytes(const artifact::GgufTensorInfo* t) {
    return static_cast<std::size_t>(t->elements() *
                                    (experts::format_from_type_code(t->type) == experts::Format::kBf16
                                         ? 2
                                         : 4));
}

}  // namespace

bool run_gdn_layer_cuda(const artifact::GgufFile& gguf, const GdnLayer& layer, const float* x,
                        float* out, GdnScratch& s, std::string& err) {
    err.clear();
    if (x == nullptr || out == nullptr) {
        err = "run_gdn_layer_cuda 的输入输出指针为空";
        return false;
    }
    const bool reload = s.w_layer != layer.layer;
    const std::uint64_t N = layer.pre.n_embd;
    const std::uint64_t cd = layer.pre.conv_dim();
    const std::uint64_t vd = layer.pre.v_dim();
    const std::uint64_t hd = layer.pre.head_dim;

    // 1. 权重按层缓存（与前段/递推的 GdnPreWeights 对接）。
    if (reload) {
        s.ssm_a.clear();
        s.dt.clear();
        s.ssm_norm.clear();
        s.ssm_a.resize(static_cast<std::size_t>(layer.pre.n_v_head));
        s.dt.resize(static_cast<std::size_t>(layer.pre.n_v_head));
        s.ssm_norm.resize(static_cast<std::size_t>(hd));
        std::vector<float> tmp;
        auto read_f32 = [&](const artifact::GgufTensorInfo* t, std::vector<float>& dst) {
            std::vector<std::uint8_t> raw(static_cast<std::size_t>(t->elements() * 4));
            if (!gguf.read_at(t->offset, raw.data(), raw.size())) return false;
            dst.assign(reinterpret_cast<const float*>(raw.data()),
                       reinterpret_cast<const float*>(raw.data()) + t->elements());
            return true;
        };
        if (!read_f32(layer.t_ssm_a, s.ssm_a) || !read_f32(layer.t_dt, s.dt) ||
            !read_f32(layer.t_norm, s.ssm_norm)) {
            err = "读 GDN 的 ssm_a/dt/norm 失败";
            return false;
        }
        // conv1d / alpha / beta 读原始字节（前段 kernel 按声明精度取）。
        auto read_raw = [&](const artifact::GgufTensorInfo* t, std::vector<std::uint8_t>& dst) {
            dst.resize(tensor_bytes(t));
            return gguf.read_at(t->offset, dst.data(), dst.size());
        };
        if (!read_raw(layer.t_conv1d, s.w_conv1d) || !read_raw(layer.t_alpha, s.w_alpha) ||
            !read_raw(layer.t_beta, s.w_beta)) {
            err = "读 GDN 的 conv1d/alpha/beta 失败";
            return false;
        }
        // 三个投影的量化字节（按各自 MatrixSpec 的行几何）。
        auto read_proj = [&](const artifact::GgufTensorInfo* t, const experts::MatrixSpec& m,
                             std::vector<std::uint8_t>& dst) {
            dst.resize(static_cast<std::size_t>(m.rows * m.row_bytes()));
            return gguf.read_at(t->offset, dst.data(), dst.size());
        };
        if (!read_proj(layer.t_qkv, layer.qkv, s.w_qkv) ||
            !read_proj(layer.t_gate, layer.gate, s.w_gate) ||
            !read_proj(layer.t_out, layer.out, s.w_out)) {
            err = "读 GDN 的三个投影权重失败";
            return false;
        }
        s.w_layer = layer.layer;
    }

    // 2. qkv = attn_qkv @ x（设备）。
    s.qkv.assign(static_cast<std::size_t>(cd), 0.0f);
    qinfer::dense::GdnPreCudaScratch pre_scratch;
    pre_scratch.d_conv = nullptr;
    pre_scratch.conv_bytes = 0;
    experts::FfnScratch gemv;
    if (!experts::quantize_input(layer.qkv, x, gemv, err) ||
        !qinfer::kernels::gdn_proj_gemv(layer.qkv, s.w_qkv.data(), gemv, s.qkv.data(), nullptr,
                                        err)) {
        return false;
    }

    // 3. 前段（设备）。
    GdnConvState conv;
    conv.frames = std::move(s.conv.frames);
    conv.tokens = s.conv.tokens;
    GdnPreOut pre;
    GdnPreWeights pw;
    pw.conv1d = GrTensor{s.w_conv1d.data(), GrPrecision::kF32};
    pw.alpha = GrTensor{s.w_alpha.data(), GrPrecision::kBf16};
    pw.beta = GrTensor{s.w_beta.data(), GrPrecision::kBf16};
    pw.dt = s.dt.data();
    pw.ssm_a = s.ssm_a.data();
    if (!gdn_preprocess_cuda(layer.pre, pw, x, s.qkv.data(), conv, 1e-6f, pre, pre_scratch, err)) {
        return false;
    }
    s.conv.frames = std::move(conv.frames);
    s.conv.tokens = conv.tokens;

    // 4. z = attn_gate @ x（设备）；q 乘 1/sqrt(S)（契约 S-56：调用方的活）。
    s.z.assign(static_cast<std::size_t>(vd), 0.0f);
    if (!experts::quantize_input(layer.gate, x, gemv, err) ||
        !qinfer::kernels::gdn_proj_gemv(layer.gate, s.w_gate.data(), gemv, s.z.data(), nullptr,
                                        err)) {
        return false;
    }
    const float qscale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(hd)));
    s.q_scaled.assign(pre.q.begin(), pre.q.end());
    for (float& v : s.q_scaled) v *= qscale;

    // 5. 递推 + 收尾（设备）。
    std::vector<float> beta_gated = pre.beta;
    if (!gdn_beta_gate(beta_gated.data(), layer.rec.h_v, err)) return false;
    std::vector<float> o(static_cast<std::size_t>(vd), 0.0f);
    if (!gdn_step_cuda(s.state, layer.rec, s.q_scaled.data(), pre.k.data(), pre.v.data(),
                       pre.gate.data(), beta_gated.data(), o.data(), nullptr, err)) {
        return false;
    }
    s.y.assign(static_cast<std::size_t>(vd), 0.0f);
    if (!gdn_closing_norm_cuda(layer.rec, o.data(), s.z.data(), s.ssm_norm.data(), 1e-6f,
                               s.y.data(), nullptr, err)) {
        return false;
    }

    // 6. out = ssm_out @ y（设备）。
    s.out.assign(static_cast<std::size_t>(N), 0.0f);
    if (!experts::quantize_input(layer.out, s.y.data(), gemv, err) ||
        !qinfer::kernels::gdn_proj_gemv(layer.out, s.w_out.data(), gemv, s.out.data(), nullptr,
                                        err)) {
        return false;
    }
    for (std::uint64_t i = 0; i < N; ++i) out[i] = s.out[static_cast<std::size_t>(i)];
    return true;
}

}  // namespace qinfer::dense