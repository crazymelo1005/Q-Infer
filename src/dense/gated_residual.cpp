#include "dense/gated_residual.hpp"

#include "kernels/bf16.hpp"

#include <cmath>

namespace qinfer::dense {

namespace {

// 权重取出成 f32：bf16 是左移 16 位即得位型、无舍入；F32 直接取。逐张力按它自己声明的精度。
inline float w_at(const GrTensor& t, std::uint64_t i) {
    if (t.precision == GrPrecision::kF32) {
        return static_cast<const float*>(t.data)[i];
    }
    return kernels::bf16_bits_to_f32(static_cast<const std::uint16_t*>(t.data)[i]);
}

// 激活侧的 bf16 舍入，口径与参考实现同（就近舍入到偶数，见 S-44 的 f32_to_bf16_bits）。
inline float round_act(float v) {
    return kernels::bf16_bits_to_f32(kernels::f32_to_bf16_bits(v));
}

inline float sigmoidf(float v) { return 1.0f / (1.0f + std::exp(-v)); }

// 某一行（或某一个的投影组）的视图：基址按行推进，精度跟着走。
inline GrTensor row_view(const GrTensor& t, std::uint64_t row, std::uint64_t row_len) {
    GrTensor r = t;
    const std::uint64_t off = row * row_len;
    r.data = t.precision == GrPrecision::kF32
                 ? static_cast<const void*>(static_cast<const float*>(t.data) + off)
                 : static_cast<const void*>(static_cast<const std::uint16_t*>(t.data) + off);
    return r;
}

// silu(v) = v · sigmoid(v)。
inline float siluf(float v) { return v * sigmoidf(v); }

}  // namespace

float rms_scale(const float* row, std::uint64_t n, float eps) {
    // 化简在 n_embd 上、每条流一条：不是整栈一条。求和用 double 累加，避免长流的顺序敏感。
    double ss = 0.0;
    for (std::uint64_t i = 0; i < n; ++i) {
        ss += static_cast<double>(row[i]) * static_cast<double>(row[i]);
    }
    const double ms = ss / static_cast<double>(n);
    return static_cast<float>(1.0 / std::sqrt(ms + static_cast<double>(eps)));
}

bool gr_read(const GrShapes& s, const GrWeights& w, const float* R, float eps, float* mixed,
             float* inject, GrScratch& scratch, std::string& err) {
    err.clear();
    if (!s.sane()) {
        err = "超连接的几何不成立（n_embd / hc / hc_lr 有零）";
        return false;
    }
    if (!w.w_norm.present() || !w.w_down.present() || !w.w_up.present()) {
        err = "gr_read 缺权重（w_norm / w_down / w_up 之一为空）";
        return false;
    }
    if (R == nullptr || mixed == nullptr || inject == nullptr) {
        err = "gr_read 的缓冲区指针不完整";
        return false;
    }
    const std::uint64_t n = s.n_embd;
    const std::uint64_t hc = s.hc;
    const std::uint64_t lr = s.hc_lr;
    const std::uint64_t hc_dim = hc * n;

    // xn = rms_norm(R) · (1 + w)，逐流一条 RMS。
    scratch.xn.resize(static_cast<std::size_t>(hc_dim));
    scratch.act.resize(static_cast<std::size_t>(hc_dim));
    for (std::uint64_t c = 0; c < hc; ++c) {
        const float* row = R + c * n;
        const float rs = rms_scale(row, n, eps);
        for (std::uint64_t d = 0; d < n; ++d) {
            const std::uint64_t i = c * n + d;
            scratch.xn[static_cast<std::size_t>(i)] = row[d] * rs * w_at(w.w_norm, i);
        }
    }
    for (std::uint64_t i = 0; i < hc_dim; ++i) {
        scratch.act[static_cast<std::size_t>(i)] = round_act(scratch.xn[static_cast<std::size_t>(i)]);
    }

    // lo = silu((bf16(xn) · w_downᵀ) / hc)：除 hc 在 silu 里面。
    scratch.lo.assign(static_cast<std::size_t>(lr), 0.0f);
    for (std::uint64_t k = 0; k < lr; ++k) {
        const GrTensor wrow = row_view(w.w_down, k, hc_dim);
        double acc = 0.0;
        for (std::uint64_t i = 0; i < hc_dim; ++i) {
            acc += static_cast<double>(scratch.act[static_cast<std::size_t>(i)]) *
                   static_cast<double>(w_at(wrow, i));
        }
        const float p = static_cast<float>(acc);
        scratch.lo[static_cast<std::size_t>(k)] = siluf(p / static_cast<float>(hc));
    }
    scratch.lq.resize(static_cast<std::size_t>(lr));
    for (std::uint64_t k = 0; k < lr; ++k) {
        scratch.lq[static_cast<std::size_t>(k)] = round_act(scratch.lo[static_cast<std::size_t>(k)]);
    }

    // mixed[d] = (1/hc) Σ_c xn[c,d] · sigmoid(bf16(lo) · w_up[c,d]ᵀ)：按流求平均，不是求和；
    // 被门乘的是**未舍入**的 xn（gated 平均要的是未舍入值）。
    for (std::uint64_t d = 0; d < n; ++d) {
        float m = 0.0f;
        for (std::uint64_t c = 0; c < hc; ++c) {
            const std::uint64_t i = c * n + d;
            const GrTensor wrow = row_view(w.w_up, i, lr);
            double acc = 0.0;
            for (std::uint64_t k = 0; k < lr; ++k) {
                acc += static_cast<double>(scratch.lq[static_cast<std::size_t>(k)]) *
                       static_cast<double>(w_at(wrow, k));
            }
            const float g = static_cast<float>(acc);
            m += scratch.xn[static_cast<std::size_t>(i)] * sigmoidf(g);
        }
        mixed[d] = m / static_cast<float>(hc);
    }

    // inject[c] = bf16(xn)[c] · w_inject[c]ᵀ。w_inject 缺席时（最后那个 mixer）原样不动。
    if (w.w_inject.present()) {
        for (std::uint64_t c = 0; c < hc; ++c) {
            const GrTensor wrow = row_view(w.w_inject, c, hc_dim);
            double acc = 0.0;
            for (std::uint64_t i = 0; i < hc_dim; ++i) {
                acc += static_cast<double>(scratch.act[static_cast<std::size_t>(i)]) *
                       static_cast<double>(w_at(wrow, i));
            }
            inject[c] = static_cast<float>(acc);
        }
    }
    return true;
}

bool gr_write(const GrShapes& s, const float* R, const float* block_out, const float* inject,
              float* R_out, std::string& err) {
    err.clear();
    if (!s.sane()) {
        err = "超连接的几何不成立（n_embd / hc / hc_lr 有零）";
        return false;
    }
    if (R == nullptr || block_out == nullptr || inject == nullptr || R_out == nullptr) {
        err = "gr_write 的缓冲区指针不完整";
        return false;
    }
    const std::uint64_t n = s.n_embd;
    const std::uint64_t hc = s.hc;
    // w[c] = 2·sigmoid(inject[c]/hc)：把门心定在 1，故零注入退化成普通残差相加。
    std::vector<float> w(static_cast<std::size_t>(hc), 0.0f);
    for (std::uint64_t c = 0; c < hc; ++c) {
        w[static_cast<std::size_t>(c)] =
            2.0f * sigmoidf(inject[c] / static_cast<float>(hc));
    }
    for (std::uint64_t c = 0; c < hc; ++c) {
        const std::uint64_t base = c * n;
        const float k = w[static_cast<std::size_t>(c)];
        for (std::uint64_t d = 0; d < n; ++d) {
            R_out[base + d] = R[base + d] + block_out[d] * k;
        }
    }
    return true;
}

}  // namespace qinfer::dense