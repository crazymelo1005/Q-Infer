#include "dense/gdn_preprocess.hpp"

#include "kernels/bf16.hpp"

#include <cmath>

namespace qinfer::dense {

namespace {

inline float w_at(const GrTensor& t, std::uint64_t i) {
    if (t.precision == GrPrecision::kF32) {
        return static_cast<const float*>(t.data)[i];
    }
    return kernels::bf16_bits_to_f32(static_cast<const std::uint16_t*>(t.data)[i]);
}

// 第 row 行的第 i 个元素（按声明精度取）。
inline float w_at_row(const GrTensor& t, std::uint64_t row, std::uint64_t row_len,
                      std::uint64_t i) {
    const std::uint64_t at = row * row_len + i;
    return t.precision == GrPrecision::kF32 ? static_cast<const float*>(t.data)[at]
                                            : kernels::bf16_bits_to_f32(
                                                  static_cast<const std::uint16_t*>(t.data)[at]);
}

inline float sigmoidf(float v) { return 1.0f / (1.0f + std::exp(-v)); }

// silu(v) = v · sigmoid(v)。
inline float siluf(float v) { return v * sigmoidf(v); }

// softplus(v) = log(1 + e^v)，数值稳定写法（大 v 直接返回 v，避免 e^v 溢出）。
inline float softplusf(float v) {
    if (v > 20.0f) return v;
    if (v < -20.0f) return std::exp(v);
    return std::log1p(std::exp(v));
}

}  // namespace

bool gdn_preprocess(const GdnShapes& s, const GdnPreWeights& w, const float* x, const float* qkv,
                    GdnConvState& state, float eps, GdnPreOut& out, std::string& err) {
    err.clear();
    if (!s.sane()) {
        err = "GDN 的几何不成立（头数/头宽/卷积核长有零）";
        return false;
    }
    if (x == nullptr || qkv == nullptr) {
        err = "gdn_preprocess 的输入指针不完整";
        return false;
    }
    if (!w.conv1d.present() || !w.alpha.present() || !w.beta.present() || w.dt == nullptr ||
        w.ssm_a == nullptr) {
        err = "gdn_preprocess 缺权重（conv1d / alpha / beta / dt / ssm_a 之一为空）";
        return false;
    }
    const std::uint64_t conv_dim = s.conv_dim();
    const std::uint64_t win = s.d_conv > 0 ? s.d_conv - 1 : 0;
    state.frames.resize(static_cast<std::size_t>(win * conv_dim), 0.0f);

    // 1. 卷积：窗口 = 状态里的 win 帧（最老在前）+ 这一帧的 qkv，共 d_conv 帧；权重第 0 行配最早一帧。
    //    头 d_conv-1 个 token 左端补零，与因果卷积一致。
    std::vector<float> conv(static_cast<std::size_t>(conv_dim), 0.0f);
    for (std::uint64_t r = 0; r < s.d_conv; ++r) {
        const float* frame = r < win ? state.frames.data() + r * conv_dim : qkv;
        for (std::uint64_t i = 0; i < conv_dim; ++i) {
            conv[static_cast<std::size_t>(i)] += w_at_row(w.conv1d, r, conv_dim, i) * frame[i];
        }
    }

    // 2. SiLU 加在**整段** conv 输出上，在拆分之前。
    for (std::uint64_t i = 0; i < conv_dim; ++i) {
        conv[static_cast<std::size_t>(i)] = siluf(conv[static_cast<std::size_t>(i)]);
    }

    // 3. 拆 q|k|v，并按头做 l2 归一——**q 与 k 各自归一，v 不归一**。
    //    1/sqrt(S) 那个尺度按契约由调用方在归一之后施加（这里只归一，不缩放）。
    out.q.assign(static_cast<std::size_t>(s.q_dim()), 0.0f);
    out.k.assign(static_cast<std::size_t>(s.k_dim()), 0.0f);
    out.v.assign(static_cast<std::size_t>(s.v_dim()), 0.0f);
    for (std::uint64_t h = 0; h < s.n_q_head; ++h) {
        const float* src = conv.data() + h * s.head_dim;
        double ss = 0.0;
        for (std::uint64_t i = 0; i < s.head_dim; ++i) ss += static_cast<double>(src[i]) * src[i];
        const float rs = static_cast<float>(1.0 / std::sqrt(ss + static_cast<double>(eps)));
        for (std::uint64_t i = 0; i < s.head_dim; ++i) {
            out.q[static_cast<std::size_t>(h * s.head_dim + i)] = src[i] * rs;
        }
    }
    for (std::uint64_t h = 0; h < s.n_k_head; ++h) {
        const float* src = conv.data() + s.q_dim() + h * s.head_dim;
        double ss = 0.0;
        for (std::uint64_t i = 0; i < s.head_dim; ++i) ss += static_cast<double>(src[i]) * src[i];
        const float rs = static_cast<float>(1.0 / std::sqrt(ss + static_cast<double>(eps)));
        for (std::uint64_t i = 0; i < s.head_dim; ++i) {
            out.k[static_cast<std::size_t>(h * s.head_dim + i)] = src[i] * rs;
        }
    }
    for (std::uint64_t i = 0; i < s.v_dim(); ++i) {
        out.v[static_cast<std::size_t>(i)] = conv[static_cast<std::size_t>(s.q_dim() + s.k_dim() + i)];
    }

    // 4. 两个门：beta = sigmoid(wbeta@x)，gate = softplus(walpha@x + dt)·ssm_a。
    out.beta.assign(static_cast<std::size_t>(s.n_v_head), 0.0f);
    out.gate.assign(static_cast<std::size_t>(s.n_v_head), 0.0f);
    for (std::uint64_t h = 0; h < s.n_v_head; ++h) {
        double ab = 0.0, ag = 0.0;
        for (std::uint64_t i = 0; i < s.n_embd; ++i) {
            ab += static_cast<double>(x[i]) * static_cast<double>(w_at(w.beta, h * s.n_embd + i));
            ag += static_cast<double>(x[i]) * static_cast<double>(w_at(w.alpha, h * s.n_embd + i));
        }
        out.beta[static_cast<std::size_t>(h)] = sigmoidf(static_cast<float>(ab));
        const float g = static_cast<float>(ag) + w.dt[h];
        out.gate[static_cast<std::size_t>(h)] = softplusf(g) * w.ssm_a[h];
    }

    // 5. 状态前移：把这一帧接到窗口尾，丢掉最老的一帧。
    if (win > 0) {
        for (std::uint64_t r = 0; r + 1 < win; ++r) {
            for (std::uint64_t i = 0; i < conv_dim; ++i) {
                state.frames[static_cast<std::size_t>(r * conv_dim + i)] =
                    state.frames[static_cast<std::size_t>((r + 1) * conv_dim + i)];
            }
        }
        for (std::uint64_t i = 0; i < conv_dim; ++i) {
            state.frames[static_cast<std::size_t>((win - 1) * conv_dim + i)] = qkv[i];
        }
    }
    ++state.tokens;
    return true;
}

}  // namespace qinfer::dense