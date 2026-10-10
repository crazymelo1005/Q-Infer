#include "dense/gdn_recurrence.hpp"

#include <cmath>

namespace qinfer::dense {

namespace {

// 状态是 (S, h_v, S) 且 j 最快：第 (i, h, j) 个元素的下标。
inline std::size_t st_at(const GdnRecShapes& s, std::uint64_t i, std::uint64_t h, std::uint64_t j) {
    return static_cast<std::size_t>((i * s.h_v + h) * s.S + j);
}

inline float sigmoidf(float v) { return 1.0f / (1.0f + std::exp(-v)); }

}  // namespace

bool gdn_step(GdnState& st, const GdnRecShapes& s, const float* q, const float* k, const float* v,
              const float* gate, const float* beta, float* o, std::string& err) {
    err.clear();
    if (!s.sane()) {
        err = "GDN 的几何不成立（S / h_k / h_v 有零，或 h_v 不是 h_k 的整数倍）";
        return false;
    }
    if (q == nullptr || k == nullptr || v == nullptr || gate == nullptr || beta == nullptr ||
        o == nullptr) {
        err = "gdn_step 的输入指针不完整";
        return false;
    }
    if (st.data.size() != static_cast<std::size_t>(s.S * s.h_v * s.S)) {
        err = "循环状态的大小与几何不符";
        return false;
    }
    // 头配对用取模：第 h 个 v 头配第 h % h_k 个 q/k 头（不是按 h_v/h_k 分组的交错）。
    for (std::uint64_t h = 0; h < s.h_v; ++h) {
        const std::uint64_t idx = h % s.h_k;
        const float* kq = k + idx * s.S;
        const float* qq = q + idx * s.S;
        const float* vv = v + h * s.S;
        const float* zz = nullptr;
        (void) zz;
        float* oo = o + h * s.S;
        const float dec = std::exp(gate[h]);
        const float b = beta[h];
        // 每一条 (j, h) 列独立：先按 dec 衰减整列，再算 d、做秩一更新，最后出 o。
        for (std::uint64_t j = 0; j < s.S; ++j) {
            // 1. 衰减先作用；顺手把 sk 累起来（sk 用的是衰减后的值）。
            float sk = 0.0f;
            for (std::uint64_t i = 0; i < s.S; ++i) {
                float& e = st.data[st_at(s, i, h, j)];
                e *= dec;
                sk += e * kq[i];
            }
            // 2. d = (v - sk) * beta（beta 进来前必须已 sigmoid 过）。
            const float d = (vv[j] - sk) * b;
            // 3. 秩一更新：state += k ⊗ d。
            for (std::uint64_t i = 0; i < s.S; ++i) {
                st.data[st_at(s, i, h, j)] += kq[i] * d;
            }
            // 4. 出 o：用更新后的状态。
            float acc = 0.0f;
            for (std::uint64_t i = 0; i < s.S; ++i) {
                acc += st.data[st_at(s, i, h, j)] * qq[i];
            }
            oo[j] = acc;
        }
    }
    return true;
}

bool gdn_beta_gate(float* beta, std::uint64_t h_v, std::string& err) {
    err.clear();
    if (beta == nullptr) {
        err = "gdn_beta_gate 的指针为空";
        return false;
    }
    for (std::uint64_t h = 0; h < h_v; ++h) beta[h] = sigmoidf(beta[h]);
    return true;
}

bool gdn_closing_norm(const GdnRecShapes& s, const float* o, const float* z, const float* ssm_norm,
                      float eps, float* y, std::string& err) {
    err.clear();
    if (!s.sane()) {
        err = "GDN 的几何不成立";
        return false;
    }
    if (o == nullptr || z == nullptr || ssm_norm == nullptr || y == nullptr) {
        err = "gdn_closing_norm 的指针不完整";
        return false;
    }
    for (std::uint64_t h = 0; h < s.h_v; ++h) {
        const float* row = o + h * s.S;
        double ss = 0.0;
        for (std::uint64_t j = 0; j < s.S; ++j) ss += static_cast<double>(row[j]) * row[j];
        const float rs = static_cast<float>(1.0 / std::sqrt(ss / static_cast<double>(s.S) +
                                                            static_cast<double>(eps)));
        for (std::uint64_t j = 0; j < s.S; ++j) {
            const std::size_t at = static_cast<std::size_t>(h * s.S + j);
            y[at] = row[j] * rs * ssm_norm[j] * sigmoidf(z[at]);
        }
    }
    return true;
}

}  // namespace qinfer::dense