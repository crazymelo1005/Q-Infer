#include "dense/gdn_recurrence.hpp"

#include "scheduling/thread_pool.hpp"

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
    //
    // 两块都要快得多的地方（改前：固定 (h,j)、内层走 i——而状态里 **j 最快**，于是每一步跨
    // h_v·S 个 float 取一个元素，三条内层循环都是这种步长）：
    //   1. 归约沿 i 走、**最内层沿 j 连续**（把 j 放到最里，相同算术、完全不同的访存）；
    //   2. 按 h 并行（每一头的状态片互相独立）。
    // 两处都不动任何一个元素的归约顺序，故与单线程**逐位相同**（tests/test_gdn_recurrence 钉住）。
    const std::uint64_t row_stride = s.h_v * s.S;
    st.sk_buf.resize(static_cast<std::size_t>(s.h_v * s.S));
    st.d_buf.resize(static_cast<std::size_t>(s.h_v * s.S));
    const auto one_head = [&](std::uint64_t h) {
        const std::uint64_t idx = h % s.h_k;
        const float* kq = k + idx * s.S;
        const float* qq = q + idx * s.S;
        const float* vv = v + h * s.S;
        float* oo = o + h * s.S;
        float* sk = st.sk_buf.data() + static_cast<std::size_t>(h) * s.S;
        float* d = st.d_buf.data() + static_cast<std::size_t>(h) * s.S;
        float* base = st.data.data() + static_cast<std::size_t>(h) * s.S;  // st_at(0, h, 0)
        const float dec = std::exp(gate[h]);
        const float b = beta[h];
        for (std::uint64_t j = 0; j < s.S; ++j) sk[j] = 0.0f;
        // 衰减先作用；顺手把 sk 累起来（sk 用的是衰减后的值）。i 升序累加，与单线程同序。
        for (std::uint64_t i = 0; i < s.S; ++i) {
            float* row = base + static_cast<std::size_t>(i) * row_stride;
            const float kv = kq[i];
            for (std::uint64_t j = 0; j < s.S; ++j) {
                row[j] *= dec;
                sk[j] += row[j] * kv;
            }
        }
        // d = (v - sk) * beta（beta 进来前必须已 sigmoid 过）。
        for (std::uint64_t j = 0; j < s.S; ++j) d[j] = (vv[j] - sk[j]) * b;
        // 秩一更新：每个元素只加一次。
        for (std::uint64_t i = 0; i < s.S; ++i) {
            float* row = base + static_cast<std::size_t>(i) * row_stride;
            const float kv = kq[i];
            for (std::uint64_t j = 0; j < s.S; ++j) row[j] += kv * d[j];
        }
        // 出 o：用更新后的状态，仍按 i 升序累加。
        for (std::uint64_t j = 0; j < s.S; ++j) oo[j] = 0.0f;
        for (std::uint64_t i = 0; i < s.S; ++i) {
            const float* row = base + static_cast<std::size_t>(i) * row_stride;
            const float qv = qq[i];
            for (std::uint64_t j = 0; j < s.S; ++j) oo[j] += row[j] * qv;
        }
    };
    if (scheduling::ThreadPool* p = scheduling::pool(); p != nullptr && s.h_v > 1) {
        p->run_rows(s.h_v, /*grain=*/1, [&](std::uint64_t b, std::uint64_t e) {
            for (std::uint64_t h = b; h < e; ++h) one_head(h);
        });
    } else {
        for (std::uint64_t h = 0; h < s.h_v; ++h) one_head(h);
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