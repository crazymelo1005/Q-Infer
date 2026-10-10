#include "dense/qsa_attend.hpp"

#include "kernels/fp16.hpp"

#include <cmath>
#include <vector>

namespace qinfer::dense {

namespace {

inline float f16(uint16_t bits) { return kernels::f16_bits_to_f32(bits); }

}  // namespace

bool qsa_attend(const float* q, const std::uint16_t* k_scratch, const std::uint16_t* v_scratch,
                std::uint64_t n_ids, const QsaShapes& s, float* attn, float* weights,
                std::string& err) {
    err.clear();
    if (!s.attention_sane()) {
        err = "QSA 的注意力几何不成立（头宽/头数为 0，或 n_head 不是 n_head_kv 的整数倍）";
        return false;
    }
    if (q == nullptr || attn == nullptr) {
        err = "qsa_attend 的指针不完整";
        return false;
    }
    const std::uint64_t hd = s.head_dim;
    const std::uint64_t nh = s.n_head;
    const std::uint64_t nkv = s.n_head_kv;
    // ids 为空：没有选中的 cell，就没有 softmax 可取 —— attn 全 0（合法，不是错误）。
    for (std::uint64_t h = 0; h < nh * hd; ++h) attn[h] = 0.0f;
    if (n_ids == 0) return true;
    if (k_scratch == nullptr || v_scratch == nullptr) {
        err = "qsa_attend 有 cell 但 scratch 指针为空";
        return false;
    }

    const double scale = 1.0 / std::sqrt(static_cast<double>(hd));
    std::vector<double> scores(static_cast<std::size_t>(n_ids), 0.0);
    for (std::uint64_t h = 0; h < nh; ++h) {
        // q → kv 头的映射是**整除**（不是取模）。24 头 2 组时它是 [0×12, 1×12]。
        const std::uint64_t kvh = h / (nh / nkv);
        double mx = -HUGE_VAL;
        for (std::uint64_t j = 0; j < n_ids; ++j) {
            const std::uint16_t* krow = k_scratch + (j * nkv + kvh) * hd;
            double acc = 0.0;
            for (std::uint64_t d = 0; d < hd; ++d) {
                acc += static_cast<double>(f16(krow[d])) * static_cast<double>(q[h * hd + d]);
            }
            scores[static_cast<std::size_t>(j)] = acc * scale;
            if (scores[static_cast<std::size_t>(j)] > mx) mx = scores[static_cast<std::size_t>(j)];
        }
        double sum = 0.0;
        for (std::uint64_t j = 0; j < n_ids; ++j) {
            const double e = std::exp(scores[static_cast<std::size_t>(j)] - mx);
            scores[static_cast<std::size_t>(j)] = e;
            sum += e;
        }
        for (std::uint64_t j = 0; j < n_ids; ++j) {
            const double w = scores[static_cast<std::size_t>(j)] / (sum > 0.0 ? sum : 1.0);
            if (weights != nullptr) weights[h * n_ids + j] = static_cast<float>(w);
            const std::uint16_t* vrow = v_scratch + (j * nkv + kvh) * hd;
            for (std::uint64_t d = 0; d < hd; ++d) {
                attn[h * hd + d] += static_cast<float>(w * static_cast<double>(f16(vrow[d])));
            }
        }
    }
    return true;
}

bool qsa_gate_apply_f32(const float* attn, const float* q_full, const QsaShapes& s, float* out,
                        std::string& err) {
    err.clear();
    if (!s.attention_sane()) {
        err = "QSA 的注意力几何不成立";
        return false;
    }
    if (attn == nullptr || q_full == nullptr || out == nullptr) {
        err = "qsa_gate_apply 的指针不完整";
        return false;
    }
    const std::uint64_t hd = s.head_dim;
    for (std::uint64_t h = 0; h < s.n_head; ++h) {
        const float* gate = q_full + h * 2 * hd + hd;  // 每个头的 2·head_dim 块的**后半段**
        for (std::uint64_t d = 0; d < hd; ++d) {
            // 乘在 double 里（与引擎一致），故本函数与它只差最后那次舍入。
            const double g = static_cast<double>(gate[d]);
            out[h * hd + d] =
                static_cast<float>(static_cast<double>(attn[h * hd + d]) * (1.0 / (1.0 + std::exp(-g))));
        }
    }
    return true;
}

}  // namespace qinfer::dense