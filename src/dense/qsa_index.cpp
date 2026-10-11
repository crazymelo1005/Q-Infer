#include "dense/qsa_index.hpp"

#include <cmath>

namespace qinfer::dense {

namespace {

// 「不满一块的尾巴」的偏置：与 llama.cpp 因果解码的输入一致，且不论 bias 传没传都要加。
constexpr float kTailBias = 1e9f;

}  // namespace

bool qsa_index(const float* pooled, std::uint64_t n_bid, std::uint64_t idx_dim,
               std::uint64_t idx_n_head, const float* q_idx, const float* bias, std::uint64_t idx_block,
               std::uint64_t n_kv, float* cell_scores, std::string& err) {
    err.clear();
    if (idx_dim == 0 || idx_n_head == 0 || idx_block == 0) {
        err = "索引器的几何不成立（idx_dim / idx_n_head / idx_block 有零）";
        return false;
    }
    if (pooled == nullptr || q_idx == nullptr || cell_scores == nullptr) {
        err = "qsa_index 的指针不完整";
        return false;
    }
    const std::uint64_t n_blocks = n_bid + 1;  // 已完成块 + 备用槽
    const std::uint64_t done = (n_kv / idx_block) * idx_block;

    for (std::uint64_t j = 0; j < n_kv; ++j) {
        // 块映射：`min(j / r, n_bid)`。尾巴那些 cell 因此落到**备用槽**上（这就是解码期掩码的全部内容）。
        const std::uint64_t cb = std::min(j / idx_block, n_bid);
        if (cb >= n_blocks) {
            err = "块下标越界";
            return false;
        }
        const float* key = pooled + cb * idx_dim;
        float score = bias != nullptr ? bias[cb] : 0.0f;
        for (std::uint64_t h = 0; h < idx_n_head; ++h) {
            const float* qq = q_idx + h * idx_dim;
            double dot = 0.0;
            for (std::uint64_t d = 0; d < idx_dim; ++d) {
                dot += static_cast<double>(key[d]) * static_cast<double>(qq[d]);
            }
            // Relu 逐头取，求和在后（这是本函数最容易写反的一处）。
            score += static_cast<float>(dot > 0.0 ? dot : 0.0);
        }
        // 不满一块的尾巴加偏置（已完成块不加）。
        if (j >= done) score += kTailBias;
        cell_scores[j] = score;
    }
    return true;
}

}  // namespace qinfer::dense