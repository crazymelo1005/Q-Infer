#include "dense/qsa_qk_prep.hpp"

#include "dense/norm.hpp"
#include "dense/rope.hpp"

namespace qinfer::dense {

namespace {

void rope_row(float* row, std::uint64_t n_rot, const std::vector<float>& cos_tab,
              const std::vector<float>& sin_tab, std::uint64_t pos) {
    const std::uint64_t half = n_rot / 2;
    if (n_rot == 0) return;
    for (std::uint64_t i = 0; i < half; ++i) {
        const std::size_t at = static_cast<std::size_t>(pos * half + i);
        float oa = 0.0f, ob = 0.0f;
        rope_neox_pair(row[i], row[i + half], cos_tab[at], sin_tab[at], oa, ob);
        row[i] = oa;
        row[i + half] = ob;
    }
}

}  // namespace

bool qsa_qk_prep(std::uint64_t n_head, std::uint64_t n_head_kv, std::uint64_t head_dim,
                 std::uint64_t n_rot, std::uint64_t pos, const float* q_full, const float* k_raw,
                 const QsaQkWeights& w, const std::vector<float>& cos_tab,
                 const std::vector<float>& sin_tab, float* q_ready, float* k_ready, float* gate,
                 std::string& err) {
    err.clear();
    if (n_head == 0 || n_head_kv == 0 || head_dim == 0) {
        err = "QSA 的注意力几何不成立";
        return false;
    }
    if (n_rot > head_dim || n_rot % 2 != 0) {
        err = "n_rot 必须是正偶数且不超过 head_dim";
        return false;
    }
    if (q_full == nullptr || k_raw == nullptr || q_ready == nullptr || k_ready == nullptr ||
        gate == nullptr) {
        err = "qsa_qk_prep 的指针不完整";
        return false;
    }
    if (w.q_norm == nullptr || w.k_norm == nullptr) {
        err = "qsa_qk_prep 缺 norm 的 gamma";
        return false;
    }

    // 1. 拆：每个头的 2·head_dim 块，前 half_block 是 q、后 half_block 是门（不是按元素交错）。
    //    门原样拷出（它不上注意力，故不归一、不 rope）。
    std::vector<float> q(static_cast<std::size_t>(n_head * head_dim), 0.0f);
    for (std::uint64_t h = 0; h < n_head; ++h) {
        const float* blk = q_full + h * 2 * head_dim;
        for (std::uint64_t d = 0; d < head_dim; ++d) {
            q[static_cast<std::size_t>(h * head_dim + d)] = blk[d];
            gate[h * head_dim + d] = blk[head_dim + d];
        }
    }

    // 2. 归一（按头，gamma 每头共用一份）→ 3. rope（只转前 n_rot 维，尾巴穿过）。
    rms_norm_rows(q.data(), n_head, head_dim, w.q_norm, w.eps, q_ready);
    for (std::uint64_t h = 0; h < n_head; ++h) {
        rope_row(q_ready + h * head_dim, n_rot, cos_tab, sin_tab, pos);
    }
    // k 同样处理，但头数是 n_head_kv。
    rms_norm_rows(k_raw, n_head_kv, head_dim, w.k_norm, w.eps, k_ready);
    for (std::uint64_t h = 0; h < n_head_kv; ++h) {
        rope_row(k_ready + h * head_dim, n_rot, cos_tab, sin_tab, pos);
    }
    return true;
}

}  // namespace qinfer::dense