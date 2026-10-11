#include "dense/qsa_indexer.hpp"

#include "dense/norm.hpp"
#include "dense/rope.hpp"

#include <cmath>

namespace qinfer::dense {

namespace {

// 归一这一处与索引器之外的两处（QSA 的 q/k 逐头、收尾的 o）共用 `dense/norm.hpp` 的那一份，
// 免得三处各写一遍口径（先归一后乘 gamma、eps 加在均值上）。
inline void norm_and_scale(const float* row, const float* gamma, std::uint64_t n, float eps,
                           float* out) {
    rms_norm_rows(row, 1, n, gamma, eps, out);
}

// 对一行做 NEOX 部分 rope：位置取表里第 pos 组（表由调用方按 n_rot 与 theta 建好）。
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

void QsaIndexerState::reset(const QsaShapes& s, std::uint64_t idx_dim) {
    const std::uint64_t r = s.idx_block == 0 ? 1 : s.idx_block;
    window.assign(static_cast<std::size_t>(r * idx_dim), 0.0f);
    dead.assign(static_cast<std::size_t>(idx_dim), 0.0f);
    pooled.assign(static_cast<std::size_t>(idx_dim), 0.0f);  // 至少一格（pos 0 时会写它）
    n_bid = 0;
}

bool indexer_key_append(const QsaShapes& s, std::uint64_t idx_dim, std::uint64_t pos,
                        std::uint64_t pos_base, const float* raw, const float* w_k_norm, float eps,
                        const std::vector<float>& cos_tab, const std::vector<float>& sin_tab,
                        QsaIndexerState& st, std::string& err) {
    err.clear();
    const std::uint64_t r = s.idx_block == 0 ? 1 : s.idx_block;
    if (idx_dim == 0 || s.head_dim == 0) {
        err = "索引器的几何不成立（idx_dim / head_dim 为 0）";
        return false;
    }
    if (raw == nullptr || w_k_norm == nullptr) {
        err = "indexer_key_append 的指针不完整";
        return false;
    }
    if (st.window.size() != static_cast<std::size_t>(r * idx_dim) ||
        st.dead.size() != static_cast<std::size_t>(idx_dim)) {
        err = "索引器状态没按几何开好（先调 reset）";
        return false;
    }
    const std::uint64_t rot = s.head_dim;  // rope 的旋转维数（这份模型是 head_dim 的一部分，由表决定）

    // 序列的第一个 cell：算备用槽的键并把它写进 dead 与 pooled[0]。
    if (pos == 0) {
        norm_and_scale(raw, w_k_norm, idx_dim, eps, st.dead.data());
        rope_row(st.dead.data(), rot, cos_tab, sin_tab, /*pos=*/0);
        st.pooled.assign(static_cast<std::size_t>(idx_dim), 0.0f);
        for (std::uint64_t d = 0; d < idx_dim; ++d) st.pooled[d] = st.dead[d];
    }

    // 把这一帧放进窗口（窗口按 cell 序对齐到块：块 b 的 R 个 cell 占 window[0..R-1]）。
    const std::uint64_t slot = pos % r;
    for (std::uint64_t d = 0; d < idx_dim; ++d) {
        st.window[static_cast<std::size_t>(slot * idx_dim + d)] = raw[d];
    }

    // 块满：算这一块的池化键。旋转位置取**该块首个 cell 的位置** `pos_base + R*b`。
    if (slot + 1 == r) {
        const std::uint64_t b = pos / r;
        std::vector<float> mean(static_cast<std::size_t>(idx_dim), 0.0f);
        for (std::uint64_t c = 0; c < r; ++c) {
            for (std::uint64_t d = 0; d < idx_dim; ++d) {
                mean[static_cast<std::size_t>(d)] +=
                    st.window[static_cast<std::size_t>(c * idx_dim + d)] /
                    static_cast<float>(r);
            }
        }
        const std::uint64_t rope_pos = pos_base + r * b;
        std::vector<float> key(static_cast<std::size_t>(idx_dim), 0.0f);
        norm_and_scale(mean.data(), w_k_norm, idx_dim, eps, key.data());
        rope_row(key.data(), rot, cos_tab, sin_tab, rope_pos);
        // pooled 长 n_bid + 2：第 b 格放块键、第 b+1 格放备用槽（dead 的镜像）。
        st.pooled.resize(static_cast<std::size_t>((b + 2) * idx_dim), 0.0f);
        for (std::uint64_t d = 0; d < idx_dim; ++d) {
            st.pooled[static_cast<std::size_t>(b * idx_dim + d)] = key[d];
            st.pooled[static_cast<std::size_t>((b + 1) * idx_dim + d)] = st.dead[d];
        }
        st.n_bid = b + 1;
    }
    return true;
}

}  // namespace qinfer::dense