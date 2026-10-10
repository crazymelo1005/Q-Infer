#include "dense/rope.hpp"

#include <cmath>

namespace qinfer::dense {

bool build_rope_table(std::uint64_t n_rot, double theta, std::uint64_t max_pos,
                      std::vector<float>& cos_tab, std::vector<float>& sin_tab, std::string& err) {
    err.clear();
    if (n_rot == 0 || n_rot % 2 != 0) {
        err = "n_rot 必须是正的偶数（NEOX 按对旋转）";
        return false;
    }
    if (theta <= 0.0) {
        err = "theta 必须是正数";
        return false;
    }
    const std::uint64_t half = n_rot / 2;
    cos_tab.assign(static_cast<std::size_t>(max_pos * half), 1.0f);
    sin_tab.assign(static_cast<std::size_t>(max_pos * half), 0.0f);
    for (std::uint64_t p = 0; p < max_pos; ++p) {
        for (std::uint64_t i = 0; i < half; ++i) {
            // 角度在 double 里算（表本身是 f32，但中间不该再掉一次精度）。
            const double ang =
                static_cast<double>(p) * std::pow(theta, -2.0 * static_cast<double>(i) /
                                                              static_cast<double>(n_rot));
            cos_tab[static_cast<std::size_t>(p * half + i)] = static_cast<float>(std::cos(ang));
            sin_tab[static_cast<std::size_t>(p * half + i)] = static_cast<float>(std::sin(ang));
        }
    }
    return true;
}

bool rope_neox_apply(const float* x, std::uint64_t rows, std::uint64_t head_dim,
                     std::uint64_t n_rot, const std::uint64_t* pos,
                     const std::vector<float>& cos_tab, const std::vector<float>& sin_tab, float* out,
                     std::string& err) {
    err.clear();
    if (x == nullptr || out == nullptr || pos == nullptr) {
        err = "rope_neox_apply 的指针不完整";
        return false;
    }
    if (n_rot == 0 || n_rot % 2 != 0 || n_rot > head_dim) {
        err = "n_rot 必须是正的偶数且不超过 head_dim";
        return false;
    }
    const std::uint64_t half = n_rot / 2;
    if (cos_tab.size() < half || sin_tab.size() < half) {
        err = "cos/sin 表太小";
        return false;
    }
    for (std::uint64_t r = 0; r < rows; ++r) {
        const float* src = x + r * head_dim;
        float* dst = out + r * head_dim;
        const std::uint64_t p = pos[r];
        const std::size_t base = static_cast<std::size_t>(p * half);
        if (base + half > cos_tab.size()) {
            err = "位置 " + std::to_string(p) + " 超出表长";
            return false;
        }
        // 前 n_rot 维：NEOX 配对 (i, i + half)。
        for (std::uint64_t i = 0; i < half; ++i) {
            float oa = 0.0f, ob = 0.0f;
            rope_neox_pair(src[i], src[i + half], cos_tab[base + static_cast<std::size_t>(i)],
                           sin_tab[base + static_cast<std::size_t>(i)], oa, ob);
            dst[i] = oa;
            dst[i + half] = ob;
        }
        // 尾巴原样穿过（部分旋转）。
        for (std::uint64_t d = n_rot; d < head_dim; ++d) dst[d] = src[d];
    }
    return true;
}

}  // namespace qinfer::dense