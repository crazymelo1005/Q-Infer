// 逐行 RMSNorm（乘 gamma）。三处要用同一份：索引器键的池化键、QSA 的 q/k 逐头归一、以及收尾的 o 归一
// （收尾那一处是「按 v 头、每头一条」的同一种算术，只是 gamma 是 `ssm_norm`）。
//
// 口径（两处都容易写反，测试里各有错的读法对照）：
//   * 分母是 `sqrt(mean(x²) + eps)`——`eps` 加在**均值**上（另一个读法 `sqrt(sum(x²) + eps)` 在
//     cols 大时差得很多；这与 GDN 的 l2 归一不同，那边 eps 是加在平方和上的绝对下限，见 [S-56]）；
//   * 次序是**先归一、后逐维乘 gamma**（先乘后归一会给出另一个形状同样正确的结果）。
#pragma once

#include <cmath>
#include <cstdint>

namespace qinfer::dense {

// x/out 是 (rows, cols) 行主序，gamma 长 cols（每行共用一份）。
inline void rms_norm_rows(const float* x, std::uint64_t rows, std::uint64_t cols, const float* gamma,
                          float eps, float* out) {
    for (std::uint64_t r = 0; r < rows; ++r) {
        const float* src = x + r * cols;
        float* dst = out + r * cols;
        double ss = 0.0;
        for (std::uint64_t d = 0; d < cols; ++d) {
            ss += static_cast<double>(src[d]) * static_cast<double>(src[d]);
        }
        const float scale = static_cast<float>(
            1.0 / std::sqrt(ss / static_cast<double>(cols) + static_cast<double>(eps)));
        for (std::uint64_t d = 0; d < cols; ++d) {
            dst[d] = gamma != nullptr ? scale * src[d] * gamma[d] : scale * src[d];
        }
    }
}

}  // namespace qinfer::dense