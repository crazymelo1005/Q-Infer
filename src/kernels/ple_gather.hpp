// 把 16 个已反量化的行拼成该层的 PLE 向量。
//
// 布局是 head-slowest：头 h 的 160 个值落在 out[h*160, (h+1)*160)。这不是随意选择——参考引擎的
// ggml_get_rows 与 gather 都按这个次序展平，换成交错布局会得到一组「形状对、值错」的向量，
// 且对同一份扁平缓冲做 reshape 比较是完全查不出来的。故这里把约定钉死在一个函数里，并由测试用
// 「每个头一行的值互不相同」的数据去验证。
#pragma once

#include "kernels/ngram.hpp"

namespace qinfer::kernels {

inline constexpr int kPleHeadDim = 160;
inline constexpr int kPleVectorDim = kPleNHeads * kPleHeadDim;   // 2560

void assemble_ple_vector(const float (*rows)[kPleHeadDim], float* out2560);

}  // namespace qinfer::kernels
