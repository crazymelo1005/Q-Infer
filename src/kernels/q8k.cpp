// Q8_K 的块布局与参考激活量化器转录自 llama.cpp 的 ggml（提交 3cf03257f219afbe7334045ff7c6a06ac68c627d，
// 2026-09-20，MIT，Copyright (c) 2023-2026 The ggml authors；来源登记见 S-35）：
//   ggml/src/ggml-common.h  l.371 block_q8_K
//   ggml/src/ggml-quants.c  l.621 nearest_int / l.2768 quantize_row_q8_K_ref
// 只取这三处。bsums 在标量点积里用不到（向量化核才用），此处按原样算出以保持结构完整。
#include "kernels/q8k.hpp"

#include <cmath>
#include <cstring>

namespace qinfer::kernels {

namespace {

// ggml 的 nearest_int：把 fval + 12582912（= 1.5·2^23）的尾数直接当整数读，从而四舍五入到最近的
// 整数（.5 远离零）。要求 |fval| ≤ 2^22。
int nearest_int(float fval) {
    const float val = fval + 12582912.0f;
    std::int32_t i;
    std::memcpy(&i, &val, sizeof(i));
    return (i & 0x007fffff) - 0x00400000;
}

}  // namespace

void q8k_quantize_row(const float* x, Q8kBlock* out, int n_blocks) {
    for (int b = 0; b < n_blocks; ++b) {
        const float* xb = x + b * kQ8kBlockElems;
        Q8kBlock& y = out[b];
        float max = 0.0f;   // 幅值最大的那个元素本身（带符号），不是幅值
        float amax = 0.0f;
        for (int j = 0; j < kQ8kBlockElems; ++j) {
            const float ax = std::fabs(xb[j]);
            if (ax > amax) {
                amax = ax;
                max = xb[j];
            }
        }
        if (amax == 0.0f) {
            y.d = 0.0f;
            std::memset(y.qs, 0, sizeof(y.qs));
            std::memset(y.bsums, 0, sizeof(y.bsums));
            continue;
        }
        const float iscale = -127.0f / max;
        for (int j = 0; j < kQ8kBlockElems; ++j) {
            const int v = nearest_int(iscale * xb[j]);
            y.qs[j] = static_cast<std::int8_t>(v < 127 ? v : 127);
        }
        for (int g = 0; g < kQ8kBlockElems / 16; ++g) {
            int sum = 0;
            for (int i = 0; i < 16; ++i) sum += y.qs[g * 16 + i];
            y.bsums[g] = static_cast<std::int16_t>(sum);
        }
        y.d = 1.0f / iscale;
    }
}

void q8k_dequant_block(const Q8kBlock& blk, float* out256) {
    for (int j = 0; j < kQ8kBlockElems; ++j) out256[j] = blk.d * static_cast<float>(blk.qs[j]);
}

}  // namespace qinfer::kernels