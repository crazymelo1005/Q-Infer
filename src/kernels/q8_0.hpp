// Q8_0：8 位块量化，每块 32 个 int8 加一个 fp16 尺度。它是 Q2_0 权重（每块 64 个值）的激活搭档
// ——一个 Q2_0 块配两个 Q8_0 块；在这个模型里它同时也作权重出现（共享专家的 down 投影）。
//
// 尺度按 fp16 存（ggml 里 d 是 ggml_half），故这里存位型而不是 float，块大小才与文件一致。
// 块布局与反量化转录来源见 q8_0.cpp 头部。参考量化器（float → Q8_0）随激活路径一起做，本片不含。
#pragma once

#include <cstdint>

namespace qinfer::kernels {

inline constexpr int kQ80BlockElems = 32;
inline constexpr int kQ80BlockBytes = 34;  // 2 字节 fp16 尺度 + 32 字节 int8

struct Q80Block {
    std::uint16_t d_bits;
    std::int8_t qs[kQ80BlockElems];
};

static_assert(sizeof(Q80Block) == kQ80BlockBytes, "Q8_0 block size mismatch");

// 参考激活量化器：尺度是块内绝对值最大者除以 127（fp16 存），量化到最近整数。
void q8_0_quantize_row(const float* x, Q80Block* out, int n_blocks);

void q8_0_dequant_block(const Q80Block& blk, float* out32);

}  // namespace qinfer::kernels