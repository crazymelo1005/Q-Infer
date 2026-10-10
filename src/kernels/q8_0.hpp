// Q8_0：8 位块量化，每块 32 个 int8 加一个 fp16 尺度。它既作 Q2_0（每块 64 个值）的激活搭档
// ——一个 Q2_0 块配两个 Q8_0 块；也作权重出现（部署那份模型共享专家的 down 投影，输入维 640
// 整除 32 成立）。激活侧的定点点积配 Q8_0 权重或 IQ4_NL 权重（`QK4_NL == QK8_0`）。
//
// 尺度按 fp16 存（ggml 里 d 是 ggml_half），故这里存位型而不是 float，块大小才与文件一致。
// 块的逐位布局与转录来源见 q8_0.cpp 头部。
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

// 直接从文件字节反量化一块（不经过 Q80Block 结构体，避免对未对齐地址做重叠读）。
void dequant_q8_0_raw(const std::uint8_t* block, float* out32);

// 一段 n_blocks 个 Q8_0 权重块与 n_blocks 个 Q8_0 激活块的点积（两侧都是 Q8_0）。
float q8_0_dot_q8_0(const std::uint8_t* w_blocks, const Q80Block* y, int n_blocks);

// 逐行点积：w 为 n_rows 行，每行 n_blocks 个 Q8_0 块，行距 n_blocks × 34 字节。
void q8_0_gemv_q8_0(const std::uint8_t* w, int n_rows, int n_blocks, const Q80Block* y, float* out);

}  // namespace qinfer::kernels