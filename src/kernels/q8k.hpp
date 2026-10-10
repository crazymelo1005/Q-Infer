// Q8_K：激活侧的 8 位块量化，每块 256 个 int8 加一个 float 尺度。CPU 未命中专家路径把激活先量化
// 成这个格式，再与 IQ2_XS 权重做定点点积——权重因此不必解码成浮点，权重侧的内存流量不翻倍
// （engine §6 第 1 条）。
//
// 尺度的符号是个陷阱：参考量化器取 iscale = -127 / max（负值），故块尺度 d = 1/iscale 为负，
// 与同号取负的 qs 相乘才还原出正的原值。逐位细节与转录来源见 q8k.cpp 头部。
#pragma once

#include <cstdint>

namespace qinfer::kernels {

inline constexpr int kQ8kBlockElems = 256;
inline constexpr int kQ8kBlockBytes = 4 + 256 + 32;  // float d + int8 qs[256] + int16 bsums[16]

struct Q8kBlock {
    float d;
    std::int8_t qs[kQ8kBlockElems];
    std::int16_t bsums[kQ8kBlockElems / 16];  // 每 16 个 qs 的和，供向量化核使用
};

static_assert(sizeof(Q8kBlock) == kQ8kBlockBytes, "Q8_K block size mismatch");

void q8k_quantize_row(const float* x, Q8kBlock* out, int n_blocks);

void q8k_dequant_block(const Q8kBlock& blk, float* out256);

}  // namespace qinfer::kernels