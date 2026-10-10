// 专家 FFN 的三矩阵组装：out += w · down · (swiglu(gate · x, up · x))。
//
// 形状（实测自 S-36 / S-38）：gate/up 的输入是 hidden、输出是 ffn；down 的输入是 ffn、输出是 hidden。
// 激活按每个矩阵自己的档选：256 值块的档（IQ2_*、IQ3_*、Q*_K）配 Q8_K，32/64 值块的档
// （Q2_0、IQ4_NL）配 Q8_0——依据是各自 generic 点积核里的 static_assert 与配对约定
// （S-37 / S-40 / S-41）。门控用 SwiGLU（引擎的 MoE 路径就是 swiglu，见 src/core/layer.cpp）。
//
// 组装本身不做近似：不支持的档、gate 与 up 激活档不一致、缺矩阵指针，都返回 false 并写 err，
// 不做「猜一个能用的档」这种事。
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "experts/expert_formats.hpp"
#include "kernels/q8_0.hpp"
#include "kernels/q8k.hpp"

namespace qinfer::experts {

// 一个专家的三个矩阵各自的起始位置：都是连续的块区，长度 = rows × row_bytes()。
// 一个专家在文件里的三个区不互相重叠，可由张量的逐专家偏移 + expert_id × (rows × row_bytes) 得到。
struct ExpertWeights {
    const std::uint8_t* gate = nullptr;
    const std::uint8_t* up = nullptr;
    const std::uint8_t* down = nullptr;

    bool complete() const { return gate != nullptr && up != nullptr && down != nullptr; }
};

// 复用的中间缓冲，避免每次前馈都分配。
struct FfnScratch {
    std::vector<float> gate;    // 长度 ffn
    std::vector<float> up;      // 长度 ffn
    std::vector<float> h;       // 长度 ffn
    std::vector<float> out_tmp; // 长度 hidden，供 moe_ffn 累加用
    std::vector<kernels::Q8kBlock> act_k;
    std::vector<kernels::Q80Block> act_32;
};

float silu(float x);

// 单个专家的一次前馈：结果写入 out（长度 hidden，覆盖写）。
bool expert_ffn(const LayerSpec& spec, const ExpertWeights& w, const float* x, FfnScratch& scratch,
                float* out, std::string& err);

// MoE 前馈：对 n 个已选专家各自前馈，按 route_weight 加权累加到 out（调用方负责 out 的初值）。
bool moe_ffn(const LayerSpec& spec, const ExpertWeights* ws, const float* route_weight, int n,
             const float* x, float* out, FfnScratch& scratch, std::string& err);

}  // namespace qinfer::experts