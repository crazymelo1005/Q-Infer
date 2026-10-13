// Q8_0 × Q8_0 逐行点积的设备实现（阶段 A dense 接线的缺口补档：合成/测试模型的稠密权重正是 Q8_0，
// --gpu 的设备路径必须覆盖这个档）。
//
// 语义与输入布局与主机那份 `kernels/q8_0` 的 `q8_0_gemv_q8_0` 一致，好让两边逐行对照。Q8_0 权限
// 的激活搭档是 Q8_0（32 值块、每块一个 fp16 尺度），与 Q2_0 的搭档同款；对照的结合顺序照主机
// `q8_0_dot_q8_0`：每块先做 32 值的整数点积，再乘「权重尺度 × 激活尺度」的积，最后加进行和。
//
// 只有 -DQINFER_ENABLE_CUDA=ON 时才编这一支（ADR-008 第 2 条）。
#pragma once

#include <cstdint>
#include <string>

#include "kernels/q8_0.hpp"

namespace qinfer::kernels {

// 设备侧逐行点积：w 为 n_rows 行、每行 n_blocks 个 Q8_0 块（行距 n_blocks × 34 字节），
// y 为 n_blocks 个 Q8_0 激活块，out 长 n_rows。
// 返回假时 err 写明原因（无设备、显存分配失败、内核报错），不静默给 0。
bool q8_0_gemv_q8_0_cuda(const std::uint8_t* w, int n_rows, int n_blocks, const Q80Block* y,
                         float* out, std::string& err);

// 全部在设备上的那一条：权重、激活、输出都是设备指针，本入口只做一次启动与同步，不分配也不拷贝。
// 常驻路径（gdn_proj_gemv 的 ddd 包装）用它把整个矩阵一次算完。
bool q8_0_gemv_q8_0_cuda_ddd(const void* d_w, int n_rows, int n_blocks, const void* d_y,
                             void* d_out, std::string& err);

}  // namespace qinfer::kernels