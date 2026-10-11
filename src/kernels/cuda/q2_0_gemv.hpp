// Q2_0 × Q8_0 逐行点积的设备实现（interfaces.md 第 8 节那条缝上的第一块 CUDA 内核）。
//
// 语义与输入布局与主机那份 `kernels/q2_0` 的 `q2_0_gemv_q8_0` 一致，好让两边逐行对照。主机那份
// 转录自 ggml 的标量码（S-37）；本文件照它写设备版，**不复用 ggml 的 CUDA 实现**——上游 CUDA 那条
// （ggml/src/ggml-cuda/mmvq.cu 与 vecdotq.cuh 的 Q2_0 分支）把权重配的是 Q8_1 激活（带 d 与 s 两个
// 尺度、整数累加时要四舍五入），本仓库的激活搭档是 Q8_0（每 32 值一个 fp16 尺度），配对物不同，
// 抄过来会得到另一套数值行为，两边就没法逐行对照了。
//
// 只有 -DQINFER_ENABLE_CUDA=ON 时才编这一支（ADR-008 第 2 条：CI 在没有 GPU 的 runner 上不编 CUDA）。
//
// 边界：本入口每次调用都要把权重搬上设备再搬回结果，是「第一块内核 + 一个可核验的入口」，不是
// 设备侧的执行路径。权重常驻、按卡分工、流与图捕获由 CUDA 组那一层负责（interfaces §8.1）。
#pragma once

#include <cstdint>
#include <string>

#include "kernels/q2_0.hpp"

namespace qinfer::kernels {

// 设备侧逐行点积：w 为 n_rows 行、每行 n_blocks 个 Q2_0 块（行距 n_blocks × 18 字节），
// y 为 2×n_blocks 个 Q8_0 激活块，out 长 n_rows。
// 返回假时 err 写明原因（无设备、显存分配失败、内核报错），不静默给 0。
bool q2_0_gemv_q8_0_cuda(const std::uint8_t* w, int n_rows, int n_blocks, const Q80Block* y,
                         float* out, std::string& err);

// 全部在设备上的那一条：权重、激活、输出都是设备指针，本入口只做一次启动与同步，不分配也不拷贝
// （`d_y` 是 2×n_blocks 个 Q8_0 块）。常驻路径用它把一次前馈的几个 kernel 串在同一个流上，
// 中间不与主机同步——接口约束见 [interfaces.md](../interfaces.md) §8.4（每步只有一个汇合点）。
bool q2_0_gemv_q8_0_cuda_ddd(const void* d_w, int n_rows, int n_blocks, const void* d_y,
                             void* d_out, std::string& err);

}  // namespace qinfer::kernels
