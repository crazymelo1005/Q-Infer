// Q6_K × Q8_K 逐行点积的设备实现（阶段 A 稠密投影档之一：QSA attn_k 是 Q6_K）。
//
// 语义与输入布局与主机那份 `kernels/q6k` 的 `q6k_gemv_q8k` 一致，好让两边逐行对照：块内偏移照抄
// （ql 在 0、qh 在 128、scales 在 192、d 在 208 末尾），整数侧按「元素位置 mod 8」分组为 8 条部分和
// （与主机 `q6k_dot_q8k` 的 aux32[8] 同形状），最后乘 d。每行一个线程、逐块循环，块内结合顺序与
// 主机逐句对应；乘加不融合（逐条 __fadd_rn/__fmul_rn）。
//
// 只有 -DQINFER_ENABLE_CUDA=ON 时才编这一支（ADR-008）。
#pragma once

#include <cstdint>
#include <string>

#include "kernels/q6k.hpp"

namespace qinfer::kernels {

// 全部在设备上：权重、激活、输出都是设备指针，只做一次启动与同步（常驻路径在多 kernel 流里用）。
bool q6k_gemv_q8k_cuda_ddd(const void* d_w, int n_rows, int n_blocks, const void* d_y, void* d_out,
                           std::string& err);

// 主机侧包装：w 为 n_rows 行 × n_blocks 个 Q6_K 块，y 为 n_blocks 个 Q8_K 块，out 长 n_rows。
bool q6k_gemv_q8k_cuda(const std::uint8_t* w, int n_rows, int n_blocks, const Q8kBlock* y,
                       float* out, std::string& err);

}  // namespace qinfer::kernels