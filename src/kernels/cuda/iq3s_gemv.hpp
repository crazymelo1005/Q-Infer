// IQ3_S × Q8_K 逐行点积的设备实现（阶段 A 稠密投影档之一：GDN attn_gate 是 IQ3_S）。
//
// 语义与输入布局与主机那份 `kernels/iq3s` 的 `iq3s_gemv_q8k` 一致，好让两边逐行对照：块内偏移
// 照抄（qs 在 2、qh 在 66、signs 在 74、scales 在 106、d 在块首 fp16 小端）；9 位格点索引、子尺度
// ls = 2×半字节 + 1、符号位掩码与格点表均与主机同一份。每行一个线程、逐块循环，块内结合顺序与
// 主机逐句对应。
//
// 只有 -DQINFER_ENABLE_CUDA=ON 时才编这一支（ADR-008）。
#pragma once

#include <cstdint>
#include <string>

#include "kernels/iq3s.hpp"

namespace qinfer::kernels {

// 全部在设备上：权重、激活、输出都是设备指针，只做一次启动与同步（常驻路径在多 kernel 流里用）。
bool iq3s_gemv_q8k_cuda_ddd(const void* d_w, int n_rows, int n_blocks, const void* d_y, void* d_out,
                            std::string& err);

// 主机侧包装：w 为 n_rows 行 × n_blocks 个 IQ3_S 块，y 为 n_blocks 个 Q8_K 块，out 长 n_rows。
bool iq3s_gemv_q8k_cuda(const std::uint8_t* w, int n_rows, int n_blocks, const Q8kBlock* y,
                        float* out, std::string& err);

}  // namespace qinfer::kernels