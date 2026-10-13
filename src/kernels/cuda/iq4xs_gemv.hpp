// IQ4_XS × Q8_K 逐行点积的设备实现（阶段 A 稠密投影档之一：GDN attn_qkv/ssm_out 与 QSA attn_q/v
// 是 IQ4_XS）。
//
// 语义与输入布局与主机那份 `kernels/iq4xs` 的 `iq4xs_gemv_q8k` 一致，好让两边逐行对照：块内偏移
// 照抄（scales_h 在 2 的小端 u16、scales_l 在 4、qs 在 8、d 在块首 fp16 小端）；6 位尺度带 -32
// 偏移、码本 kIq4nlCode 与主机同一份；半字节序前 16 个取低、后 16 个取高，每 32 值前进 16 个 qs
// 字节。每行一个线程、逐块循环，块内结合顺序与主机逐句对应。
//
// 只有 -DQINFER_ENABLE_CUDA=ON 时才编这一支（ADR-008）。
#pragma once

#include <cstdint>
#include <string>

#include "kernels/iq4xs.hpp"

namespace qinfer::kernels {

// 全部在设备上：权重、激活、输出都是设备指针，只做一次启动与同步（常驻路径在多 kernel 流里用）。
bool iq4xs_gemv_q8k_cuda_ddd(const void* d_w, int n_rows, int n_blocks, const void* d_y, void* d_out,
                             std::string& err);

// 主机侧包装：w 为 n_rows 行 × n_blocks 个 IQ4_XS 块，y 为 n_blocks 个 Q8_K 块，out 长 n_rows。
bool iq4xs_gemv_q8k_cuda(const std::uint8_t* w, int n_rows, int n_blocks, const Q8kBlock* y,
                         float* out, std::string& err);

}  // namespace qinfer::kernels