// IQ 家族里配 Q8_K 的另两档（IQ2_XXS、IQ1_M）的逐行点积设备实现。
//
// 为什么两档放一处：两者的配对物都是 Q8_K、都是「每行一个线程 + 整数域累加再乘尺度」的结构，
// 常量表（格点、符号表、符号位）也共用一批；分档写只会把同一段脚手架抄两遍。语义各自照本仓库
// 主机那份写（`kernels/iq2xxs`、`kernels/iq1m`），块内偏移与取位逐句对应。
//
// 不复用上游的 CUDA 实现（ggml 的 `mmvq.cu` / `vecdotq.cuh` 里这两档同样配 Q8_1 激活），
// 原因与 Q2_0、IQ2_S 那两块相同：配对物不同就是另一套数值行为，两边没法逐行对照。
//
// 只有 -DQINFER_ENABLE_CUDA=ON 时才编这一支（ADR-008 第 2 条）。
#pragma once

#include <cstdint>
#include <string>

#include "kernels/iq1m.hpp"
#include "kernels/iq2xxs.hpp"

namespace qinfer::kernels {

// IQ2_XXS × Q8_K：w 为 n_rows 行、每行 n_blocks 个块（行距 n_blocks × 66 字节），y 为 n_blocks 个 Q8_K 块。
bool iq2xxs_gemv_q8k_cuda(const std::uint8_t* w, int n_rows, int n_blocks, const Q8kBlock* y,
                          float* out, std::string& err);

// IQ1_M × Q8_K：同上，行距 n_blocks × 56 字节。
bool iq1m_gemv_q8k_cuda(const std::uint8_t* w, int n_rows, int n_blocks, const Q8kBlock* y,
                        float* out, std::string& err);

}  // namespace qinfer::kernels
