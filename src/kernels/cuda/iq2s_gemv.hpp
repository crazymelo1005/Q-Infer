// IQ2_S × Q8_K 逐行点积的设备实现（interfaces.md 第 8 节那条缝上的第二块内核）。
//
// 与主机那份 `kernels/iq2s` 的 `iq2s_gemv_q8k` 同语义、同输入布局：块内偏移（d 在 0、qs 在 2、
// 符号在 34、qh 在 66、scales 在 74）、10 位格点索引的取位（低 8 位来自 qs、高 2 位来自 qh）、
// 尺度规则（ls = 1 + 2·4bit，每 16 个值一组）与 Q8_K 的整数累加都逐句对应。
//
// 为什么不用上游的 CUDA 实现：ggml 的 `vecdotq.cuh` 里 IQ2_S 配的是 **Q8_1** 激活
// （`vec_dot_iq2_s_q8_1`，带 d 与 s 两个尺度、整数累加时要四舍五入），本仓库的激活搭档是 Q8_K
// （每 256 值一个 float 尺度），配对物不同，抄过来就是另一套数值行为、两边没法逐行对照。
//
// 只有 -DQINFER_ENABLE_CUDA=ON 时才编这一支（ADR-008 第 2 条）。
#pragma once

#include <cstdint>
#include <string>

#include "kernels/iq2s.hpp"

namespace qinfer::kernels {

// 设备侧逐行点积：w 为 n_rows 行、每行 n_blocks 个 IQ2_S 块（行距 n_blocks × 82 字节），
// y 为 n_blocks 个 Q8_K 激活块，out 长 n_rows。
// 返回假时 err 写明原因（无设备、常量表上传失败、显存分配失败、内核报错），不静默给 0。
bool iq2s_gemv_q8k_cuda(const std::uint8_t* w, int n_rows, int n_blocks, const Q8kBlock* y,
                        float* out, std::string& err);

}  // namespace qinfer::kernels
