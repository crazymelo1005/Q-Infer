// GDN 循环本体与收尾的设备实现（阶段 A）。
//
// 语义与主机 `dense/gdn_recurrence` 的 `gdn_step` / `gdn_closing_norm` 一致：每 v 头独立的递推
// （dec = exp(gate[h]) → st 衰减 → sk 沿 i 升序累加（j 连续访问）→ d = (v - sk)·beta →
// 秩一更新 st += k·d → o 沿 i 升序累加）、头配对 h % h_k（取模，非交错）。收尾
// y = rms_norm(o, ssm_norm)·sigmoid(z)。状态布局 (S, h_v, S) j 最快，与主机同；本入口把状态
// 拷上设备、kernel 内原地更新、再拷回（跨 token 存活语义与主机一致；常驻优化留后续阶段）。
// 归约顺序（i 升序）与主机完全一致；收尾归一与主机逐元素同式。float 累加（ADR-009 分布级判据）。
//
// 只有 -DQINFER_ENABLE_CUDA=ON 时才编这一支（ADR-008）。
#pragma once

#include <cstdint>
#include <string>

#include "dense/gdn_recurrence.hpp"

#include <cuda_runtime.h>  // cudaStream_t

namespace qinfer::dense {

// 一步递推（设备），状态跨调用经 st 的 data 传入传出（跨 token 存活由调用方维护）。
// q、k 各 (h_k, S) 已 l2 归一且 q 已乘 1/sqrt(S)；v、o 各 (h_v, S)；gate、beta 各 (h_v,)。
// beta 进来前必须已 sigmoid 过（与主机 gdn_step 同契约）。
bool gdn_step_cuda(GdnState& st, const GdnRecShapes& s, const float* q, const float* k,
                   const float* v, const float* gate, const float* beta, float* o,
                   cudaStream_t stream, std::string& err);

// 收尾（设备）：y = rms_norm(o 第 h 行)·ssm_norm[j]·sigmoid(z)。o、z 各 (h_v, S)。
bool gdn_closing_norm_cuda(const GdnRecShapes& s, const float* o, const float* z,
                           const float* ssm_norm, float eps, float* y, cudaStream_t stream,
                           std::string& err);

}  // namespace qinfer::dense