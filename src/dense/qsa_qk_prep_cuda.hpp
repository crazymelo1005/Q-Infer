// QSA 的 q/k 预处理设备实现（阶段 A）。
//
// 语义与主机 `dense/qsa_qk_prep` 一致：从每头 2·head_dim 的 q_full 块拆出 q（前 head_dim）与 gate
// （后 head_dim，**不**按元素交错）；q/k 各自按头 rms_norm（分母 sqrt(mean(x²)+eps)、先归一再乘
// gamma）；部分 NEOX rope（只转每头前 n_rot 维，配对 (i, i+n_rot/2)，cos/sin 表按位置查询）；
// gate 原样拷出（不过归一不过 rope）。host 侧建表（double 算角），表拷上设备。
//
// 只有 -DQINFER_ENABLE_CUDA=ON 时才编这一支（ADR-008）。
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "dense/qsa_qk_prep.hpp"

#include <cuda_runtime.h>  // cudaStream_t

namespace qinfer::dense {

// 设备侧临时缓冲（图内复用）：cos/sin 表拷上设备（构建一次）。
struct QsaQkPrepCudaScratch {
    void* d_cos = nullptr;
    void* d_sin = nullptr;
    std::size_t tab_bytes = 0;
};

// q_full 是 attn_q 的原始输出（n_head × 2·head_dim）；k_raw 是 attn_k 的原始输出。
// 输出 q_ready / k_ready（已归一已 rope）、gate（未归一未 rope）。pos 是当前 token 位置。
bool qsa_qk_prep_cuda(std::uint64_t n_head, std::uint64_t n_head_kv, std::uint64_t head_dim,
                      std::uint64_t n_rot, std::uint64_t pos, const float* q_full,
                      const float* k_raw, const QsaQkWeights& w,
                      const std::vector<float>& cos_tab, const std::vector<float>& sin_tab,
                      float* q_ready, float* k_ready, float* gate,
                      QsaQkPrepCudaScratch& scratch, cudaStream_t stream, std::string& err);

}  // namespace qinfer::dense