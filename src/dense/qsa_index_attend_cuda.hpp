// QSA 索引器与 attend 的设备实现（阶段 A，Task 7）。
//
// qsa_index_cuda 语义与主机 `dense/qsa_index` 一致：逐 cell `j` 的分数
// `score[j] = bias[cb] + Σ_h relu(dot(pooled[cb], q_idx[h]))`，块映射 `cb = min(j / r, n_bid)`、
// 尾巴 `j >= (n_kv / r)*r` 加 1e9 偏置（即使 bias 为 null 也加）。按 cell 并行（每 cell 一个线程）。
//
// qsa_attend_cuda（含 gate apply）语义与主机 `dense/qsa_attend` 一致：每 q 头 softmax GQA——kv 头
// 映射 `kvh = h / (nh / nkv)`（整除）、分数 = dot(k, q)·scale（scale = 1/sqrt(hd)）、softmax 用
// max 减稳、attn += w·v（fp16 位型解码）；门 `out = attn · sigmoid(gate)`（gate 取每头 2·hd 块的
// 后半段）。逐头并行：每头一个 block，softmax 归约用共享内存（uint64 长度内）。
//
// 只有 -DQINFER_ENABLE_CUDA=ON 时才编这一支（ADR-008）。
#pragma once

#include <cstdint>
#include <string>

#include "dense/qsa_select.hpp"

#include <cuda_runtime.h>  // cudaStream_t

namespace qinfer::dense {

// 逐 cell 分数（qsa_index 的设备版）。pooled (n_bid+1, idx_dim)、q_idx (idx_n_head, idx_dim)，
// bias (n_bid+1) 或 null；cell_scores 长 n_kv，覆盖写。
bool qsa_index_cuda(const float* pooled, std::uint64_t n_bid, std::uint64_t idx_dim,
                    std::uint64_t idx_n_head, const float* q_idx, const float* bias,
                    std::uint64_t idx_block, std::uint64_t n_kv, float* cell_scores,
                    cudaStream_t stream, std::string& err);

// softmax GQA + 门（qsa_attend + qsa_gate_apply_f32 的设备版）。
// q (n_head, head_dim)；k_scratch/v_scratch (n_ids, n_head_kv, head_dim) fp16 位型；
// q_full 是 attn_q 原始投影 (n_head, 2·head_dim)（gate 取后半段）；attn (n_head, head_dim) 输出。
// weights 可空（n_head × n_ids）。n_ids 为 0 时 attn 全 0（合法）。
bool qsa_attend_cuda(const float* q, const std::uint16_t* k_scratch, const std::uint16_t* v_scratch,
                     std::uint64_t n_ids, const QsaShapes& s, const float* q_full, float* attn,
                     float* weights, cudaStream_t stream, std::string& err);

// 设备缓冲直读变体（阶段 B：KV 上卡后 k/v scratch 已在设备，不再每步 H2D）。
// 六个输入/输出指针全部是设备指针（d_q/d_q_full/d_k/d_v 输入，d_attn/d_weights 输出），
// 本函数只做一次设备端归零 + 两次内核启动（attn + gate）；attn/weights 由调用方搬回。
// 语义与 qsa_attend_cuda 完全一致（同一内核、同一归约顺序）。
bool qsa_attend_cuda_dev(const float* d_q, const float* d_q_full,
                         const std::uint16_t* d_k, const std::uint16_t* d_v,
                         std::uint64_t n_ids, const QsaShapes& s, float* d_attn,
                         float* d_weights, cudaStream_t stream, std::string& err);

// BF16 逐行点积的设备版（索引器 q/k 投影；语义与主机 run_bf16_gemv 一致：激活先舍入到 bf16 位型、
// 逐行 bf16 点积、f32 累加、每行独立）。w_bf16 (rows, cols)、x 是 f32 激活（内部舍入）、out 长 rows。
bool bf16_gemv_cuda(const std::uint16_t* w_bf16, std::uint64_t rows, std::uint64_t cols,
                    const float* x, float* out, cudaStream_t stream, std::string& err);

}  // namespace qinfer::dense