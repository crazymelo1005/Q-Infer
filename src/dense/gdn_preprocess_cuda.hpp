// GDN 前段的设备实现（阶段 A：卷积 → SiLU → 拆 q|k|v → q/k 头归一 → 双门 → 状态前移）。
//
// 语义与输出布局与主机 `dense/gdn_preprocess` 一致，好让两边逐元素对照：卷积 = 窗口帧（状态里最老
// 的 win 帧 + 当前 qkv）与 conv1d 权重的点积、头 d_conv-1 个 token 左端补零；SiLU 加在整段 conv 上
// （拆分之前）；q/k 各自按头 l2 归一（v 不归一）；两个门 beta = sigmoid(wbeta@x)、
// gate = softplus(walpha@x + dt)·ssm_a；状态前移 = 新帧进队尾、最老出队。权重精度逐张力读声明
// （conv1d/dt/ssm_a 是 F32，alpha/beta 是 BF16），与主机 w_at / w_at_row 的口径一致。
//
// 只有 -DQINFER_ENABLE_CUDA=ON 时才编这一支（ADR-008）。
#pragma once

#include <cstdint>
#include <string>

#include "dense/gdn_preprocess.hpp"

#include <cuda_runtime.h>  // cudaStream_t

namespace qinfer::dense {

// 设备侧临时缓冲（流内复用，免每 token 分配）。
struct GdnPreCudaScratch {
    void* d_conv = nullptr;    // conv_dim 个 float：K1 输出、K2 输入
    std::size_t conv_bytes = 0;
};

// x 是 n_embd 激活、qkv 是 conv_dim 原始投影（未过卷积）——与主机 gdn_preprocess 同签名。
// state 的 frames 由本入口拷贝上设备、kernel 内前移、再拷回（跨 token 存活）；tokens 由调用方维护。
bool gdn_preprocess_cuda(const GdnShapes& s, const GdnPreWeights& w, const float* x,
                         const float* qkv, GdnConvState& state, float eps, GdnPreOut& out,
                         GdnPreCudaScratch& scratch, std::string& err);

}  // namespace qinfer::dense