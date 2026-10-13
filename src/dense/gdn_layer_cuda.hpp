// GDN 整层设备执行（阶段 A：把已验收的四段设备内核串成一层）。
//
// 顺序照主机 `run_gdn_layer`：qkv = attn_qkv @ x（gdn_proj_gemv）→ gdn_preprocess_cuda（卷积+SiLU
// +拆 q|k|v+归一+双门）→ z = attn_gate @ x → q 乘 1/sqrt(S) → gdn_step_cuda（递推，状态原地）+
// gdn_closing_norm_cuda → out = ssm_out @ y。三处投影走 gdn_proj_gemv 统一入口（真模型档：
// attn_qkv/ssm_out 是 IQ4_XS、attn_gate 是 IQ3_S），与主机 run_gemv 同一分派来源。
//
// 本轮是「正确性对照」形态：权重按层缓存（host 侧每层读一次，`w_layer` 记账换层重读，与主机
// GdnScratch 同语义），每段设备内核各自的 H2D/D2H 保留（每段独立验收过）；设备侧同流常驻与
// 每步单汇合交给 Task 9 的 CudaGroupFleet 接线。状态与 conv 跨 token 存活（gdn_preprocess_cuda /
// gdn_step_cuda 已把状态带出）。只有 -DQINFER_ENABLE_CUDA=ON 时才编这一支（ADR-008）。
#pragma once

#include <cstdint>
#include <string>

#include "dense/gdn_layer.hpp"

namespace qinfer::dense {

// x 长 n_embd、out 长 n_embd（覆盖写）。state 与 conv 跨 token 存活（经 s 传入传出）。
// 需要持有本层的权重字节（s 的 w_* 缓冲），首次读到该层时填充、换层重读——与主机 GdnScratch 同款
// 的 w_layer 记账。
bool run_gdn_layer_cuda(const artifact::GgufFile& gguf, const GdnLayer& layer, const float* x,
                        float* out, GdnScratch& s, std::string& err);

}  // namespace qinfer::dense