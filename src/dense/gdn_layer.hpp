// 一整层 GDN（主机侧）。把已有的四段接起来，顺序取参考引擎 `layer.hpp` 的文件头（[S-55]）：
//
//   qkv = attn_qkv @ x            (量化投影，逐层换档：[S-54])
//   ── gdn_preprocess ──          卷积+SiLU → 拆 q|k|v → q/k 的 l2 归一 → beta/gate 两个门
//   z   = attn_gate @ x           (量化投影；收尾那个 sigmoid 的输入)
//   q 再乘 1/sqrt(S)              (**调用方**的活，见 [S-56]：两条 llama.cpp 路径把这步尺度放在不同位置)
//   ── gdn_recurrence ──          delta 规则递推（状态原地更新）→ y = rms_norm(o,ssm_norm)·sigmoid(z)
//   out = ssm_out @ y             (量化投影，Q8_0)
//
// 三处投影走的是与专家矩阵同一份「档 → 内核 + 激活搭档」分派表（`experts` 的 `quantize_input` +
// `run_gemv`），不另立一张——注意力与专家在这件事上没有区别，重复的表差一处就是静默错值。
//
// 权重精度逐张力读声明（[S-53] 的口径）：`ssm_conv1d`/`ssm_a`/`ssm_dt`/`ssm_norm` 是 F32，
// `ssm_alpha`/`ssm_beta` 是 BF16；三个投影是量化档。
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "artifact/gguf_table.hpp"
#include "dense/gdn_preprocess.hpp"
#include "dense/gdn_recurrence.hpp"
#include "experts/expert_formats.hpp"
#include "experts/ffn.hpp"  // FfnScratch（量化 GEMV 复用的激活像缓冲）

namespace qinfer::dense {

// 一层的静态描述：三个量化投影的档与块几何 + 其余张力的登记项。构建一次、逐 token 复用。
// 里面的指针指向 GgufFile 内部的张力表，故 GgufFile 在 GdnLayer 存活期间不得重新 open。
struct GdnLayer {
    int layer = -1;
    GdnShapes pre;                  // n_embd / 头数 / 头宽 / 卷积核长
    GdnRecShapes rec;               // S / h_k / h_v（与 pre 必须自洽）
    experts::MatrixSpec qkv, gate, out;
    const artifact::GgufTensorInfo* t_qkv = nullptr;
    const artifact::GgufTensorInfo* t_gate = nullptr;
    const artifact::GgufTensorInfo* t_out = nullptr;
    const artifact::GgufTensorInfo* t_conv1d = nullptr;   // [d_conv, conv_dim] F32（GGML 原生布局）
    const artifact::GgufTensorInfo* t_alpha = nullptr;    // [n_embd, n_v_head] BF16
    const artifact::GgufTensorInfo* t_beta = nullptr;     // 同上
    const artifact::GgufTensorInfo* t_dt = nullptr;       // [n_v_head] F32
    const artifact::GgufTensorInfo* t_ssm_a = nullptr;    // [n_v_head] F32
    const artifact::GgufTensorInfo* t_norm = nullptr;     // [head_dim] F32（每 v 头共用）
};

// 按张力名建一层。名字取 llama.cpp 的表（[S-53] 同源）：GDN 层是 `attn_qkv` / `attn_gate` /
// `ssm_*`；若这一层没有 `attn_qkv`（即 QSA 层）或任一必需张力缺失、几何不自洽，则返回假并写 err。
bool build_gdn_layer(const artifact::GgufFile& gguf, int layer, std::uint64_t n_embd,
                     std::uint64_t head_dim, std::uint64_t d_conv, GdnLayer& out, std::string& err);

struct GdnScratch {
    GdnConvState conv;
    GdnState state;
    GdnPreOut pre;
    experts::FfnScratch gemv;
    std::vector<std::uint8_t> w_conv1d, w_alpha, w_beta, w_qkv, w_gate, w_out;
    std::vector<float> dt, ssm_a, ssm_norm, qkv, z, y, out;
    std::vector<float> q_scaled;  // q 乘过 1/sqrt(S) 的那一份
};

// 跑一层：x 长 n_embd，out 长 n_embd（覆盖写）。state 与 conv 跨 token 存活。
bool run_gdn_layer(const artifact::GgufFile& gguf, const GdnLayer& layer, const float* x, float* out,
                   GdnScratch& s, std::string& err);

}  // namespace qinfer::dense