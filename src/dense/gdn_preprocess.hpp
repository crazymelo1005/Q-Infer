// GDN（Gated Delta Net）一层的前半段：卷积 → SiLU → 拆 q|k|v → q/k 的 l2 归一 → 两个门。
//
// 顺序取自参考引擎 `include/strata/core/layer.hpp` 的文件头，它把 `ref/gdn.py::gdn_decode` 的
// 顺序逐字抄了下来（[S-53] 同源说明）：
//
//     qkv  = wqkv @ x                   (10240 = q|k|v，各是 128 宽头的连续段)
//     conv -> SiLU                      (SiLU 加在**整段** conv 输出上，在拆分之前)
//     q,k  = l2_norm                    (v 不做)
//     beta = sigmoid(wbeta @ x)
//     gate = softplus(walpha @ x + dt) * ssm_a
//     ...（循环与 y = rms_norm(o, ssm_norm)·sigmoid(z)、out = ssm_out @ y 不在本文件里）
//
// 四处「一读就过、错值却可信」的地方，每一处都在测试里按错的算一遍要求它明显偏离：
//   1. SiLU 加在整段 conv 输出上（拆成 q/k/v 之后再加，是同一套算术作用在别的数据上）；
//   2. v **不做** l2 归一（归一作用在 q 与 k 各自的头上，不是 v 那 48 个头）；
//   3. 门上是 **softplus** 不是 sigmoid（`softplus(walpha@x + dt) · ssm_a`，且 dt 是加在 softplus
//      里面的偏置——把它当乘子或加在 softplus 外面都是能算出一个漂亮数的那种错）；
//   4. conv 的状态是**移位**的：新帧进队尾、最老的出队，取的是 `d_conv` 帧与权重的点积。
//      llama.cpp 的实现在时间维上把权重与状态对齐（最早一帧配权重第 0 行），错一位看不出来。
//
// 两处口径边界：`wqkv @ x` 与这里要用的 `qkv` 是同一个东西，但本文件**收 qkv 作为输入**——
// 那两个投影是普通的量化 GEMV（attn_qkv 是 IQ4_XS/IQ3_S，见 [S-54]），与本仓库已有的量化内核是
// 一件事，接线时再补；本文件负责的是顺序、状态与逐头的那几步。权重精度逐张力读声明（`GrTensor`
// 同一套口径）：`ssm_conv1d` / `ssm_a` / `ssm_dt` 是 F32，`ssm_alpha` / `ssm_beta` 是 BF16。
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "dense/gated_residual.hpp"  // 复用 GrTensor / GrPrecision（逐张力声明精度）

namespace qinfer::dense {

// 一层的几何。真模型（[S-54]）：n_embd 2560、q/k/v 头数 20/16/48、头宽 128、卷积核长 4。
struct GdnShapes {
    std::uint64_t n_embd = 0;
    std::uint64_t n_q_head = 0;
    std::uint64_t n_k_head = 0;
    std::uint64_t n_v_head = 0;
    std::uint64_t head_dim = 0;
    std::uint64_t d_conv = 0;

    std::uint64_t q_dim() const { return n_q_head * head_dim; }
    std::uint64_t k_dim() const { return n_k_head * head_dim; }
    std::uint64_t v_dim() const { return n_v_head * head_dim; }
    std::uint64_t conv_dim() const { return q_dim() + k_dim() + v_dim(); }
    bool sane() const {
        return n_embd != 0 && head_dim != 0 && d_conv > 0 && n_q_head != 0 && n_k_head != 0 &&
               n_v_head != 0;
    }
};

// 两个门的权重与偏置，加卷积核。alpha/beta 是 BF16（[n_embd, n_v_head] 行主序），其余是 F32。
struct GdnPreWeights {
    GrTensor conv1d;  // d_conv × conv_dim，F32；行 r 是第 r 个时间偏移的核
    GrTensor alpha;   // n_embd × n_v_head，BF16
    GrTensor beta;    // n_embd × n_v_head，BF16
    const float* dt = nullptr;      // n_v_head
    const float* ssm_a = nullptr;   // n_v_head
};

// 跨 token 存活的状态：卷积窗口的前 d_conv-1 帧（conv_dim 宽），行主序、最老在前。
struct GdnConvState {
    std::vector<float> frames;  // (d_conv-1) × conv_dim
    std::uint64_t tokens = 0;   // 已吃进多少 token（前 d_conv-1 个 token 的窗口左侧补零）
};

// 一层的输出：归一后的 q、k（各按头），未归一的 v，两个门（各 n_v_head）。
struct GdnPreOut {
    std::vector<float> q;      // q_dim，按头归一过
    std::vector<float> k;      // k_dim
    std::vector<float> v;      // v_dim，未归一
    std::vector<float> beta;   // n_v_head
    std::vector<float> gate;   // n_v_head
};

// x 是 n_embd 的激活（两个门用它），qkv 是 conv_dim 的原始投影（**未过卷积**）。
// eps 是 l2 归一的分母下限（参考实现用 1e-6 量级；见 gdn.hpp 的 l2 归一）。
bool gdn_preprocess(const GdnShapes& s, const GdnPreWeights& w, const float* x, const float* qkv,
                    GdnConvState& state, float eps, GdnPreOut& out, std::string& err);

}  // namespace qinfer::dense