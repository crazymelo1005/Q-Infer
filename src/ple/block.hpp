// 第 1 层的 PLE 块：把一条 2560 宽的 n-gram 表行向量并进 hc 条流的残差栈。
//
// 口径取自参考引擎的 `include/strata/kernels/ple.hpp`（契约）与 `src/kernels/cuda/ple.cu`（可执行的
// 定义）、`src/kernels/ple_parity.cpp` 里那段主机参考实现（第三份写法，本文件的算术逐句照它）——见
// [S-63]。`emb` 是表行按 head-slowest 拼出的那 2560（`ple/row_path` 给），`hidden` 是 hc 条流的残差栈：
//
//   key        = grouped_norm(ple_key @ emb)              hc_dim，逐流 RMS 再逐维乘 gamma
//   query      = grouped_norm(hidden, ple_norm_query)
//   value      = ple_value @ emb                          n_embd
//   s[c]       = Σ_d key[c][d]·query[c][d] / sqrt(n_embd)
//   gate[c]    = sigmoid(sign(s)·sqrt(max(|s|, 1e-6)))    带符号的平方根，再 sigmoid
//   gated[c][d]= value[d]·gate[c]                         value 在 hc 条流上广播
//   normalized = grouped_norm(gated, ple_norm_conv)       **卷积的输入是它，不是 gated**
//   conv[c]    = silu(Σ_k kW[k][c]·row(conv[i] 的四个抽头))
//   out[i]     = hidden[i] + gated[i] + conv[i]
//
// 三处「写错了也长得对」的地方（引擎自己点名，测试里各按错的算一遍并要求可观测）：
//   * 卷积的输入是 normalized（gated 的逐流归一），不是 gated 本身；
//   * `ple_conv1d.weight` 是 ggml 原生布局 `kW[k + kConvKernel·c]`（k 快、c 慢）；转置读法
//     `kW[k·hc_dim + c]` 是「自然写法」，读出来是转置的核；
//   * 历史是 row-fastest（`hist[c·kHistRows + r]`，同一通道的 9 行连续），通道慢的读法读到转置的状态。
//
// 两处与 `dense/norm.hpp` 的 `rms_norm_rows` 不同、不能顺手复用的地方（引擎的注释点名过第一处）：
// 平方和是 `Σ (double)(x·x)`（乘积先在 f32 里舍一遍再放宽，不是先放宽再相乘），尺度是 f32 的
// `1/sqrtf(mean + eps)`（不是 `(float)(1/sqrt(double))`）。两处都在 `grouped_norm` 里照抄。
//
// 历史的推进**不在这里**：与引擎同形，这个块是它输入的纯函数，调用方持有状态并调
// `ple_history_advance`（这样「一段 vs 多次」才可测）。残差是就地改写的（`result` 可与 `hidden` 同址）。
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "kernels/ngram.hpp"

namespace qinfer::ple {

// 卷积抽头数与历史行数。引擎的常量表里是 `PLE_CONV_KERNEL = 4`、`NG_HIST = (4-1)·ngram_size = 9`；
// 这两个值在 GGUF 的元数据里也是 4 与 3（`qwen4exp.ple.conv_kernel` / `ple.ngram_size`），见 [S-63]。
inline constexpr int kConvKernel = 4;
inline constexpr int kHistRows = (kConvKernel - 1) * kernels::kNgramSize;  // 9
inline constexpr float kPleRmsEps = 1e-6f;

struct BlockShapes {
    std::uint64_t n_embd = 0;  // 2560
    std::uint64_t hc = 0;      // 4 条残差流
    std::uint64_t hc_dim() const { return n_embd * hc; }
    bool sane() const { return n_embd > 0 && hc > 0; }
};

// 六张张力的视图。类型按张力自己声明的读（两张投影是 BF16、三条 gamma 是 F32、卷积核是 F16），
// 投影的激活搭档随权重走（BF16 的张力两边都按 bf16 舍入，见 kernels/bf16.hpp 与 [S-44]）。
struct BlockWeights {
    const std::uint16_t* key_bf16 = nullptr;    // [hc_dim, n_embd]，行距 n_embd
    const std::uint16_t* value_bf16 = nullptr;  // [n_embd, n_embd]
    const float* norm_key = nullptr;            // hc_dim
    const float* norm_query = nullptr;          // hc_dim
    const float* norm_conv = nullptr;           // hc_dim
    const std::uint16_t* conv1d_f16 = nullptr;  // [kConvKernel, hc_dim]，`k + kConvKernel·c`
    bool complete() const {
        return key_bf16 != nullptr && value_bf16 != nullptr && norm_key != nullptr &&
               norm_query != nullptr && norm_conv != nullptr && conv1d_f16 != nullptr;
    }
};

// 逐段输出，都给上能让「对不上」指到具体是哪一段（引擎的 PleOut 同形；只要结果就只给 result）。
// 每个非空指针都是调用方自己的缓冲，长度见注释；`result` 可与 `hidden` 同址。
struct BlockOut {
    float* key = nullptr;         // hc_dim
    float* value = nullptr;       // n_embd
    float* gate = nullptr;        // hc
    float* gated = nullptr;       // hc_dim
    float* normalized = nullptr;  // hc_dim
    float* conv = nullptr;        // hc_dim（已过 silu）
    float* result = nullptr;      // hc_dim（必须给：残差的新值）
};

// 中间结果的家（主机侧不分配：与引擎一样由调用方持有，好让 `ple_block` 是个纯函数）。
struct BlockScratch {
    std::vector<float> key, query, value, gated, normalized, conv, gate;
    bool resize(const BlockShapes& g) {
        if (!g.sane()) return false;
        key.resize(static_cast<std::size_t>(g.hc_dim()));
        query.resize(static_cast<std::size_t>(g.hc_dim()));
        value.resize(static_cast<std::size_t>(g.n_embd));
        gated.resize(static_cast<std::size_t>(g.hc_dim()));
        normalized.resize(static_cast<std::size_t>(g.hc_dim()));
        conv.resize(static_cast<std::size_t>(g.hc_dim()));
        gate.resize(static_cast<std::size_t>(g.hc));
        return true;
    }
};

// 跑一个 token。`emb` 长 n_embd；`hidden`/`result` 长 hc_dim；`hist` 长 kHistRows×hc_dim（row-fastest，
// 最旧的在前）。返回假时 err 写明（几何不成立、权重不全、指针缺失）。
bool ple_block(const float* emb, float* hidden, const float* hist, const BlockWeights& w,
               const BlockShapes& g, BlockOut& out, BlockScratch& s, std::string& err);

// 投影已经算好的那一半：`key_raw` 长 hc_dim（**会被就地归一**，故必须可写——与引擎的主机参考一致）、
// `value_raw` 长 n_embd。与引擎的 `ple_block_projected` 同形，用来把「结构」在与激活契约无关的权重下
// 核一遍（夹具要按 F32 权重、不做激活转换时走这条），也让「投影有没有被重算」成为一个可比的量。
bool ple_block_projected(float* key_raw, const float* value_raw, float* hidden, const float* hist,
                         const BlockWeights& w, const BlockShapes& g, BlockOut& out, BlockScratch& s,
                         std::string& err);

// 历史推进：每个通道左移一格、末行写 normalized。就地、逐通道——**不是整块 memmove**（那样会把通道
// 搅在一起：引擎把这一点单列成一条注释，见 [S-63]）。
void ple_history_advance(float* hist, const float* normalized, const BlockShapes& g);

}  // namespace qinfer::ple