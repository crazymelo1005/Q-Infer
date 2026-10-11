// MTP 草稿层的入口：把「当前 token 的 embedding」与「主模型在该位置的最终多流残差」合成该层的初值 R。
//
// 算法取自上游两处逐句对应的实现（[S-62]）：参考引擎 `src/core/mtp.cpp` l.488-497 与
// vLLM v0.31.0 `vllm/models/qwen4_exp/nvidia/mtp.py` l.190-307（后者注释明写 fc_hidden 是
// shared across HC branches）：
//
//   en      = rms_n(tok_embd) ⊙ gamma_embed          （N 宽）
//   hn      = rms_{HC·N}(R_in) ⊙ gamma_hidden         （**整栈**一次归一，不是逐流）
//   引擎那份：h2[c] = fc_hidden(hn 的第 c 条流)  ；R[c] = h2[c] + en   （en 广播到每条流）
//   社区那份：R[c] = eh_proj( concat(en, hn 的第 c 条流) )             （两张矩阵融进 eh_proj）
//
// 两份 pack 在数学上同一件事：`eh_proj` 是 `fc_embedding` 与 `fc_hidden` 沿**输入轴**叠起来的结果
// （5120 = 2560 + 2560），故把 concat 喂进去等价于「两张各算一次再相加」。本模块两方言都实现，
// 且在合成侧用**同一组权重值**分别编码成两份 pack、断言两条路径给出同一个 R —— 把这条上游结论钉住。
//
// 三处一读就顺、错值却可信的地方（测试里各按错的读法算一遍，要求与对的读法有明显差别）：
//   ① `en` 是**广播**到每条流的（`add_streams_broadcast`），不是只加在第一条流上；
//   ② 残差那支的归一在**整栈 HC·N** 上做（一条 RMS 管 10240），不是逐流各一条；
//   ③ `fc_hidden` 的输入是**单条流**的 N（同一张矩阵对 HC 条流各算一次），不是整栈 HC·N。
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "artifact/gguf_table.hpp"
#include "dense/mtp_pack.hpp"
#include "experts/ffn.hpp"

namespace qinfer::dense {

// 入口用到的权重字节与中间缓冲（构建一次、逐 token 复用）。
struct MtpEntryScratch {
    // 按每个张力自己声明的档读进来的字节（融合的那几张按切片取行区间）。
    std::vector<std::uint8_t> w_gamma_embed, w_gamma_hidden;      // 一维 F32 视作字节
    std::vector<std::uint8_t> w_fc_embed, w_fc_hidden, w_eh_proj;
    std::vector<float> gamma_embed, gamma_hidden;  // 一维 gamma 解出成 float
    experts::FfnScratch gemv;
    std::vector<float> en, hn, h2, tmp;
};

// 入口：`R_in` 是 (hc, n_embd) 的多流残差、`tok_embd` 是长 n_embd 的 embedding 行；
// 输出 `R_out` 同形（覆盖写）。两份 pack 都支持。任一处读不出/档没内核/形状不符都返回假并写 err。
bool mtp_entry(const artifact::GgufFile& gguf, const MtpPack& pack, const float* tok_embd,
               const float* R_in, MtpEntryScratch& s, float* R_out, std::string& err);

}  // namespace qinfer::dense