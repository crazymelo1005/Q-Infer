// token embedding 的行读：把 token id 变成残差栈的初值来源。这是步循环之外那半边的入口——[S-61] 说
// 「真序列的首个 token 就是 embedding 广播到每条流」，而广播进去的那个向量就是这里读出来的这一行。
//
// 张力名 `token_embd.weight`（llama.cpp 的表，与 [S-54] 的直读一致）；真模型里它是
// **[2560 × 248320] 的 IQ4_XS**（[S-61] 补：码 23），一行 2560 个值 = 10 个 256 值块。
//
// 两点口径：**行号是 token id、行距是 `n_embd/256 × 136` 字节**（不是按元素数），且**逐元素反量化**
// （embedding 行不做点积）。本路径只收它自己声明的那一个档（IQ4_XS）——声明成别的档就明确报错，
// 不静默按别的布局解（「格式类事实必须从已核对的来源读」，见 S-36 的教训）。
#pragma once

#include <cstdint>
#include <string>

#include "artifact/gguf_table.hpp"

namespace qinfer::dense {

struct TokenEmbed {
    std::uint64_t n_embd = 0;
    std::uint64_t vocab = 0;
    std::uint64_t row_blocks = 0;  // n_embd / 256
    const artifact::GgufTensorInfo* t = nullptr;
};

// 校验 token_embd.weight 在：二维、`dims[0] == n_embd`、类型是 IQ4_XS（码 23）、n_embd 是 256 的倍数。
// 任一条不成立都返回假并写 err（含张力名）。
bool build_token_embed(const artifact::GgufFile& gguf, std::uint64_t n_embd, TokenEmbed& out,
                       std::string& err);

// 读 token 那一行到 out（长 n_embd）。token 越界或读失败返回假。
bool read_token_embedding(const artifact::GgufFile& gguf, const TokenEmbed& e, std::uint64_t token,
                          float* out, std::string& err);

// 贪心取最大值的下标（并列取小索引）。基线口径就是贪心（tag design-v0.1 的 docs/research/01 §10.3）。
std::size_t greedy_argmax(const float* v, std::size_t n);

}  // namespace qinfer::dense
