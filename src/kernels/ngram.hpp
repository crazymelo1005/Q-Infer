// n-gram 表行索引：由 token 序列确定性地算出每个位置要读的 16 个表行。
//
// 规则与常量取自参考引擎的实现（见 S-34），并已与它自带的 6 组 oracle 向量逐位比对通过
// （measure/ple_locality.py --selftest）。三处易错且都会给出「可信但错误」的索引：
//
//   * 用异或而不是求和：两个相同的 token 在异或下抵消、在求和下翻倍。
//   * 对 vocab 取模而不是位与：16 个头的空间大小都不是 2 的幂。
//   * EOS 截断向前传播：某位是 EOS，则它及其更旧的前序一律按 EOS 参与哈希；
//     但 token 自身的 EOS 不截断自身的上下文。缺前序按 -1 处理，等价于一次 EOS 截断。
#pragma once

#include <cstdint>

namespace qinfer::kernels {

inline constexpr int kNgramSize = 3;
inline constexpr int kHeadsPerNgram = 8;
inline constexpr int kPleNHeads = (kNgramSize - 1) * kHeadsPerNgram;  // 16
inline constexpr std::int32_t kPleEosTokenId = 248044;
inline constexpr std::int32_t kTokenNull = -1;

struct PleConsts {
    std::uint64_t mult[kNgramSize];
    std::uint64_t vocab[kPleNHeads];
    std::uint64_t offset[kPleNHeads];
};

const PleConsts& ple_artifact_consts();

// mixed_n = (ctx[0]·m[0]) ^ (ctx[1]·m[1]) ^ …，逐项按 2^64 取模。
std::uint64_t ngram_mixed(const std::int64_t* ctx, const std::uint64_t* mult, int n);

// tokens：n_tokens 个 token；prev：n_tokens×2，越靠前越旧（[i*2+0] 是 i-2 位、[i*2+1] 是 i-1 位），
// 无前序处填 kTokenNull；out：n_tokens×16，token-major。
void ngram_rows(const std::int32_t* tokens, const std::int32_t* prev, int n_tokens,
                const PleConsts& c, std::uint32_t* out);

}  // namespace qinfer::kernels
