// 投机解码的验证环（MTP 的机制面）。基线口径是贪心（[01-strata-engine.md] §10.3），故这里也按贪心验证：
// 草稿 token 逐个与目标模型的贪心选择比对，**接受尽可能长的前缀**；在首个分歧处改用目标模型自己的 token
// 并把它也算作本轮产出（bonus）；草稿全部被接受时再取一个 bonus。
//
// 三条性质（测试里都钉住）：
//   1. **投机不改变输出序列**——接受/拒绝只影响模型步数，不改变贪心轨迹（这是这个机制成立的前提）；
//   2. 草稿全对时一轮产出 k+1 个 token、只用 k 次模型前向（草稿 0 个时一轮正好 1 个 token、1 次前向）；
//   3. 状态与 logits 必须同步推进：拒绝时**不消费**被拒的那个草稿 token，接受时才消费——错一处就会
//      让后续轨迹整体错位。
//
// 本片只做机制与状态机，不含真模型的 MTP 头（草稿由调用方给）。真机的草稿来源是模型自带的 MTP 模块，
// 它的张力清单尚未盘（见 engine §6）。
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "artifact/gguf_table.hpp"
#include "dense/step_loop.hpp"
#include "dense/token_embed.hpp"

namespace qinfer::dense {

// 验证环持有的最小状态：对**下一个位置**的预测。它由最近一次 `spec_commit` 刷新。
struct SpecDecoder {
    std::vector<float> logits;
    bool primed = false;
};

// 把 token 喂进模型并刷新 logits（也就是把状态推进一格）。第一个 token 之后 `primed` 为真。
bool spec_commit(const artifact::GgufFile& gguf, const StepModel& model, const TokenEmbed& e,
                 std::uint64_t token, StepScratch& s, SpecDecoder& d, std::string& err);

// 一次投机：验证 `draft[0..k-1]`。`out` 是本轮**确认**的 token 序列（接受的草稿前缀 + 末尾那个由目标
// 模型给出的 token），并把模型状态推进到该序列之后。要求先 `spec_commit` 过一次（否则返回假）。
bool spec_step(const artifact::GgufFile& gguf, const StepModel& model, const TokenEmbed& e,
               const std::uint64_t* draft, std::size_t k, StepScratch& s, SpecDecoder& d,
               std::vector<std::uint64_t>& out, std::string& err);

}  // namespace qinfer::dense
