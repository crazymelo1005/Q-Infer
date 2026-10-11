#include "dense/spec_decode.hpp"

namespace qinfer::dense {

bool spec_commit(const artifact::GgufFile& gguf, const StepModel& model, const TokenEmbed& e,
                 std::uint64_t token, StepScratch& s, SpecDecoder& d, std::string& err) {
    if (!run_token_step(gguf, model, e, token, s, err)) return false;
    d.logits = s.head.logits;
    d.primed = true;
    return true;
}

bool spec_step(const artifact::GgufFile& gguf, const StepModel& model, const TokenEmbed& e,
               const std::uint64_t* draft, std::size_t k, StepScratch& s, SpecDecoder& d,
               std::vector<std::uint64_t>& out, std::string& err) {
    err.clear();
    out.clear();
    if (!d.primed) {
        err = "投机步之前要先 spec_commit 一次（否则没有对下一个位置的预测）";
        return false;
    }
    if (k > 0 && draft == nullptr) {
        err = "草稿指针为空但长度不为 0";
        return false;
    }
    for (std::size_t j = 0; j < k; ++j) {
        const std::uint64_t greedy =
            static_cast<std::uint64_t>(greedy_argmax(d.logits.data(), d.logits.size()));
        if (greedy != draft[j]) {
            // 分歧：丢弃这个草稿 token 与它之后的，改用目标模型自己的 token，并推进一格。
            out.push_back(greedy);
            return spec_commit(gguf, model, e, greedy, s, d, err);
        }
        // 一致：接受它（消费掉，推进一格），继续看下一个草稿。
        out.push_back(draft[j]);
        if (!spec_commit(gguf, model, e, draft[j], s, d, err)) return false;
    }
    // 草稿全部被接受：再取一个目标模型自己的 token 作为本轮最后一个产出。
    const std::uint64_t bonus =
        static_cast<std::uint64_t>(greedy_argmax(d.logits.data(), d.logits.size()));
    out.push_back(bonus);
    return spec_commit(gguf, model, e, bonus, s, d, err);
}

}  // namespace qinfer::dense
