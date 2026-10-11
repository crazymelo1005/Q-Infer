#include "dense/token_embed.hpp"

#include "kernels/iq4xs.hpp"

#include <vector>

namespace qinfer::dense {

namespace {

constexpr std::uint32_t kIq4Xs = 23;  // ggml_type 的 IQ4_XS（与 [S-61] 的实测码一致）

}  // namespace

bool build_token_embed(const artifact::GgufFile& gguf, std::uint64_t n_embd, TokenEmbed& out,
                       std::string& err) {
    err.clear();
    out = TokenEmbed{};
    out.t = gguf.find("token_embd.weight");
    if (out.t == nullptr) {
        err = "token_embd.weight 不存在";
        return false;
    }
    if (out.t->dims.size() != 2) {
        err = "token_embd.weight 不是二维";
        return false;
    }
    if (n_embd % kernels::kIq4xsValuesPerBlock != 0) {
        err = "n_embd " + std::to_string(n_embd) + " 不是 256 的整数倍";
        return false;
    }
    if (out.t->dims[0] != n_embd) {
        err = "token_embd.weight 的行宽 " + std::to_string(out.t->dims[0]) + " 与 n_embd " +
              std::to_string(n_embd) + " 不一致";
        return false;
    }
    if (out.t->type != kIq4Xs) {
        err = "token_embd.weight 的类型码是 " + std::to_string(out.t->type) +
              "，本路径只按 IQ4_XS（码 23）读";
        return false;
    }
    out.n_embd = n_embd;
    out.vocab = out.t->dims[1];
    out.row_blocks = n_embd / kernels::kIq4xsValuesPerBlock;
    return true;
}

bool read_token_embedding(const artifact::GgufFile& gguf, const TokenEmbed& e, std::uint64_t token,
                          float* out, std::string& err) {
    err.clear();
    if (out == nullptr) {
        err = "read_token_embedding 的输出为空";
        return false;
    }
    if (e.t == nullptr) {
        err = "TokenEmbed 未构建";
        return false;
    }
    if (token >= e.vocab) {
        err = "token " + std::to_string(token) + " 越界（词表 " + std::to_string(e.vocab) + "）";
        return false;
    }
    const std::uint64_t row_bytes = e.row_blocks * kernels::kIq4xsBlockBytes;
    std::vector<std::uint8_t> raw(static_cast<std::size_t>(row_bytes));
    if (!gguf.read_at(e.t->offset + token * row_bytes, raw.data(), raw.size())) {
        err = "读 token_embd 的第 " + std::to_string(token) + " 行失败";
        return false;
    }
    for (std::uint64_t b = 0; b < e.row_blocks; ++b) {
        kernels::dequant_iq4xs_block(raw.data() + static_cast<std::size_t>(b) * kernels::kIq4xsBlockBytes,
                                     out + b * kernels::kIq4xsValuesPerBlock);
    }
    return true;
}

std::size_t greedy_argmax(const float* v, std::size_t n) {
    if (n == 0) return 0;
    std::size_t best = 0;
    for (std::size_t i = 1; i < n; ++i) {
        if (v[i] > v[best]) best = i;
    }
    return best;
}

}  // namespace qinfer::dense
