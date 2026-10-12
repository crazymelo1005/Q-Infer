// Qwen 系的字节级 BPE 分词器。口径转录自引擎自带的那份实现（`tools/strata_tokenizer.py`，登记 [S-65]）
// ——它在文件头自述是「the tokenizer, as the reference implementation of record」，本仓库照它写 C++ 侧。
//
// 三处口径，每一处错了都仍然产出「看着合理」的 token id，故每处都在测试里单列：
//   * 字节 ↔ 可打印字符的映射（GPT-2 那套）：256 个字节里 68 个按 256+n 顺移，**顺移的顺序**错了在
//     token id 上看不出来；判据是逐字节往返（字节层无损，故是精确的）。
//   * 预分词正则（`qwen35`）：逐字取自 llama.cpp 的 `LLAMA_VOCAB_PRE_TYPE_QWEN35`，关键是
//     `[\p{L}\p{M}]` 而不是 `\p{L}`——同族的 QWEN3 模式长得几乎一样、往返也过，但少了 `\p{M}` 就会把
//     合并记号并进后面的标点里（上游实测：1875 个串里错 3 个，只有与 oracle 逐 token 对照才看得见）。
//   * 特殊 token 分两类：type 4（USER_DEFINED，如 ` thinking`）**永远**按字面匹配；type 3（CONTROL，
//     如 `<|im_start|>`）只在 parse_special 时匹配。这是引擎那边实测出来的口径（把两类同等对待会把
//     `<|im_end|>` 拆成普通片段，chat 提示词就变了）。
//
// Unicode 分类表（`unicode_ranges.inc`）是**用 oracle 自己的 `regex` 库**逐码点扫出来生成的（见
// `measure/gen_unicode_ranges.py`），故「哪一段算字母」与它逐条一致，不靠猜、也不靠别的 Unicode 版本。
//
// 本实现不做引擎那份里的 `plain`（把引用里的字面量当普通文本，上游 #537）：本仓库的调用面是自己渲染
// chat 模板，不存在「用户消息里带特殊字面量」这条输入。
#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "artifact/gguf_table.hpp"

namespace qinfer::text {

class Tokenizer {
public:
    // 从 GGUF 的元数据建：`tokenizer.ggml.{tokens,merges,token_type,pre}` 与各 `*_token_id`。
    // 缺 tokens/merges 或 model 不是 gpt2 就明确失败（那份 pack 不带分词器）。
    static bool from_gguf(const artifact::GgufFile& g, Tokenizer& out, std::string& err);

    // 直接给词表与合并规则（夹具用）。`token_types` 可空。
    bool init(std::vector<std::string> tokens, std::vector<std::string> merges,
              const std::vector<std::uint64_t>& token_types, std::string& err);

    // 编码。`parse_special` 为真时连 type 3（CONTROL）也按字面匹配。
    bool encode(const std::string& text, bool parse_special, std::vector<std::int32_t>& out,
                std::string& err) const;

    // 解码成**原始字节**（字节层无损，故与输入的字节相同）。
    std::string decode_bytes(const std::vector<std::int32_t>& ids) const;
    // 解码成 UTF-8 文本：不能构成合法序列的部分按 U+FFFD 替换（与 Python 的 errors="replace" 同口径）。
    std::string decode(const std::vector<std::int32_t>& ids) const;

    // 一个 token 的原始字节。
    bool token_bytes(std::int32_t id, std::string& out) const;

    // 单个字节的映射形态（字节级 BPE 的第一步）。**测试要直接断言它**：映射是 256 个码点的一个置换，
    // 只测「编码再反解」照不出置换错（两边用同一张错表时往返仍然成立）。
    std::string byte_piece(unsigned char b) const;

    std::size_t vocab_size() const { return tokens_.size(); }
    std::size_t merge_count() const { return ranks_.size(); }
    const std::string& pre_type() const { return pre_; }
    // 特殊字面量的 id（`<|im_start|>` 等）；不在词表里返回 -1。
    std::int32_t special_id(const std::string& literal) const;
    // 词表里某个 token 字符串的 id（按映射后的形态比对，供测试用）；没有返回 -1。
    std::int32_t id_of(const std::string& token) const;
    // 编码一个「普通片段」（不经特殊字面量）：预分词 + 逐片段 BPE。留给测试单测预分词。
    bool encode_plain(const std::string& text, std::vector<std::int32_t>& out, std::string& err) const;

    // 预分词：把文本切成片段（返回每段的字节区间）。测试用它单测 `\p{M}` 那条。
    std::vector<std::pair<std::size_t, std::size_t>> pre_tokenize(const std::string& text) const;

    // 一个片段的 BPE 结果（映射后的符号串），供测试。
    std::vector<std::string> bpe_of_piece(const std::string& piece) const;

private:
    // 长片段的合并：链表 + 以 (次序, 位置) 为键的堆。弹出的那对若已被别的合并改过就作废跳过。
    std::vector<std::string> bpe_heap(const std::vector<std::string>& parts) const;

    std::vector<std::string> tokens_;
    std::string pre_;
    std::unordered_map<std::string, std::int32_t> ids_;         // 映射后的 token 串 → id
    std::unordered_map<std::string, std::int32_t> ranks_;       // "左\0右" → 合并次序
    std::unordered_map<std::string, std::int32_t> special_;     // 字面量 → id（type 3/4）
    std::vector<std::string> special_always_;                   // type 4：永远匹配
    std::vector<std::string> special_lits_;                     // type 3+4：parse_special 时匹配
    mutable std::unordered_map<std::string, std::vector<std::int32_t>> piece_cache_;
};

}  // namespace qinfer::text