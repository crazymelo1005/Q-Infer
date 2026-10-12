// chat 模板渲染器：吃 GGUF 的 tokenizer.chat_template 原文，渲染消息列表。
// 只支持主案例模板实际用到的结构：字符串字面量、变量替换、+ 拼接、下标访问（m['role']）、
// if / elif / else / set / for；其余结构（macro、call、过滤器等）明确报错——静默错译等于在算另一个
// 提示词。`{%-` / `-%}` 连字符修饰的空白削减按 Jinja 语义：语句标签贴边时吃掉相邻的一行空白。
#pragma once

#include <string>
#include <vector>

#include "artifact/gguf_table.hpp"

namespace qinfer::serve {

struct Message {
    std::string role;
    std::string content;
};

// 从 GGUF 元数据读 tokenizer.chat_template；缺失或非字符串 → 返回假并写 err（拒绝裸拼接）。
bool chat_template_from_gguf(const artifact::GgufFile& g, std::string& out, std::string& err);

// 渲染。tpl 为模板原文；add_generation_prompt 决定是否追加生成提示段。
// 任何不支持的结构、未闭合块、未知变量 → 返回假并写 err（err 指明模板里的位置）。
bool render_chat(const std::string& tpl, const std::vector<Message>& msgs, bool add_generation_prompt,
                 std::string& out, std::string& err);

}  // namespace qinfer::serve