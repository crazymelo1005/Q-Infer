// chat 模板渲染器：吃 GGUF 的 tokenizer.chat_template 原文，渲染消息列表。
// 按 Jinja2 语义执行真实 text-only 模板（主案例是 Qwen3.8-Flash 的 170 行模板）：
//   - 语句：if/elif/else/endif、for/endfor（附 loop.index0/first/last/previtem/nextitem）、
//     set、macro/endmacro 及宏调用（参数默认值、位置实参）；
//   - 表达式：字面量、变量、属性取用（message.reasoning_content → undefined 级联）、
//     下标与负步长整表反转（messages[::-1]）、+ - ~、比较与 in/not in、and/or、三元、
//     is 测试（defined/undefined/string/none/true/false/iterable/mapping/number/sequence）、
//     过滤器（trim/default/length/safe/string）、namespace(...)、raise_exception(...)；
//   - 空白：`{%-`/`{{-` 吃掉前一文本段尾部空白，`-%}`/`-}}` 吃掉后一段前导空白，
//     与 Jinja 默认 keep_trailing_newline=False（模板末尾一个换行不输出）一致。
// 未定义语义：serve 侧只注入 role/content，其余变量（tools/vision/reasoning 等）在渲染时
// 均为 undefined——if 条件为假、length 为 0、属性访问继续级联为 undefined。
// 不认识的过滤器、未实装的运行时语义（tojson/items）、raise_exception、未知变量落入输出
// 位置 → 明确报错（err 带模板位置）——静默错译等于在算另一个提示词。
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