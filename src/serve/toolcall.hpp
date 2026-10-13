// 工具调用编解码（M1 工具回合）：输入侧（tools 归一、历史 OpenAi tool_calls 解析）
// 与输出侧（生成文本里的 <tool_call>…</tool_call> 块切分与解析、OpenAI/Anthropic 响应形状）。
// 语义以 Strata 的 serve/frontend.py（_tool_list / tool_arguments / parse_tool_call /
// param_end / call_end / OutputParser）为对拍：
//   - 参数值按工具 schema 判定——声明 string 原文取，其余 JSON 自动、失败回原文；
//   - 值里的 </parameter> / </tool_call> 字面按 param_end 的后随判定留在值内（#210）；
//   - 流式切分把半个标记留在缓冲等齐——kText 事件永远不含半截 <tool_；
//   - 未闭合调用在 finish() 时按原样文本释放（#211）。
#pragma once

#include <optional>
#include <string>
#include <vector>

#include "serve/json.hpp"

namespace qinfer::serve {

// 解析出来的一个工具调用：名字 + 参数字典（值类型见头注释，与 Strata parse_tool_call 一致）。
struct ToolCall {
    std::string name;
    Json arguments;   // 对象：参数名 → 值
};

// 生成文本拆出来的事件：一段文本，或一个完整调用（text 为空）。
struct GenEvent {
    std::string text;
    std::optional<ToolCall> call;
};

// 把请求的 tools 参数归一成内部工具形状 {{"name", "description", "parameters"}}：
//   OpenAI（anthropic=false）：解包 {"type":"function","function":{...}} 取 function 体，
//     裸 {"name":...} 原样；description/parameters 原样保留；
//   Anthropic（anthropic=true）：input_schema 改名 parameters（旧键不残留），
//     description 缺省补空串。
// 非法形状（非数组、空名字、function/schema 非对象）→ 返回假并写 err（上层回 400）。
bool normalize_tools(const Json& value, bool anthropic, std::vector<Json>& out, std::string& err);

// 客户端历史里一次调用的 arguments → 模板渲染的参数对象（对拍 Strata tool_arguments）：
// 空（null / 全空白串）→ {}；JSON 对象（文本或已解析）原样；其他（坏 JSON、数组、数字）→
// {"arguments": 原文}，拒绝干净而不是难为整段对话（Strata #510）。
Json tool_arguments(const Json* raw);

// 历史 assistant 消息的 OpenAI tool_calls 列表 → 内部形状
// {"function": {"name": str, "arguments": 对象}}（模板 assistant 工具分支渲染用）。
// 元素无 function 包装时取元素本身；缺名字或非数组 → 返回假并写 err。
bool parse_openai_tool_calls(const Json& value, std::vector<Json>& out, std::string& err);

// 解析一整块 <tool_call>…</tool_call> → ToolCall。tools 按名字查 schema 决定参数取值。
// 缺 <tool_call> 开头 / 缺 <function= / 空函数名 / <parameter= 后无 >（截断）→ 假 + err。
bool parse_tool_block(const std::string& raw, const std::vector<Json>& tools, ToolCall& out,
                      std::string& err);

// 增量输出切分器（对拍 Strata OutputParser 的非流式工具路径）：
//   - 标记跨增量切碎时，半个 <tool_call> 留在缓冲等齐——kText 事件永远不含半截标记；
//   - 调用前的尾换行与调用后的前换行按 Strata 丢弃（文本/调用交界干净、分片=整段）；
//   - 完整调用块在闭标记齐全的瞬间整块出现（kBlock，含首尾标记原文）；
//   - 未闭合调用在 finish() 时按原样文本释放（#211）。
class ToolCallStreamer {
public:
    enum class Kind { kText, kBlock };
    struct Event {
        Kind kind;
        std::string text;   // kBlock = 完整块原文（含 <tool_call> … </tool_call>）
    };

    void feed(const std::string& piece, std::vector<Event>& out);
    void finish(std::vector<Event>& out);
    bool in_block() const { return in_block_; }

private:
    std::string buf_;
    bool in_block_ = false;
    bool lead_ = false;   // 调用刚收尾：接续文本的前导换行丢弃（对拍 Strata self.lead）
};

// 整段生成文本 → 事件：kBlock 尝试 parse_tool_block，失败回退为文本（模型幻觉出的坏标记
// 不当错误毁掉整次请求）；相邻文本碎片合并成一段（含失败块的原文接续）。
void split_generation(const std::string& text, const std::vector<Json>& tools,
                      std::vector<GenEvent>& out);

// 响应形状：OpenAI 的 tool_call 对象与 Anthropic 的 tool_use 块。
Json openai_tool_call(const std::string& id, const std::string& name, const Json& arguments);
Json anthropic_tool_use(const std::string& id, const std::string& name, const Json& arguments);

}  // namespace qinfer::serve