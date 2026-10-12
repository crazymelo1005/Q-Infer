// -*- coding: utf-8 -*-
#include "check.hpp"
#include "serve/prompt.hpp"
#include <cstdio>
#include <string>
#include <vector>

using namespace qinfer::serve;

namespace {

// Qwen 系回归模板（tokenizer_config.json 的标准形态；主案例以 kMainCaseTemplate 为准）。
const char* kQwenTemplate =
    "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}"
    "{%- for message in messages -%}{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + "
    "'<|im_end|>' + '\\n'}}{%- endfor -%}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}";

// 主案例 GGUF 的 tokenizer.chat_template 原文（170 行 8953 字节，逐字嵌入；
// 字节以 .superpowers/sdd/2026-10-12-serve-layer/task-18a-template.txt 为准）。
const char* kMainCaseTemplate = R"TPL({%- set image_count = namespace(value=0) %}
{%- set video_count = namespace(value=0) %}
{%- macro render_content(content, do_vision_count, is_system_content=false) %}
    {%- if content is string %}
        {{- content }}
    {%- elif content is iterable and content is not mapping %}
        {%- for item in content %}
            {%- if 'image' in item or 'image_url' in item or item.type == 'image' %}
                {%- if is_system_content %}
                    {{- raise_exception('System message cannot contain images.') }}
                {%- endif %}
                {%- if do_vision_count %}
                    {%- set image_count.value = image_count.value + 1 %}
                {%- endif %}
                {%- if add_vision_id %}
                    {{- 'Picture ' ~ image_count.value ~ ': ' }}
                {%- endif %}
                {{- '<|vision_start|><|image_pad|><|vision_end|>' }}
            {%- elif 'video' in item or item.type == 'video' %}
                {%- if is_system_content %}
                    {{- raise_exception('System message cannot contain videos.') }}
                {%- endif %}
                {%- if do_vision_count %}
                    {%- set video_count.value = video_count.value + 1 %}
                {%- endif %}
                {%- if add_vision_id %}
                    {{- 'Video ' ~ video_count.value ~ ': ' }}
                {%- endif %}
                {{- '<|vision_start|><|video_pad|><|vision_end|>' }}
            {%- elif 'text' in item %}
                {{- item.text }}
            {%- else %}
                {{- raise_exception('Unexpected item type in content.') }}
            {%- endif %}
        {%- endfor %}
    {%- elif content is none or content is undefined %}
        {{- '' }}
    {%- else %}
        {{- raise_exception('Unexpected content type.') }}
    {%- endif %}
{%- endmacro %}
{%- if not messages %}
    {{- raise_exception('No messages provided.') }}
{%- endif %}
{%- set reasoning_instructions = '' %}
{%- if enable_thinking is undefined or enable_thinking is true %}
    {%- set resolved_reasoning_effort = reasoning_effort|default('xhigh') %}
    {%- if resolved_reasoning_effort not in ('xhigh', 'medium', 'low') %}
        {{- raise_exception('Unexpected reasoning effort ' ~ reasoning_effort ~ '. Supported types are xhigh (default), medium, and low.') }}
    {%- endif %}
    {%- if resolved_reasoning_effort == 'xhigh' %}
        {%- set reasoning_instructions = 'Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.' %}
    {%- elif resolved_reasoning_effort == 'low' %}
        {%- set reasoning_instructions = 'Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.' %}
    {%- endif %}
{%- endif %}
{%- if tools and tools is iterable and tools is not mapping %}
    {{- '<|im_start|>system\n' }}
    {%- if reasoning_instructions %}
        {{- reasoning_instructions + '\n\n' }}
    {%- endif %}
    {{- "# Tools\n\nYou have access to the following functions:\n\n<tools>" }}
    {%- for tool in tools %}
        {{- "\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\n</tools>" }}
    {{- '\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>' }}
    {%- if messages[0].role == 'system' %}
        {%- set content = render_content(messages[0].content, false, true)|trim %}
        {%- if content %}
            {{- '\n\n' + content }}
        {%- endif %}
    {%- endif %}
    {{- '<|im_end|>\n' }}
{%- else %}
    {%- if messages[0].role == 'system' %}
        {%- set content = render_content(messages[0].content, false, true)|trim %}
        {%- if content %}
            {{- '<|im_start|>system\n' + (reasoning_instructions + '\n\n' if reasoning_instructions else '')  + content + '<|im_end|>\n' }}
        {%- elif reasoning_instructions %}
            {{- '<|im_start|>system\n' + reasoning_instructions + '<|im_end|>\n' }}
        {%- endif %}
    {%- elif reasoning_instructions %}
        {{- '<|im_start|>system\n' + reasoning_instructions + '<|im_end|>\n' }}
    {%- endif %}
{%- endif %}
{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}
{%- for message in messages[::-1] %}
    {%- set index = (messages|length - 1) - loop.index0 %}
    {%- if ns.multi_step_tool and message.role == "user" %}
        {%- set content = render_content(message.content, false)|trim %}
        {%- if not(content.startswith('<tool_response>') and content.endswith('</tool_response>')) %}
            {%- set ns.multi_step_tool = false %}
            {%- set ns.last_query_index = index %}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if ns.multi_step_tool %}
    {{- raise_exception('No user query found in messages.') }}
{%- endif %}
{%- for message in messages %}
    {%- set content = render_content(message.content, true)|trim %}
    {%- if message.role == "system" %}
        {%- if not loop.first %}
            {{- raise_exception('System message must be at the beginning.') }}
        {%- endif %}
    {%- elif message.role == "user" %}
        {{- '<|im_start|>' + message.role + '\n' + content + '<|im_end|>' + '\n' }}
    {%- elif message.role == "assistant" %}
        {%- set reasoning_content = '' %}
        {%- if message.reasoning_content is string %}
            {%- set reasoning_content = message.reasoning_content %}
        {%- endif %}
        {%- set reasoning_content = reasoning_content|trim %}
        {%- if preserve_thinking is undefined or preserve_thinking is true or loop.index0 > ns.last_query_index %}
            {{- '<|im_start|>' + message.role + '\n<think>\n' + reasoning_content + '\n</think>\n\n' + content }}
        {%- else %}
            {{- '<|im_start|>' + message.role + '\n' + content }}
        {%- endif %}
        {%- if message.tool_calls and message.tool_calls is iterable and message.tool_calls is not mapping %}
            {%- for tool_call in message.tool_calls %}
                {%- if tool_call.function is defined %}
                    {%- set tool_call = tool_call.function %}
                {%- endif %}
                {%- if loop.first %}
                    {%- if content|trim %}
                        {{- '\n\n<tool_call>\n<function=' + tool_call.name + '>\n' }}
                    {%- else %}
                        {{- '<tool_call>\n<function=' + tool_call.name + '>\n' }}
                    {%- endif %}
                {%- else %}
                    {{- '\n<tool_call>\n<function=' + tool_call.name + '>\n' }}
                {%- endif %}
                {%- if tool_call.arguments is defined and tool_call.arguments != '' %}
                    {%- for args_name, args_value in tool_call.arguments|items %}
                        {{- '<parameter=' + args_name + '>\n' }}
                        {%- set args_value = args_value | string if args_value is string else args_value | tojson | safe %}
                        {{- args_value }}
                        {{- '\n</parameter>\n' }}
                    {%- endfor %}
                {%- endif %}
                {{- '</function>\n</tool_call>' }}
            {%- endfor %}
        {%- endif %}
        {{- '<|im_end|>\n' }}
    {%- elif message.role == "tool" %}
        {%- if loop.previtem and loop.previtem.role != "tool" %}
            {{- '<|im_start|>user' }}
        {%- endif %}
        {{- '\n<tool_response>\n' }}
        {{- content }}
        {{- '\n</tool_response>' }}
        {%- if not loop.last and loop.nextitem.role != "tool" %}
            {{- '<|im_end|>\n' }}
        {%- elif loop.last %}
            {{- '<|im_end|>\n' }}
        {%- endif %}
    {%- else %}
        {{- raise_exception('Unexpected message role.') }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
    {%- if enable_thinking is defined and enable_thinking is false %}
        {{- '<think>\n\n</think>\n\n' }}
    {%- else %}
        {{- '<think>\n' }}
    {%- endif %}
{%- endif %}
)TPL";

std::string render_with(const std::vector<Message>& msgs, bool add_gen) {
    std::string out, err;
    CHECK(render_chat(kQwenTemplate, msgs, add_gen, out, err));
    return out;
}

void expect_render(const char* tpl, const std::vector<Message>& msgs, bool add_gen, const std::string& want) {
    std::string out, err;
    CHECK(render_chat(tpl, msgs, add_gen, out, err));
    CHECK(out == want);
    (void) err;
}

}  // namespace

int main() {
    // 回归：一轮 system + user，加生成提示（Qwen 简模板）。
    {
        const std::string got =
            render_with({{"system", "You are helpful."}, {"user", "hi"}}, /*add_gen=*/true);
        const std::string want = "<|im_start|>system\nYou are helpful.<|im_end|>\n"
                                 "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n";
        CHECK(got == want);
        std::printf("回归两段消息 + gen 提示：逐字符一致\n");
    }
    // 回归：多轮 assistant 消息原样保留。
    {
        const std::vector<Message> msgs = {{"user", "1"}, {"assistant", "2"}, {"user", "3"}};
        const std::string got = render_with(msgs, false);
        CHECK(got == "<|im_start|>user\n1<|im_end|>\n<|im_start|>assistant\n2<|im_end|>\n"
                     "<|im_start|>user\n3<|im_end|>\n");
        std::printf("回归三轮：多轮拼接正确\n");
    }
    // 回归：空消息列表 + 无生成提示 → 空串。
    {
        CHECK(render_with({}, false).empty());
        std::printf("回归空列表：渲染为空\n");
    }
    // 回归：内容里的换行、单引号、花括号原样保留（字面量不做二次解释）。
    {
        const std::string got = render_with({{"user", "say 'hi'\n{ not a tag }"}}, false);
        CHECK(got.find("say 'hi'\n{ not a tag }") != std::string::npos);
        std::printf("回归转义内容：原样保留\n");
    }
    // 未闭合的 for → 报错。
    {
        std::string out, err;
        CHECK(!render_chat("{% for m in messages %}{{'x'}}", {{"user", "a"}}, false, out, err));
        CHECK(!err.empty());
        std::printf("未闭合 for：报错（%s）\n", err.c_str());
    }
    // 回归：role 比较分支 == / elif / else 逐段拼接。
    {
        const char* role_tpl = "{%- for m in messages -%}"
            "{%- if m['role'] == 'system' -%}[S]{{ m['content'] }}"
            "{%- elif m['role'] == 'user' -%}[U]{{ m['content'] }}"
            "{%- else -%}[?]{{ m['content'] }}"
            "{%- endif -%}{%- endfor -%}";
        std::string out, err;
        CHECK(render_chat(role_tpl, {{"system", "s"}, {"user", "u"}, {"assistant", "a"}}, false, out, err));
        CHECK(out == "[S]s[U]u[?]a");
        std::printf("回归 role 比较分支：==/elif/else 逐段正确\n");
    }
    // 未知变量 → 报错（主案例语义：{{ mystery_var }} 直接输出仍报错）。
    {
        std::string out, err;
        CHECK(!render_chat("{{ mystery_var }}", {{"user", "a"}}, false, out, err));
        CHECK(!err.empty());
        std::printf("未知变量直接输出：报错（%s）\n", err.c_str());
    }

    // === 向量 1：用户单轮（含 reasoning_instructions 系统块与生成提示，add_gen=true）===
    {
        const std::string want = "<|im_start|>system\nReasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.<|im_end|>\n<|im_start|>user\n数数：1、2、<|im_end|>\n<|im_start|>assistant\n<think>\n";
        expect_render(kMainCaseTemplate, {{"user", "数数：1、2、"}}, /*add_gen=*/true, want);
        std::printf("向量 1：用户单轮主案例模板逐字符一致（%zu 字节）\n", want.size());
    }
    // === 向量 2：系统 + 用户（系统内容过 render_content|trim，reasoning_instructions 块并入）===
    {
        const std::string want = "<|im_start|>system\nReasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.\n\nYou are helpful.<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n";
        expect_render(kMainCaseTemplate, {{"system", "You are helpful."}, {"user", "hi"}}, /*add_gen=*/true, want);
        std::printf("向量 2：系统+用户主案例模板逐字符一致（%zu 字节）\n", want.size());
    }
    // === 向量 3：用户/助手/用户（assistant 走 thinking 分支；reversed 预扫描）===
    {
        const std::string want = "<|im_start|>system\nReasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.<|im_end|>\n<|im_start|>user\n1<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n2<|im_end|>\n<|im_start|>user\n3<|im_end|>\n<|im_start|>assistant\n<think>\n";
        const std::vector<Message> msgs = {{"user", "1"}, {"assistant", "2"}, {"user", "3"}};
        expect_render(kMainCaseTemplate, msgs, /*add_gen=*/true, want);
        std::printf("向量 3：三轮主案例模板逐字符一致（%zu 字节）\n", want.size());
    }
    // === 向量 4：真模板全文嵌入的字节数核对 + 向量 2/3 形态重跑 ===
    {
        const std::string embedded(kMainCaseTemplate);
        CHECK(embedded.size() == 8953);
        CHECK(embedded.rfind("{%- set image_count = namespace(value=0) %}", 0) == 0);
        CHECK(embedded.find("{%- endif %}\n") != std::string::npos);
        const std::string want2 = "<|im_start|>system\nReasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.\n\nYou are helpful.<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n";
        const std::string want3 = "<|im_start|>system\nReasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.<|im_end|>\n<|im_start|>user\n1<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n2<|im_end|>\n<|im_start|>user\n3<|im_end|>\n<|im_start|>assistant\n<think>\n";
        expect_render(kMainCaseTemplate, {{"system", "You are helpful."}, {"user", "hi"}}, /*add_gen=*/true, want2);
        expect_render(kMainCaseTemplate, {{"user", "1"}, {"assistant", "2"}, {"user", "3"}}, /*add_gen=*/true, want3);
        std::printf("向量 4：真模板全文嵌入，字节数与同款消息对拍一致\n");
    }
    // === 向量 5：宏定义不调用 → 渲染成功无副作用 ===
    {
        std::string out, err;
        CHECK(render_chat("{% macro m() %}x{% endmacro %}", {{"user", "a"}}, false, out, err));
        CHECK(out.empty());
        std::printf("向量 5：宏定义不调用：渲染成功无副作用\n");
    }
    // 宏默认参数与位置实参调用（含参数替换与宏体输出）。
    {
        std::string out, err;
        CHECK(render_chat("{% macro m(a='d') %}{{ a }}{% endmacro %}{{ m('z') }}",
                          {{"user", "a"}}, false, out, err));
        CHECK(out == "z");
        std::printf("宏默认参数 + 位置实参调用：正确\n");
    }
    // === 向量 6：不认识的过滤器 → 执行时报错（不静默错译）===
    {
        std::string out, err;
        CHECK(!render_chat("{{ 'a'|bogus_filter }}", {{"user", "a"}}, false, out, err));
        CHECK(!err.empty());
        std::printf("向量 6：未知过滤器执行报错（%s）\n", err.c_str());
    }
    // set 对已存在属性写回：命中键原位更新、不追加重复键 → |length 恒 1（修复前 2|2）。
    {
        std::string out, err;
        CHECK(render_chat("{% set ns = namespace(x=1) %}{% set ns.x = 2 %}{{ ns | length }}|{{ ns.x }}",
                          {{"user", "a"}}, false, out, err));
        CHECK(out == "1|2");
        std::printf("set 已存在属性：写回不追加重复键（|length 恒 1）\n");
    }
    std::puts("prompt: 通过");
    return 0;
}

