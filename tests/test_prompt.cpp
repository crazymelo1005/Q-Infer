#include "check.hpp"
#include "serve/prompt.hpp"
#include <cstdio>
#include <vector>

using namespace qinfer::serve;

namespace {

// Qwen 系官方模板（tokenizer_config.json 的标准形态，主案例 GGUF 的 chat_template 应在 M1.5 逐字符核对）。
const char* kQwenTemplate =
    "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}"
    "{%- for message in messages -%}{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + "
    "'<|im_end|>' + '\\n'}}{%- endfor -%}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}";

std::string render_with(const std::vector<Message>& msgs, bool add_gen) {
    std::string out, err;
    CHECK(render_chat(kQwenTemplate, msgs, add_gen, out, err));
    return out;
}

}  // namespace

int main() {
    // 一轮 system + user，加生成提示。
    {
        const std::string got =
            render_with({{"system", "You are helpful."}, {"user", "hi"}}, /*add_gen=*/true);
        const std::string want = "<|im_start|>system\nYou are helpful.<|im_end|>\n"
                                 "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n";
        CHECK(got == want);
        std::printf("两段消息 + gen 提示：逐字符一致\n");
    }
    // 多轮 assistant 消息原样保留。
    {
        const std::vector<Message> msgs = {{"user", "1"}, {"assistant", "2"}, {"user", "3"}};
        const std::string got = render_with(msgs, false);
        CHECK(got == "<|im_start|>user\n1<|im_end|>\n<|im_start|>assistant\n2<|im_end|>\n"
                     "<|im_start|>user\n3<|im_end|>\n");
        std::printf("三轮：多轮拼接正确\n");
    }
    // 空消息列表 + 无生成提示 → 空串。
    {
        CHECK(render_with({}, false).empty());
        std::printf("空列表：渲染为空\n");
    }
    // 内容里的换行、单引号、花括号原样保留（字面量不做二次解释）。
    {
        const std::string got = render_with({{"user", "say 'hi'\n{ not a tag }"}}, false);
        CHECK(got.find("say 'hi'\n{ not a tag }") != std::string::npos);
        std::printf("转义内容：原样保留\n");
    }
    // 不认识的结构 → 明确报错，不静默错译。
    {
        std::string out, err;
        CHECK(!render_chat("{% macro m() %}x{% endmacro %}", {{"user", "a"}}, false, out, err));
        CHECK(!err.empty());
        std::printf("宏结构：报错（%s）\n", err.c_str());
    }
    // 未闭合的 for → 报错。
    {
        std::string out, err;
        CHECK(!render_chat("{% for m in messages %}{{'x'}}", {{"user", "a"}}, false, out, err));
        CHECK(!err.empty());
        std::printf("未闭合 for：报错（%s）\n", err.c_str());
    }
    // role 比较分支（较新的 Qwen 模板形态）：== / elif / else 逐段拼接。
    {
        const char* role_tpl = "{%- for m in messages -%}"
            "{%- if m['role'] == 'system' -%}[S]{{ m['content'] }}"
            "{%- elif m['role'] == 'user' -%}[U]{{ m['content'] }}"
            "{%- else -%}[?]{{ m['content'] }}"
            "{%- endif -%}{%- endfor -%}";
        std::string out, err;
        CHECK(render_chat(role_tpl, {{"system", "s"}, {"user", "u"}, {"assistant", "a"}}, false, out, err));
        CHECK(out == "[S]s[U]u[?]a");
        std::printf("role 比较分支：==/elif/else 逐段正确\n");
    }
    // 未知变量 → 报错。
    {
        std::string out, err;
        CHECK(!render_chat("{{ mystery_var }}", {{"user", "a"}}, false, out, err));
        CHECK(!err.empty());
        std::printf("未知变量：报错（%s）\n", err.c_str());
    }
    std::puts("prompt: 通过");
    return 0;
}