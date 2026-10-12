#include "serve/prompt.hpp"

#include <cctype>
#include <cstdint>

namespace qinfer::serve {

bool chat_template_from_gguf(const artifact::GgufFile& g, std::string& out, std::string& err) {
    const std::string* s = g.meta_str("tokenizer.chat_template");
    if (!s) {
        err = "GGUF 元数据没有 tokenizer.chat_template，无法组提示词（拒绝裸拼接）";
        return false;
    }
    out = *s;
    return true;
}

namespace {

enum class Kind { kText, kExpr, kIf, kElif, kElse, kEndif, kFor, kEndfor, kSet };

struct Op {
    Kind kind = Kind::kText;
    std::string a;              // kText 输出文本；kExpr 表达式源；kIf/kElif 条件源；kSet 的变量名
    std::string b;              // kSet 的表达式源
    std::size_t jump = 0;       // 条件假 / 已执行分支的跳过目标（Op 下标）
    std::size_t end = 0;        // kFor 的 endfor 下标
};

struct Env {
    const std::vector<Message>* msgs = nullptr;
    bool add_gen = false;
    std::string bos, eos;
    const Message* loop_msg = nullptr;      // for 遍历中的当前消息
    bool taken = false;                     // 当前是否已取某分支：从已取主体顺延到分支标记 → 跳过其余
    std::vector<bool> chain;                // if/elif/else 链的外层 taken 栈：kIf 压入、kEndif 弹出
};

std::string trim(const std::string& s) {
    std::size_t a = 0, b = s.size();
    while (a < b && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
    while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
    return s.substr(a, b - a);
}

// 按顶层 '+' 切分（字符串字面量内的 '+' 不动）。
std::vector<std::string> split_top(const std::string& s) {
    std::vector<std::string> parts;
    std::string cur;
    char q = 0;
    for (char c : s) {
        if (q) {
            cur += c;
            if (c == q) q = 0;
        } else if (c == '\'' || c == '"') {
            q = c;
            cur += c;
        } else if (c == '+') {
            parts.push_back(trim(cur));
            cur.clear();
        } else {
            cur += c;
        }
    }
    parts.push_back(trim(cur));
    return parts;
}

// 字符串字面量解码（\n \t \r \" \' \\）。
bool decode_literal(const std::string& s, std::string& out, std::string& err) {
    if (s.size() < 2 || (s.front() != '\'' && s.front() != '"') ||
        s.back() != s.front()) {
        err = "表达式不是字符串字面量：" + s;
        return false;
    }
    const char q = s.front();
    out.clear();
    for (std::size_t i = 1; i + 1 < s.size(); ++i) {
        if (s[i] == '\\' && i + 2 < s.size()) {
            const char c = s[++i];
            switch (c) {
                case 'n': out += '\n'; break;
                case 't': out += '\t'; break;
                case 'r': out += '\r'; break;
                default:
                    if (c == q || c == '\\') out += c;
                    else { err = "不认识的转义 \\" + std::string(1, c); return false; }
            }
        } else {
            out += s[i];
        }
    }
    return true;
}

// 一个术语求值：纯字面量；message/m 的下标；bos/eos/add_generation_prompt 变量。
bool eval_term(const std::string& term, const Env& env, std::string& out, std::string& err) {
    if (term.empty()) { err = "空表达式项"; return false; }
    if (term.front() == '\'' || term.front() == '"') return decode_literal(term, out, err);
    const std::size_t lb = term.find('[');
    if (lb != std::string::npos) {
        const std::string base = trim(term.substr(0, lb));
        if (base != "message" && base != "m") {
            err = "只支持 message['role'] 与 message['content']：'" + term + "'";
            return false;
        }
        if (!env.loop_msg) { err = "message 下标出现在消息循环之外"; return false; }
        if (term.back() != ']') { err = "下标未闭合：" + term; return false; }
        std::string key;
        if (!decode_literal(trim(term.substr(lb + 1, term.size() - lb - 2)), key, err)) return false;
        if (key == "role") out = env.loop_msg->role;
        else if (key == "content") out = env.loop_msg->content;
        else { err = "未知消息键：" + key; return false; }
        return true;
    }
    if (term == "bos_token") { out = env.bos; return true; }
    if (term == "eos_token") { out = env.eos; return true; }
    if (term == "add_generation_prompt") { out = env.add_gen ? "true" : "false"; return true; }
    if (term == "true") { out = "true"; return true; }
    if (term == "false") { out = "false"; return true; }
    err = "未知变量：" + term;
    return false;
}

// 表达式求值（+ 拼接）。
bool eval_expr(const std::string& src, const Env& env, std::string& out, std::string& err) {
    out.clear();
    for (const std::string& part : split_top(src)) {
        std::string piece;
        if (!eval_term(part, env, piece, err)) return false;
        out += piece;
    }
    return true;
}

// 按空白切分（字符串字面量内的空白不动）。条件表达式按词解析；比较运算符 == / != 需要词级切分。
std::vector<std::string> split_words(const std::string& s) {
    std::vector<std::string> out;
    std::string cur;
    char q = 0;
    for (char c : s) {
        if (q) {
            cur += c;
            if (c == q) q = 0;
        } else if (c == '\'' || c == '"') {
            q = c;
            cur += c;
        } else if (std::isspace(static_cast<unsigned char>(c))) {
            if (!cur.empty()) {
                out.push_back(cur);
                cur.clear();
            }
        } else {
            cur += c;
        }
    }
    if (!cur.empty()) out.push_back(cur);
    return out;
}

// 条件求值：not X / X is defined / X is not defined / 字符串相等与不等 / 变量真值 / true / false。
bool eval_cond(const std::string& src, const Env& env, bool& out, std::string& err) {
    const std::vector<std::string> toks = split_words(trim(src));
    if (toks.size() == 1) {
        if (toks[0] == "true") { out = true; return true; }
        if (toks[0] == "false") { out = false; return true; }
        if (toks[0] == "add_generation_prompt") { out = env.add_gen; return true; }
        out = false;   // 未知变量的真值按假处理（模板意图如此，不报错）
        return true;
    }
    if (toks.size() == 2 && toks[0] == "not") {
        bool v;
        if (!eval_cond(toks[1], env, v, err)) return false;
        out = !v;
        return true;
    }
    if (toks.size() == 3 && toks[1] == "is" && toks[2] == "defined") {
        out = (toks[0] == "add_generation_prompt" || toks[0] == "bos_token" ||
               toks[0] == "eos_token" || toks[0] == "messages");
        return true;
    }
    if (toks.size() == 4 && toks[1] == "is" && toks[2] == "not" && toks[3] == "defined") {
        bool d;
        if (!eval_cond(toks[0] + " is defined", env, d, err)) return false;
        out = !d;
        return true;
    }
    // `not X is defined`：Jinja 里 `is defined` 先结合，取反的是「是否已定义」（Qwen 主模板的头一支）。
    if (toks.size() == 4 && toks[0] == "not" && toks[2] == "is" && toks[3] == "defined") {
        bool d;
        if (!eval_cond(toks[1] + " is defined", env, d, err)) return false;
        out = !d;
        return true;
    }
    // 双操作数字符串比较：`m['role'] == 'system'` 这类（两边都按字符串求值）。不等号同理取反。
    if (toks.size() == 3 && (toks[1] == "==" || toks[1] == "!=")) {
        std::string a, b;
        if (!eval_term(toks[0], env, a, err) || !eval_term(toks[2], env, b, err)) return false;
        out = (a == b) == (toks[1] == "==");
        return true;
    }
    err = "不认识的 if 条件：" + src;
    return false;
}

// 词法与编译：扫描出 Op 序列并配对 if/for 块。连字符语义：`{%-` 吃掉随后文本的头部空白，
// `-%}` 吃掉前一文本的尾部空白。
bool compile(const std::string& tpl, std::vector<Op>& ops, std::string& err) {
    ops.clear();
    std::string text;
    bool prev_rtrim = false;      // 上一个标签以 -%} 结尾 → 本段文本尾部空白要吃掉
    bool ltrim_next = false;      // 上一个标签以 {%- 开头 → 本段文本头部空白要吃掉
    std::size_t i = 0;
    const auto flush = [&]() {
        if (prev_rtrim) {
            std::size_t k = text.size();
            while (k > 0 && std::isspace(static_cast<unsigned char>(text[k - 1]))) --k;
            text.resize(k);
        }
        if (ltrim_next) {
            std::size_t k = 0;
            while (k < text.size() && std::isspace(static_cast<unsigned char>(text[k]))) ++k;
            text.erase(0, k);
            ltrim_next = false;
        }
        if (!text.empty()) {
            Op op;
            op.kind = Kind::kText;
            op.a = text;
            ops.push_back(op);
        }
        text.clear();
    };
    const auto fail = [&](const std::string& why, std::size_t at) {
        err = "模板第 " + std::to_string(at) + " 字符处：" + why;
        return false;
    };
    while (i < tpl.size()) {
        const std::string open2 = tpl.compare(i, 2, "{{") == 0 ? "{{" :
                                  (tpl.compare(i, 2, "{%") == 0 ? "{%" : std::string());
        if (open2.empty()) {
            text += tpl[i++];
            continue;
        }
        flush();
        const bool trim_after_open = i + 2 < tpl.size() && tpl[i + 2] == '-';
        const std::size_t open_adj = trim_after_open ? 3 : 2;
        const std::string close2 = open2 == "{{" ? "}}" : "%}";
        const std::size_t close_at = tpl.find(close2, i + open_adj);
        if (close_at == std::string::npos) return fail("标签未闭合", i);
        const bool trim_before_close = close_at > i + open_adj && tpl[close_at - 1] == '-';
        const std::size_t begin = i + open_adj;
        const std::size_t len = close_at - begin - (trim_before_close ? 1 : 0);
        const std::string body = trim(tpl.substr(begin, len));
        if (open2 == "{{") {
            Op op;
            op.kind = Kind::kExpr;
            op.a = body;
            ops.push_back(op);
        } else {
            const std::size_t sp = body.find(' ');
            const std::string k = sp == std::string::npos ? body : body.substr(0, sp);
            std::size_t rest = sp == std::string::npos ? std::string::npos : sp + 1;
            const std::string tail = rest == std::string::npos ? "" : trim(body.substr(rest));
            Op op;
            if (k == "if") {
                op.kind = Kind::kIf;
                op.a = tail;
            } else if (k == "elif") {
                op.kind = Kind::kElif;
                op.a = tail;
            } else if (k == "else") {
                op.kind = Kind::kElse;
            } else if (k == "endif") {
                op.kind = Kind::kEndif;
            } else if (k == "endfor") {
                op.kind = Kind::kEndfor;
            } else if (k == "for") {
                const std::vector<std::string> toks = split_words(tail);
                if (toks.size() != 3 || toks[1] != "in" || toks[2] != "messages") {
                    return fail("只支持 for <var> in messages", i);
                }
                op.kind = Kind::kFor;
                op.a = toks[0];
            } else if (k == "set") {
                const std::size_t eq = tail.find('=');
                if (eq == std::string::npos) return fail("set 缺 =", i);
                op.kind = Kind::kSet;
                op.a = trim(tail.substr(0, eq));
                op.b = trim(tail.substr(eq + 1));
            } else {
                return fail("不认识的语句：" + body, i);
            }
            ops.push_back(op);
        }
        prev_rtrim = trim_before_close;
        ltrim_next = trim_after_open;
        i = close_at + 2;
    }
    flush();
    if (!text.empty()) {   // 模板以纯文本结尾，这段在上一个 flush 后不可能再攒
        Op op;
        op.kind = Kind::kText;
        op.a = text;
        ops.push_back(op);
    }
    (void) fail;   // fail 在下面的配对错误处复用

    // 配对：if/elif/else/endif 与 for/endfor。
    std::vector<std::size_t> ifs, fors;
    for (std::size_t idx = 0; idx < ops.size(); ++idx) {
        switch (ops[idx].kind) {
            case Kind::kIf: ifs.push_back(idx); break;
            case Kind::kFor: fors.push_back(idx); break;
            case Kind::kEndfor: {
                if (fors.empty()) { err = "多余的 endfor"; return false; }
                const std::size_t f = fors.back();
                fors.pop_back();
                ops[f].end = idx;
                break;
            }
            case Kind::kEndif: {
                if (ifs.empty()) { err = "多余的 endif"; return false; }
                const std::size_t start = ifs.back();
                ifs.pop_back();
                // 只配对本层的分支标记：嵌套的 if/endif 会改变深度，不得改它们的跳转。
                std::vector<std::size_t> chain;
                int depth = 0;
                for (std::size_t c = start; c < idx; ++c) {
                    switch (ops[c].kind) {
                        case Kind::kIf:
                            if (c == start) chain.push_back(c);   // 本层链头（区域首个 op 必是它）
                            else ++depth;                          // 链内的嵌套链：只计深度
                            break;
                        case Kind::kElif:
                        case Kind::kElse:
                            if (depth == 0) chain.push_back(c);   // 本层的分支标记
                            break;
                        case Kind::kEndif:
                            if (depth > 0) --depth;                // 嵌套链闭合
                            break;
                        default: break;
                    }
                }
                for (std::size_t k = 0; k + 1 < chain.size(); ++k) {
                    ops[chain[k]].jump = chain[k + 1];   // 条件假 → 下一个分支标记
                }
                ops[chain.back()].jump = idx;            // 链尾假（或已取分支顺延）→ endif
                break;
            }
            default: break;
        }
    }
    if (!ifs.empty()) { err = "if 未闭合"; return false; }
    if (!fors.empty()) { err = "for 未闭合"; return false; }
    return true;
}

// 解释执行 [from, to) 区间。返回 false → err 非空。
bool run(const std::vector<Op>& ops, std::size_t from, std::size_t to, Env& env, std::string& out,
         std::string& err) {
    std::size_t i = from;
    while (i < to) {
        const Op& op = ops[i];
        switch (op.kind) {
            case Kind::kText: out += op.a; ++i; break;
            case Kind::kExpr: {
                std::string v;
                if (!eval_expr(op.a, env, v, err)) {
                    err = "求值失败：" + op.a + "（" + err + "）";
                    return false;
                }
                out += v;
                ++i;
                break;
            }
            case Kind::kIf: {
                bool c;
                if (!eval_cond(op.a, env, c, err)) {
                    err = "条件求值失败：" + op.a + "（" + err + "）";
                    return false;
                }
                env.chain.push_back(env.taken);   // 链开始：保存外层「已取」状态，kEndif 处还原
                if (c) { ++i; env.taken = true; }
                else { env.taken = false; i = op.jump; }
                break;
            }
            case Kind::kElif: {
                if (env.taken) { i = op.jump; break; }   // 顺延自已取分支 → 跳过剩余分支
                bool c;
                if (!eval_cond(op.a, env, c, err)) {
                    err = "条件求值失败：" + op.a + "（" + err + "）";
                    return false;
                }
                if (c) { ++i; env.taken = true; }
                else { i = op.jump; }
                break;
            }
            case Kind::kElse:
                if (env.taken) { i = op.jump; break; }   // 顺延自已取分支 → 跳过 else 主体
                env.taken = true;                        // 前面分支全未取 → 执行 else 主体
                ++i;
                break;
            case Kind::kEndif:
                ++i;
                env.taken = env.chain.back();            // 链结束：还原外层「已取」状态
                env.chain.pop_back();
                break;
            case Kind::kSet: {
                // 只支持 set <已知变量> = <字面量或 true/false>；Qwen 模板只用 set add_generation_prompt。
                std::string v;
                if (!eval_expr(op.b, env, v, err)) { err = "set 求值失败：" + op.b; return false; }
                if (op.a == "add_generation_prompt") env.add_gen = (v == "true");
                else if (op.a == "bos_token") env.bos = v;
                else if (op.a == "eos_token") env.eos = v;
                else { err = "set 不认识的变量：" + op.a; return false; }
                ++i;
                break;
            }
            case Kind::kFor: {
                if (!env.msgs) { err = "messages 为空"; return false; }
                for (const Message& m : *env.msgs) {
                    env.loop_msg = &m;
                    if (!run(ops, i + 1, ops[i].end, env, out, err)) {
                        err = "循环体内：" + err;
                        return false;
                    }
                }
                env.loop_msg = nullptr;
                i = ops[i].end + 1;
                break;
            }
            case Kind::kEndfor: i = to; break;   // 到不了（kFor 内循环处理），防御
        }
    }
    return true;
}

}  // namespace

bool render_chat(const std::string& tpl, const std::vector<Message>& msgs, bool add_generation_prompt,
                 std::string& out, std::string& err) {
    std::vector<Op> ops;
    if (!compile(tpl, ops, err)) return false;
    Env env;
    env.msgs = &msgs;
    env.add_gen = add_generation_prompt;
    out.clear();
    return run(ops, 0, ops.size(), env, out, err);
}

}  // namespace qinfer::serve