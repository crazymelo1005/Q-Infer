#include "serve/prompt.hpp"

#include <cctype>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

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

// ---- 值模型 ---------------------------------------------------------------
// 覆盖主案例真模板（Qwen3 Flash-Next 系）文本-only 执行谱所需的最小值类型集合。
struct Expr;
struct Jv {
    enum class K { kUndef, kNull, kBool, kNum, kStr, kList, kDict, kMsg, kBound, kMacro };
    K k = K::kUndef;
    bool b = false;                  // kBool
    long long n = 0;                 // kNum（kBool 也借住在这里）
    std::string s;                   // kStr
    std::shared_ptr<std::vector<Jv>> list;                        // kList
    std::shared_ptr<std::vector<std::pair<std::string, Jv>>> dict;  // kDict（namespace）
    const Message* msg = nullptr;    // kMsg（消息代理：role/content 可读，其余属性未定义）

    struct Macro {
        std::string name;
        std::vector<std::string> params;
        std::vector<std::shared_ptr<Expr>> defaults;   // 形参默认值表达式；nullptr=必填
        std::size_t begin = 0, end = 0;                // 宏体 op 区间 [begin+1, end)
    };
    std::shared_ptr<Macro> macro;                      // kMacro
    std::shared_ptr<std::pair<std::string, std::string>> bound;  // kBound（字符串方法绑定：(原文, 方法名)）
};

Jv jv_str(std::string v) { Jv j; j.k = Jv::K::kStr; j.s = std::move(v); return j; }
Jv jv_num(long long v) { Jv j; j.k = Jv::K::kNum; j.n = v; return j; }
Jv jv_bool(bool v) { Jv j; j.k = Jv::K::kBool; j.b = v; j.n = v; return j; }
Jv jv_null() { Jv j; j.k = Jv::K::kNull; return j; }
Jv jv_list() { Jv j; j.k = Jv::K::kList; j.list = std::make_shared<std::vector<Jv>>(); return j; }
Jv jv_dict() {
    Jv j;
    j.k = Jv::K::kDict;
    j.dict = std::make_shared<std::vector<std::pair<std::string, Jv>>>();
    return j;
}
Jv jv_msg(const Message* m) { Jv j; j.k = Jv::K::kMsg; j.msg = m; return j; }
Jv jv_bound(std::string base, std::string method) {
    Jv j;
    j.k = Jv::K::kBound;
    j.bound = std::make_shared<std::pair<std::string, std::string>>(std::move(base), std::move(method));
    return j;
}

// 真值语义与 Jinja 一致：未定义/空/零/空串/空列表/空对象为假。
bool jv_truthy(const Jv& v) {
    switch (v.k) {
        case Jv::K::kUndef:
        case Jv::K::kNull: return false;
        case Jv::K::kBool: return v.b;
        case Jv::K::kNum: return v.n != 0;
        case Jv::K::kStr: return !v.s.empty();
        case Jv::K::kList: return v.list && !v.list->empty();
        case Jv::K::kDict: return v.dict && !v.dict->empty();
        default: return true;   // kMsg / kBound / kMacro
    }
}

bool jv_equal(const Jv& a, const Jv& b) {
    if (a.k != b.k) return false;
    switch (a.k) {
        case Jv::K::kUndef:
        case Jv::K::kNull: return true;
        case Jv::K::kBool: return a.b == b.b;
        case Jv::K::kNum: return a.n == b.n;
        case Jv::K::kStr: return a.s == b.s;
        default: return false;
    }
}

// 字符串化（输出与 ~ 拼接共用；未定义值与不可转类型报错——不静默错译）。
bool jv_to_str(const Jv& v, std::string& s, std::string& err) {
    switch (v.k) {
        case Jv::K::kStr: s = v.s; return true;
        case Jv::K::kNum: s = std::to_string(v.n); return true;
        case Jv::K::kBool: s = v.b ? "True" : "False"; return true;
        case Jv::K::kNull: s = "None"; return true;
        case Jv::K::kUndef: err = "未定义值不能转字符串"; return false;
        default: err = "该值类型不能转字符串"; return false;
    }
}

// ---- 词法 ----------------------------------------------------------------
// 记号：标识符 / 字符串字面量（C 风格转义已解码）/ 整数 / 符号。
struct Tok {
    enum class T { kEnd, kIdent, kStr, kNum, kSym };
    T t = T::kEnd;
    std::string s;   // kIdent 名 / kStr 解码后原文 / kSym 符号
    long long n = 0; // kNum
};

bool tok_is_sym(const Tok& t, const char* s) { return t.t == Tok::T::kSym && t.s == s; }
bool tok_is_ident(const Tok& t, const char* s) { return t.t == Tok::T::kIdent && t.s == s; }

std::vector<Tok> tokenize(const std::string& src, std::string& err) {
    std::vector<Tok> out;
    std::size_t i = 0;
    while (i < src.size()) {
        const char c = src[i];
        if (std::isspace(static_cast<unsigned char>(c))) { ++i; continue; }
        if (c == '\'' || c == '"') {
            const char q = c;
            std::string lit;
            ++i;
            bool closed = false;
            while (i < src.size()) {
                const char d = src[i];
                if (d == '\\' && i + 1 < src.size()) {
                    const char e = src[i + 1];
                    switch (e) {
                        case 'n': lit += '\n'; break;
                        case 't': lit += '\t'; break;
                        case 'r': lit += '\r'; break;
                        default:
                            if (e == q || e == '\\' || e == '\'') lit += e;
                            else { err = std::string("表达式里不认识的转义 \\") + e; return {}; }
                    }
                    i += 2;
                } else if (d == q) {
                    ++i;
                    closed = true;
                    break;
                } else {
                    lit += d;
                    ++i;
                }
            }
            if (!closed) { err = "表达式里字符串字面量未闭合"; return {}; }
            Tok t; t.t = Tok::T::kStr; t.s = std::move(lit);
            out.push_back(std::move(t));
            continue;
        }
        if (std::isdigit(static_cast<unsigned char>(c))) {
            long long v = 0;
            while (i < src.size() && std::isdigit(static_cast<unsigned char>(src[i]))) {
                v = v * 10 + (src[i] - '0');
                ++i;
            }
            Tok t; t.t = Tok::T::kNum; t.n = v;
            out.push_back(std::move(t));
            continue;
        }
        if (std::isalpha(static_cast<unsigned char>(c)) || c == '_') {
            std::string id;
            while (i < src.size() &&
                   (std::isalnum(static_cast<unsigned char>(src[i])) || src[i] == '_')) {
                id += src[i++];
            }
            Tok t; t.t = Tok::T::kIdent; t.s = std::move(id);
            out.push_back(std::move(t));
            continue;
        }
        // 双字符符号优先。
        if (i + 1 < src.size()) {
            const std::string two = src.substr(i, 2);
            if (two == "==" || two == "!=" || two == ">=" || two == "<=") {
                Tok t; t.t = Tok::T::kSym; t.s = two;
                out.push_back(std::move(t));
                i += 2;
                continue;
            }
        }
        const std::string one = std::string(1, c);
        if (one == "(" || one == ")" || one == "[" || one == "]" || one == "." ||
            one == "," || one == "|" || one == ":" || one == "~" || one == "+" ||
            one == "-" || one == ">" || one == "<" || one == "=") {
            Tok t; t.t = Tok::T::kSym; t.s = one;
            out.push_back(std::move(t));
            ++i;
            continue;
        }
        err = std::string("表达式里不认识的符号 '") + c + "'";
        return {};
    }
    Tok end;
    out.push_back(end);
    return out;
}

// ---- 表达式 AST 与解析 ----------------------------------------------------
enum class ET { kStr, kNum, kBool, kNull, kVar, kNeg, kNot, kAdd, kSub, kConcat,
                kEq, kNe, kGt, kLt, kGe, kLe, kIn, kNotIn, kTest, kAnd, kOr, kCond,
                kAttr, kIndex, kSlice, kCall, kFilter, kList };

struct Expr {
    ET t = ET::kVar;
    std::string s;        // kVar 名 / 属性名 / 过滤器名 / 测试名 / kStr 字面量（已解码）
    long long n = 0;      // kNum / kBool
    std::vector<std::shared_ptr<Expr>> kids;   // 运算数槽（kCond: [真, 条件, 假]；kList: 元素）
    std::vector<std::shared_ptr<Expr>> args;   // kCall / kFilter 实参
    std::vector<std::string> kwargs;           // kCall 关键字实参名（与 args 并行，缺位为空串）
};

std::shared_ptr<Expr> mk_expr(ET t) { auto e = std::make_shared<Expr>(); e->t = t; return e; }

// 递归下降解析（优先级自低到高：条件式 < or < and < not < 比较 < ~ < +- < 前导/后缀）。
class ExpParser {
  public:
    ExpParser(std::vector<Tok> toks, std::string& err) : toks_(std::move(toks)), err_(err) {}

    std::shared_ptr<Expr> parse() {
        auto e = parse_cond();
        if (!at_end()) { fail(err_str()); return nullptr; }
        return e;
    }

    // 只解析一个条件表达式（不要求整段结束）——宏默认值等「停在下个分隔符」的场景。
    std::shared_ptr<Expr> parse_cond_expr() { return parse_cond(); }

  private:
    std::vector<Tok> toks_;
    std::size_t p_ = 0;
    std::string& err_;

    public:
    const Tok& cur() const { return toks_[p_]; }
    bool at_end() const { return cur().t == Tok::T::kEnd; }
    void next() { if (!at_end()) ++p_; }
    bool sym(const char* s) const { return tok_is_sym(cur(), s); }
    bool ident(const char* s) const { return tok_is_ident(cur(), s); }

  private:

    void fail(const std::string& why) {
        if (err_.empty()) err_ = std::string("表达式解析失败：") + why;
    }
    std::string err_str() const {
        if (at_end()) return "表达式意外结束";
        if (cur().t == Tok::T::kStr) return "意外的字符串字面量 '" + cur().s + "'";
        if (cur().t == Tok::T::kIdent) return "意外的标识符 " + cur().s;
        if (cur().t == Tok::T::kNum) return "意外的数字 " + std::to_string(cur().n);
        return "意外的符号 '" + cur().s + "'";
    }

    std::shared_ptr<Expr> parse_cond() {
        auto a = parse_or();
        if (ident("if")) {
            next();
            auto c = parse_or();
            if (!ident("else")) { fail("条件式缺 else"); return nullptr; }
            next();
            auto b = parse_cond();
            auto e = mk_expr(ET::kCond);
            e->kids = {a, c, b};
            return e;
        }
        return a;
    }

    std::shared_ptr<Expr> parse_or() {
        auto a = parse_and();
        while (ident("or")) {
            next();
            auto b = parse_and();
            auto e = mk_expr(ET::kOr);
            e->kids = {a, b};
            a = e;
        }
        return a;
    }

    std::shared_ptr<Expr> parse_and() {
        auto a = parse_not();
        while (ident("and")) {
            next();
            auto b = parse_not();
            auto e = mk_expr(ET::kAnd);
            e->kids = {a, b};
            a = e;
        }
        return a;
    }

    std::shared_ptr<Expr> parse_not() {
        if (ident("not")) {
            next();
            auto v = parse_not();
            auto e = mk_expr(ET::kNot);
            e->kids = {v};
            return e;
        }
        return parse_cmp();
    }

    std::shared_ptr<Expr> parse_cmp() {
        auto a = parse_concat();
        for (;;) {
            if (sym("==") || sym("!=") || sym(">") || sym("<") || sym(">=") || sym("<=")) {
                const ET op = cur().s == "==" ? ET::kEq : cur().s == "!=" ? ET::kNe :
                              cur().s == ">" ? ET::kGt : cur().s == "<" ? ET::kLt :
                              cur().s == ">=" ? ET::kGe : ET::kLe;
                next();
                auto b = parse_concat();
                auto e = mk_expr(op);
                e->kids = {a, b};
                a = e;
                continue;
            }
            if (ident("in")) {
                next();
                auto b = parse_concat();
                auto e = mk_expr(ET::kIn);
                e->kids = {a, b};
                a = e;
                continue;
            }
            if (ident("not") && p_ + 1 < toks_.size() && tok_is_ident(toks_[p_ + 1], "in")) {
                next();
                next();
                auto b = parse_concat();
                auto e = mk_expr(ET::kNotIn);
                e->kids = {a, b};
                a = e;
                continue;
            }
            if (ident("is")) {
                next();
                bool neg = false;
                if (ident("not")) { neg = true; next(); }
                if (cur().t != Tok::T::kIdent) { fail("is 测试缺测试名"); return nullptr; }
                const std::string name = cur().s;
                next();
                auto e = mk_expr(ET::kTest);
                e->kids = {a};
                e->s = name;
                e->n = neg;
                a = e;
                continue;
            }
            break;
        }
        return a;
    }

    std::shared_ptr<Expr> parse_concat() {
        auto a = parse_arith();
        while (sym("~")) {
            next();
            auto b = parse_postfix();
            auto e = mk_expr(ET::kConcat);
            e->kids = {a, b};
            a = e;
        }
        return a;
    }

    std::shared_ptr<Expr> parse_arith() {
        auto a = parse_postfix();
        while (sym("+") || sym("-")) {
            const bool add = sym("+");
            next();
            auto b = parse_postfix();
            auto e = mk_expr(add ? ET::kAdd : ET::kSub);
            e->kids = {a, b};
            a = e;
        }
        return a;
    }

    std::shared_ptr<Expr> parse_postfix() {
        auto a = parse_term();
        for (;;) {
            if (sym(".")) {
                next();
                if (cur().t != Tok::T::kIdent) { fail("属性名缺失"); return nullptr; }
                const std::string name = cur().s;
                next();
                auto e = mk_expr(ET::kAttr);
                e->kids = {a};
                e->s = name;
                a = e;
                continue;
            }
            if (sym("[")) {
                next();
                std::shared_ptr<Expr> start, stop, step;
                if (!sym(":")) start = parse_cond();
                if (sym(":")) {
                    next();
                    if (!sym(":") && !sym("]")) stop = parse_cond();
                    if (sym(":")) {
                        next();
                        if (!sym("]")) step = parse_cond();
                    }
                    if (!sym("]")) { fail("切片缺 ]"); return nullptr; }
                    next();
                    auto e = mk_expr(ET::kSlice);
                    e->kids = {a, start, stop, step};
                    a = e;
                    continue;
                }
                if (!sym("]")) { fail("下标缺 ]"); return nullptr; }
                next();
                auto e = mk_expr(ET::kIndex);
                e->kids = {a, start};
                a = e;
                continue;
            }
            if (sym("(")) {
                auto e = mk_expr(ET::kCall);
                e->kids = {a};
                next();
                if (!sym(")")) {
                    for (;;) {
                        if (cur().t == Tok::T::kIdent && p_ + 1 < toks_.size() &&
                            tok_is_sym(toks_[p_ + 1], "=")) {
                            const std::string name = cur().s;
                            next();
                            next();
                            auto v = parse_cond();
                            e->kwargs.push_back(name);
                            e->args.push_back(v);
                        } else {
                            auto v = parse_cond();
                            e->kwargs.push_back("");
                            e->args.push_back(v);
                        }
                        if (sym(",")) { next(); continue; }
                        break;
                    }
                }
                if (!sym(")")) { fail("调用缺 )"); return nullptr; }
                next();
                a = e;
                continue;
            }
            if (sym("|")) {
                next();
                if (cur().t != Tok::T::kIdent) { fail("过滤器名缺失"); return nullptr; }
                const std::string name = cur().s;
                next();
                auto e = mk_expr(ET::kFilter);
                e->kids = {a};
                e->s = name;
                if (sym("(")) {
                    next();
                    if (!sym(")")) {
                        for (;;) {
                            auto v = parse_cond();
                            e->args.push_back(v);
                            if (sym(",")) { next(); continue; }
                            break;
                        }
                    }
                    if (!sym(")")) { fail("过滤器缺 )"); return nullptr; }
                    next();
                }
                a = e;
                continue;
            }
            break;
        }
        return a;
    }

    std::shared_ptr<Expr> parse_term() {
        if (sym("-")) {
            next();
            auto v = parse_term();
            auto e = mk_expr(ET::kNeg);
            e->kids = {v};
            return e;
        }
        return parse_primary();
    }

    std::shared_ptr<Expr> parse_primary() {
        if (cur().t == Tok::T::kNum) {
            auto e = mk_expr(ET::kNum);
            e->n = cur().n;
            next();
            return e;
        }
        if (cur().t == Tok::T::kStr) {
            auto e = mk_expr(ET::kStr);
            e->s = cur().s;
            next();
            return e;
        }
        if (sym("(")) {
            next();
            if (sym(")")) {
                next();
                return mk_expr(ET::kList);
            }
            auto first = parse_cond();
            if (sym(",")) {
                auto e = mk_expr(ET::kList);
                e->kids.push_back(first);
                while (sym(",")) {
                    next();
                    if (sym(")")) break;   // 允许尾逗号
                    e->kids.push_back(parse_cond());
                }
                if (!sym(")")) { fail("元组缺 )"); return nullptr; }
                next();
                return e;
            }
            if (!sym(")")) { fail("括号缺 )"); return nullptr; }
            next();
            return first;
        }
        if (sym("[")) {
            next();
            auto e = mk_expr(ET::kList);
            if (sym("]")) {
                next();
                return e;
            }
            for (;;) {
                e->kids.push_back(parse_cond());
                if (sym(",")) { next(); continue; }
                break;
            }
            if (!sym("]")) { fail("列表字面量缺 ]"); return nullptr; }
            next();
            return e;
        }
        if (cur().t == Tok::T::kIdent) {
            const std::string name = cur().s;
            next();
            if (name == "true") { auto e = mk_expr(ET::kBool); e->n = 1; return e; }
            if (name == "false") { auto e = mk_expr(ET::kBool); e->n = 0; return e; }
            if (name == "none") return mk_expr(ET::kNull);
            auto e = mk_expr(ET::kVar);
            e->s = name;
            return e;
        }
        fail(err_str());
        return nullptr;
    }
};

// 把一段源码解析成表达式树；失败返回 nullptr 并写 err。
std::shared_ptr<Expr> parse_expr(const std::string& src, std::string& err) {
    auto toks = tokenize(src, err);
    if (toks.empty()) return nullptr;
    ExpParser p(std::move(toks), err);
    return p.parse();
}

// ---- 语句 Op 与编译 -------------------------------------------------------
enum class Kind { kText, kExpr, kIf, kElif, kElse, kEndif, kFor, kEndfor, kSet, kMacro, kEndmacro };

struct Op {
    Kind kind = Kind::kText;
    std::string a;                          // kText 输出 / kExpr·kIf·kElif 表达式原文 / kFor 变量 / kSet 目标 / kMacro 名
    std::string b;                          // kSet 目标的属性名（空=普通变量）
    std::string v2;                         // kFor 解包第二变量（空=单变量）
    std::size_t jump = 0;                   // 条件假 / 已取分支的跳过目标（Op 下标）
    std::size_t end = 0;                    // kFor / kMacro 的闭合下标
    std::size_t pos = 0;                    // 模板字符位置（报错用）
    std::shared_ptr<Expr> expr;             // kExpr / kIf / kElif / kFor 迭代式 / kSet 右值
    std::vector<std::string> params;        // kMacro 形参名
    std::vector<std::shared_ptr<Expr>> defaults;  // kMacro 形参默认值（与 params 并行，nullptr=必填）
};

// 顶部空白切分（字符串字面量内的空白不进词）。
std::string trim(const std::string& s) {
    std::size_t a = 0, b = s.size();
    while (a < b && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
    while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
    return s.substr(a, b - a);
}

void lstrip_ws(std::string& s) {
    std::size_t k = 0;
    while (k < s.size() && std::isspace(static_cast<unsigned char>(s[k]))) ++k;
    s.erase(0, k);
}

void rstrip_ws(std::string& s) {
    std::size_t k = s.size();
    while (k > 0 && std::isspace(static_cast<unsigned char>(s[k - 1]))) --k;
    s.resize(k);
}

// 词法与编译。连字符语义按 Jinja：标签开 `{%-`/`{{-` 吃掉前一段文本的尾部空白，
// 标签收 `-%}`/`-}}` 吃掉后一段文本的头部空白（可跨多行、可叠加）。`{# #}` 注释同规则。
bool compile(const std::string& tpl_in, std::vector<Op>& ops, std::string& err) {
    std::string tpl = tpl_in;
    if (!tpl.empty() && tpl.back() == '\n') tpl.pop_back();   // keep_trailing_newline=false：源末尾单个换行不输出
    ops.clear();
    std::string text;
    bool prev_minus_close = false;   // 上一标签以 '-' 收尾 → 下一段文本头空白要吃掉
    std::size_t i = 0;
    const auto fail = [&](const std::string& why, std::size_t at) {
        err = "模板第 " + std::to_string(at) + " 字符处：" + why;
        return false;
    };
    const auto emit_text = [&]() {
        if (prev_minus_close) {
            lstrip_ws(text);
            prev_minus_close = false;
        }
        if (!text.empty()) {
            Op op;
            op.kind = Kind::kText;
            op.a = text;
            ops.push_back(std::move(op));
        }
        text.clear();
    };
    for (;;) {
        // 找下一处标签开端（{{ / {% / {# 中最早者）。
        std::size_t best = std::string::npos;
        std::string open;
        for (const char* o : {"{{", "{%", "{#"}) {
            const std::size_t at = tpl.find(o, i);
            if (at != std::string::npos && (best == std::string::npos || at < best)) {
                best = at;
                open = o;
            }
        }
        if (best == std::string::npos) {
            text += tpl.substr(i);
            break;
        }
        text += tpl.substr(i, best - i);
        i = best;
        const bool minus_open = i + 2 < tpl.size() && tpl[i + 2] == '-';
        const std::size_t open_adj = minus_open ? 3 : 2;
        const std::string close = open == "{{" ? "}}" : open == "{%" ? "%}" : "#}";
        const std::size_t close_at = tpl.find(close, i + open_adj);
        if (close_at == std::string::npos) return fail("标签未闭合", i);
        const bool minus_close = close_at > i + open_adj && tpl[close_at - 1] == '-';
        const std::size_t begin = i + open_adj;
        const std::size_t len = close_at - begin - (minus_close ? 1 : 0);
        const std::string body = trim(tpl.substr(begin, len));
        // 空白削减（顺序无关：一吃尾一吃头）。
        if (minus_open) rstrip_ws(text);
        emit_text();
        prev_minus_close = minus_close;
        if (open == "{#") { i = close_at + 2; continue; }   // 注释：不产生输出
        if (open == "{{") {
            Op op;
            op.kind = Kind::kExpr;
            op.a = body;
            op.pos = i;
            op.expr = parse_expr(body, err);
            if (!op.expr) return fail(err, i);
            ops.push_back(std::move(op));
            i = close_at + 2;
            continue;
        }
        // {% ... %} 语句。
        const std::size_t sp = body.find(' ');
        const std::string k = sp == std::string::npos ? body : body.substr(0, sp);
        const std::string tail = sp == std::string::npos ? "" : trim(body.substr(sp + 1));
        Op op;
        op.pos = i;
        if (k == "if" || k == "elif") {
            op.kind = k == "if" ? Kind::kIf : Kind::kElif;
            op.a = tail;
            op.expr = parse_expr(tail, err);
            if (!op.expr) return fail(err, i);
        } else if (k == "else") {
            op.kind = Kind::kElse;
        } else if (k == "endif") {
            op.kind = Kind::kEndif;
        } else if (k == "endfor") {
            op.kind = Kind::kEndfor;
        } else if (k == "set") {
            op.kind = Kind::kSet;
            auto toks = tokenize(tail, err);
            if (toks.empty()) return fail(err, i);
            std::size_t eq = std::string::npos;
            int depth = 0;
            for (std::size_t j = 0; j < toks.size(); ++j) {
                if (tok_is_sym(toks[j], "(") || tok_is_sym(toks[j], "[")) ++depth;
                else if (tok_is_sym(toks[j], ")") || tok_is_sym(toks[j], "]")) --depth;
                else if (depth == 0 && tok_is_sym(toks[j], "=")) { eq = j; break; }
            }
            if (eq == std::string::npos || eq == 0) return fail("set 缺 =", i);
            // 目标：标识符 [ . 标识符 ]
            op.a = toks[0].s;
            if (toks[0].t != Tok::T::kIdent) return fail("set 目标不是变量", i);
            if (eq >= 3 && tok_is_sym(toks[1], ".") && toks[2].t == Tok::T::kIdent) {
                op.a = toks[0].s;
                op.b = toks[2].s;
                if (eq != 3) return fail("set 属性目标格式错误", i);
            } else if (eq != 1) {
                return fail("set 目标格式错误", i);
            }
            std::vector<Tok> rhs(toks.begin() + static_cast<std::ptrdiff_t>(eq + 1), toks.end());
            op.expr = [&]() -> std::shared_ptr<Expr> {
                ExpParser p(std::move(rhs), err);
                return p.parse();
            }();
            if (!op.expr) return fail(err, i);
        } else if (k == "for") {
            op.kind = Kind::kFor;
            auto toks = tokenize(tail, err);
            if (toks.empty()) return fail(err, i);
            ExpParser p(std::move(toks), err);
            if (p.cur().t != Tok::T::kIdent) return fail("for 缺循环变量", i);
            op.a = p.cur().s;
            p.next();
            if (tok_is_sym(p.cur(), ",")) {
                p.next();
                if (p.cur().t != Tok::T::kIdent) return fail("for 解包缺第二变量", i);
                op.v2 = p.cur().s;
                p.next();
            }
            if (!tok_is_ident(p.cur(), "in")) return fail("for 缺 in", i);
            p.next();
            op.expr = p.parse();
            if (!op.expr) return fail(err, i);
        } else if (k == "macro") {
            op.kind = Kind::kMacro;
            // 形如 macro 名(形参列表)；形参可为「名=默认值表达式」。
            auto toks = tokenize(tail, err);
            if (toks.empty()) return fail("macro 缺名字", i);
            ExpParser p(std::move(toks), err);
            if (p.cur().t != Tok::T::kIdent) return fail("macro 名缺失", i);
            op.a = p.cur().s;
            p.next();
            if (tok_is_sym(p.cur(), "(")) {
                p.next();
                if (!tok_is_sym(p.cur(), ")")) {
                    for (;;) {
                        if (p.cur().t != Tok::T::kIdent) return fail("macro 形参名缺失", i);
                        op.params.push_back(p.cur().s);
                        p.next();
                        if (tok_is_sym(p.cur(), "=")) {
                            p.next();
                            auto def = p.parse_cond_expr();
                            if (!def) return fail(err, i);
                            op.defaults.push_back(def);
                        } else {
                            op.defaults.push_back(nullptr);
                        }
                        if (tok_is_sym(p.cur(), ",")) { p.next(); continue; }
                        break;
                    }
                }
                if (!tok_is_sym(p.cur(), ")")) return fail("macro 参数缺 )", i);
                p.next();
            } else if (!p.at_end()) {
                return fail("macro 参数格式错误", i);
            }
            if (!p.at_end()) return fail("macro 定义多余内容", i);
        } else if (k == "endmacro") {
            op.kind = Kind::kEndmacro;
        } else {
            return fail("不认识的语句：" + body, i);
        }
        ops.push_back(std::move(op));
        i = close_at + 2;
    }
    // 模板尾文本（若上一标签以 '-' 收尾，同样去头空白）。
    emit_text();
    if (!text.empty()) {
        Op op;
        op.kind = Kind::kText;
        op.a = text;
        ops.push_back(std::move(op));
    }

    // 配对：if/elif/else/endif、for/endfor、macro/endmacro。
    std::vector<std::size_t> ifs, fors, macros;
    for (std::size_t idx = 0; idx < ops.size(); ++idx) {
        switch (ops[idx].kind) {
            case Kind::kIf: ifs.push_back(idx); break;
            case Kind::kFor: fors.push_back(idx); break;
            case Kind::kMacro: macros.push_back(idx); break;
            case Kind::kEndfor: {
                if (fors.empty()) { err = "多余的 endfor"; return false; }
                const std::size_t f = fors.back();
                fors.pop_back();
                ops[f].end = idx;
                break;
            }
            case Kind::kEndmacro: {
                if (macros.empty()) { err = "多余的 endmacro"; return false; }
                const std::size_t f = macros.back();
                macros.pop_back();
                ops[f].end = idx;
                break;
            }
            case Kind::kEndif: {
                if (ifs.empty()) { err = "多余的 endif"; return false; }
                const std::size_t start = ifs.back();
                ifs.pop_back();
                std::vector<std::size_t> chain;
                int depth = 0;
                for (std::size_t c = start; c < idx; ++c) {
                    switch (ops[c].kind) {
                        case Kind::kIf:
                            if (c == start) chain.push_back(c);
                            else ++depth;
                            break;
                        case Kind::kElif:
                        case Kind::kElse:
                            if (depth == 0) chain.push_back(c);
                            break;
                        case Kind::kEndif:
                            if (depth > 0) --depth;
                            break;
                        default: break;
                    }
                }
                for (std::size_t k = 0; k + 1 < chain.size(); ++k) {
                    ops[chain[k]].jump = chain[k + 1];
                }
                ops[chain.back()].jump = idx;
                break;
            }
            default: break;
        }
    }
    if (!ifs.empty()) { err = "if 未闭合"; return false; }
    if (!fors.empty()) { err = "for 未闭合"; return false; }
    if (!macros.empty()) { err = "macro 未闭合"; return false; }
    return true;
}

// ---- 运行环境 -------------------------------------------------------------
struct Env {
    const std::vector<Message>* msgs = nullptr;
    const std::vector<Op>* ops = nullptr;
    std::vector<std::map<std::string, Jv>> scopes;   // 作用域栈：内层优先
    bool taken = false;                              // if/elif 链：已取分支顺延
    std::vector<bool> chain;                         // 链外层 taken 栈（kIf 压入、kEndif 弹出）

    Jv* find(const std::string& name) {
        for (auto it = scopes.rbegin(); it != scopes.rend(); ++it) {
            auto f = it->find(name);
            if (f != it->end()) return &f->second;
        }
        return nullptr;
    }
};

// ---- 表达式求值 -----------------------------------------------------------
bool eval_expr(const Expr& e, Env& env, Jv& out, std::string& err);
bool run(const std::vector<Op>& ops, std::size_t from, std::size_t to, Env& env,
         std::string& out, std::string& err);

// 取属性/方法绑定。未定义属性一律软性返回未定义（裁决：消息的 reasoning_content 等属性未定义）。
Jv get_attr(const Jv& v, const std::string& name) {
    switch (v.k) {
        case Jv::K::kDict:
            if (v.dict) {
                for (const auto& kv : *v.dict) {
                    if (kv.first == name) return kv.second;
                }
            }
            return Jv();
        case Jv::K::kMsg:
            if (name == "role") return jv_str(v.msg->role);
            if (name == "content") return jv_str(v.msg->content);
            return Jv();   // reasoning_content / tool_calls 等 → 未定义
        case Jv::K::kStr:
            if (name == "startswith" || name == "endswith") return jv_bound(v.s, name);
            return Jv();
        default: return Jv();
    }
}

bool eval_filter(const std::string& name, const Jv& v, const std::vector<std::shared_ptr<Expr>>& args,
                 Env& env, Jv& out, std::string& err) {
    if (name == "trim") {
        if (v.k != Jv::K::kStr) { err = "trim 只能作用于字符串"; return false; }
        std::string s = v.s;
        lstrip_ws(s);
        rstrip_ws(s);
        out = jv_str(std::move(s));
        return true;
    }
    if (name == "default") {
        if (args.size() != 1) { err = "default 需要 1 个参数"; return false; }
        if (v.k == Jv::K::kUndef) return eval_expr(*args[0], env, out, err);
        out = v;
        return true;
    }
    if (name == "length") {
        switch (v.k) {
            case Jv::K::kList: out = jv_num(v.list ? static_cast<long long>(v.list->size()) : 0); return true;
            case Jv::K::kStr: out = jv_num(static_cast<long long>(v.s.size())); return true;
            case Jv::K::kDict: out = jv_num(v.dict ? static_cast<long long>(v.dict->size()) : 0); return true;
            case Jv::K::kUndef: out = jv_num(0); return true;   // 裁决：未定义 |length 按 0
            default: err = "length 不能作用于该类型"; return false;
        }
    }
    if (name == "safe") { out = v; return true; }               // 从不做 HTML 转义，safe 恒等
    if (name == "string") {
        std::string s;
        if (!jv_to_str(v, s, err)) return false;
        out = jv_str(std::move(s));
        return true;
    }
    if (name == "tojson" || name == "items") {
        err = "过滤器 " + name + " 未实现（仅工具/视觉分支出现，文本输入不执行）——遇执行即报错，不静默错译";
        return false;
    }
    err = "不认识的过滤器：" + name;
    return false;
}

bool eval_test(const std::string& name, bool neg, const Jv& v, Jv& out, std::string& err) {
    bool r = false;
    if (name == "defined") r = v.k != Jv::K::kUndef;
    else if (name == "undefined") r = v.k == Jv::K::kUndef;
    else if (name == "string") r = v.k == Jv::K::kStr;
    else if (name == "none") r = v.k == Jv::K::kNull;
    else if (name == "true") r = v.k == Jv::K::kBool && v.b;
    else if (name == "false") r = v.k == Jv::K::kBool && !v.b;
    else if (name == "iterable") r = v.k == Jv::K::kStr || v.k == Jv::K::kList || v.k == Jv::K::kDict;
    else if (name == "mapping") r = v.k == Jv::K::kDict;
    else if (name == "number") r = v.k == Jv::K::kNum;
    else if (name == "sequence") r = v.k == Jv::K::kStr || v.k == Jv::K::kList;
    else { err = "不认识的测试：" + name; return false; }
    out = jv_bool(neg ? !r : r);
    return true;
}

bool call_macro(const Jv& m, const std::vector<std::shared_ptr<Expr>>& args,
                const std::vector<std::string>& kwargs, Env& env, Jv& out, std::string& err) {
    for (const std::string& kw : kwargs) {
        if (!kw.empty()) { err = "宏调用不支持关键字实参：" + kw; return false; }
    }
    if (args.size() > m.macro->params.size()) { err = "宏实参过多：" + m.macro->name; return false; }
    std::vector<Jv> vals;
    vals.reserve(args.size());
    for (const auto& a : args) {
        Jv v;
        if (!eval_expr(*a, env, v, err)) return false;
        vals.push_back(std::move(v));
    }
    env.scopes.push_back({});
    auto& s = env.scopes.back();
    for (std::size_t i = 0; i < m.macro->params.size(); ++i) {
        if (i < vals.size()) { s[m.macro->params[i]] = vals[i]; }
        else if (m.macro->defaults[i]) {
            Jv d;
            if (!eval_expr(*m.macro->defaults[i], env, d, err)) { env.scopes.pop_back(); return false; }
            s[m.macro->params[i]] = d;
        } else {
            s[m.macro->params[i]] = Jv();
        }
    }
    std::string buf;
    const bool ok = run(*env.ops, m.macro->begin + 1, m.macro->end, env, buf, err);
    env.scopes.pop_back();
    if (!ok) {
        std::string prefix = "宏 " + m.macro->name + " 执行：";
        if (err.rfind(prefix, 0) != 0) err = prefix + err;
        return false;
    }
    out = jv_str(std::move(buf));
    return true;
}

bool eval_expr(const Expr& e, Env& env, Jv& out, std::string& err) {
    switch (e.t) {
        case ET::kStr: out = jv_str(e.s); return true;
        case ET::kNum: out = jv_num(e.n); return true;
        case ET::kBool: out = jv_bool(e.n != 0); return true;
        case ET::kNull: out = jv_null(); return true;
        case ET::kVar: {
            if (Jv* v = env.find(e.s)) { out = *v; return true; }
            out = Jv();   // 取值位置不报错；判据在「直接输出 / 参与运算」等消费点
            return true;
        }
        case ET::kList: {
            Jv l = jv_list();
            for (const auto& k : e.kids) {
                Jv v;
                if (!eval_expr(*k, env, v, err)) return false;
                l.list->push_back(std::move(v));
            }
            out = std::move(l);
            return true;
        }
        case ET::kNeg: {
            Jv a;
            if (!eval_expr(*e.kids[0], env, a, err)) return false;
            if (a.k != Jv::K::kNum) { err = "取负的运算数不是数字"; return false; }
            out = jv_num(-a.n);
            return true;
        }
        case ET::kNot: {
            Jv a;
            if (!eval_expr(*e.kids[0], env, a, err)) return false;
            out = jv_bool(!jv_truthy(a));
            return true;
        }
        case ET::kAdd: {
            Jv a, b;
            if (!eval_expr(*e.kids[0], env, a, err) || !eval_expr(*e.kids[1], env, b, err)) return false;
            if (a.k == Jv::K::kStr && b.k == Jv::K::kStr) { out = jv_str(a.s + b.s); return true; }
            if (a.k == Jv::K::kNum && b.k == Jv::K::kNum) { out = jv_num(a.n + b.n); return true; }
            if (a.k == Jv::K::kUndef || b.k == Jv::K::kUndef) { err = "未定义值参与加法"; return false; }
            err = "加法两侧类型不一致（与 Jinja 一致：字符串只拼字符串、数字加法）";
            return false;
        }
        case ET::kSub: {
            Jv a, b;
            if (!eval_expr(*e.kids[0], env, a, err) || !eval_expr(*e.kids[1], env, b, err)) return false;
            if (a.k == Jv::K::kNum && b.k == Jv::K::kNum) { out = jv_num(a.n - b.n); return true; }
            if (a.k == Jv::K::kUndef || b.k == Jv::K::kUndef) { err = "未定义值参与减法"; return false; }
            err = "减法两侧必须是数字";
            return false;
        }
        case ET::kConcat: {
            Jv a, b;
            if (!eval_expr(*e.kids[0], env, a, err) || !eval_expr(*e.kids[1], env, b, err)) return false;
            std::string x, y;
            if (!jv_to_str(a, x, err)) return false;
            if (!jv_to_str(b, y, err)) return false;
            out = jv_str(x + y);
            return true;
        }
        case ET::kEq:
        case ET::kNe: {
            Jv a, b;
            if (!eval_expr(*e.kids[0], env, a, err) || !eval_expr(*e.kids[1], env, b, err)) return false;
            const bool eq = jv_equal(a, b);
            out = jv_bool(e.t == ET::kEq ? eq : !eq);
            return true;
        }
        case ET::kGt:
        case ET::kLt:
        case ET::kGe:
        case ET::kLe: {
            Jv a, b;
            if (!eval_expr(*e.kids[0], env, a, err) || !eval_expr(*e.kids[1], env, b, err)) return false;
            int cmp = 0;
            if (a.k == Jv::K::kNum && b.k == Jv::K::kNum) cmp = (a.n > b.n) - (a.n < b.n);
            else if (a.k == Jv::K::kStr && b.k == Jv::K::kStr) cmp = (a.s > b.s) - (a.s < b.s);
            else { err = "比较两侧类型不一致"; return false; }
            const bool r = e.t == ET::kGt ? cmp > 0 : e.t == ET::kLt ? cmp < 0 :
                           e.t == ET::kGe ? cmp >= 0 : cmp <= 0;
            out = jv_bool(r);
            return true;
        }
        case ET::kIn:
        case ET::kNotIn: {
            Jv a, b;
            if (!eval_expr(*e.kids[0], env, a, err) || !eval_expr(*e.kids[1], env, b, err)) return false;
            bool m = false;
            if (a.k == Jv::K::kUndef || b.k == Jv::K::kUndef) m = false;
            else if (b.k == Jv::K::kList) {
                for (const auto& el : *b.list) {
                    if (jv_equal(a, el)) { m = true; break; }
                }
            } else if (b.k == Jv::K::kStr && a.k == Jv::K::kStr) {
                m = b.s.find(a.s) != std::string::npos;
            } else if (b.k == Jv::K::kDict && a.k == Jv::K::kStr) {
                for (const auto& kv : *b.dict) {
                    if (kv.first == a.s) { m = true; break; }
                }
            } else {
                err = "in 右侧类型不可判成员"; return false;
            }
            out = jv_bool(e.t == ET::kIn ? m : !m);
            return true;
        }
        case ET::kTest: {
            Jv a;
            if (!eval_expr(*e.kids[0], env, a, err)) return false;
            return eval_test(e.s, e.n != 0, a, out, err);
        }
        case ET::kAnd: {
            Jv a;
            if (!eval_expr(*e.kids[0], env, a, err)) return false;
            if (!jv_truthy(a)) { out = std::move(a); return true; }
            return eval_expr(*e.kids[1], env, out, err);
        }
        case ET::kOr: {
            Jv a;
            if (!eval_expr(*e.kids[0], env, a, err)) return false;
            if (jv_truthy(a)) { out = std::move(a); return true; }
            return eval_expr(*e.kids[1], env, out, err);
        }
        case ET::kCond: {
            Jv c;
            if (!eval_expr(*e.kids[1], env, c, err)) return false;
            return eval_expr(*e.kids[jv_truthy(c) ? 0 : 2], env, out, err);
        }
        case ET::kAttr: {
            Jv a;
            if (!eval_expr(*e.kids[0], env, a, err)) return false;
            out = get_attr(a, e.s);
            return true;
        }
        case ET::kIndex: {
            Jv a, key;
            if (!eval_expr(*e.kids[0], env, a, err) || !eval_expr(*e.kids[1], env, key, err)) return false;
            if (a.k == Jv::K::kList && key.k == Jv::K::kNum) {
                if (key.n < 0 || static_cast<std::size_t>(key.n) >= a.list->size()) {
                    err = "列表下标越界：messages[" + std::to_string(key.n) + "]";
                    return false;
                }
                out = (*a.list)[static_cast<std::size_t>(key.n)];
                return true;
            }
            if (a.k == Jv::K::kMsg && key.k == Jv::K::kStr) {
                if (key.s == "role") { out = jv_str(a.msg->role); return true; }
                if (key.s == "content") { out = jv_str(a.msg->content); return true; }
                err = "未知消息键：" + key.s;
                return false;
            }
            if (a.k == Jv::K::kDict && key.k == Jv::K::kStr) {
                for (const auto& kv : *a.dict) {
                    if (kv.first == key.s) { out = kv.second; return true; }
                }
                out = Jv();
                return true;
            }
            err = "不能对该值做下标";
            return false;
        }
        case ET::kSlice: {
            Jv a;
            if (!eval_expr(*e.kids[0], env, a, err)) return false;
            if (a.k != Jv::K::kList) { err = "切片只能用于列表"; return false; }
            if (e.kids[1] || e.kids[2]) { err = "只支持完整反向切片 [::-1]"; return false; }
            Jv step;
            if (!eval_expr(*e.kids[3], env, step, err)) return false;
            if (step.k != Jv::K::kNum) { err = "切片步长必须是整数"; return false; }
            if (step.n >= 0) { err = "切片只支持负步长"; return false; }
            Jv l = jv_list();
            const std::size_t n = a.list->size();
            l.list->reserve(n);
            for (std::size_t k = n; k-- > 0;) l.list->push_back((*a.list)[k]);
            out = std::move(l);
            return true;
        }
        case ET::kCall: {
            const Expr* c = e.kids[0].get();
            if (c && c->t == ET::kVar) {
                if (c->s == "raise_exception") {
                    if (e.args.size() != 1) { err = "raise_exception 需要恰好 1 个参数"; return false; }
                    Jv m;
                    if (!eval_expr(*e.args[0], env, m, err)) return false;
                    std::string mm;
                    if (!jv_to_str(m, mm, err)) return false;
                    err = "raise_exception 触发：" + mm;
                    return false;
                }
                if (c->s == "namespace") {
                    Jv d = jv_dict();
                    for (std::size_t i = 0; i < e.args.size(); ++i) {
                        if (i >= e.kwargs.size() || e.kwargs[i].empty()) {
                            err = "namespace 只支持「名=值」实参";
                            return false;
                        }
                        Jv v;
                        if (!eval_expr(*e.args[i], env, v, err)) return false;
                        d.dict->emplace_back(e.kwargs[i], std::move(v));
                    }
                    out = std::move(d);
                    return true;
                }
            }
            Jv callee;
            if (!eval_expr(*e.kids[0], env, callee, err)) return false;
            if (callee.k == Jv::K::kMacro) return call_macro(callee, e.args, e.kwargs, env, out, err);
            if (callee.k == Jv::K::kBound) {
                if (e.args.size() != 1) { err = "字符串方法需要 1 个参数"; return false; }
                Jv p;
                if (!eval_expr(*e.args[0], env, p, err)) return false;
                if (p.k != Jv::K::kStr) { err = "字符串方法参数必须是字符串"; return false; }
                const std::string& base = callee.bound->first;
                const std::string& name = callee.bound->second;
                if (name == "startswith") {
                    out = jv_bool(base.rfind(p.s, 0) == 0);
                    return true;
                }
                if (name == "endswith") {
                    out = jv_bool(base.size() >= p.s.size() &&
                                  base.compare(base.size() - p.s.size(), p.s.size(), p.s) == 0);
                    return true;
                }
                err = "不支持的方法：" + name;
                return false;
            }
            err = "不能调用该值";
            return false;
        }
        case ET::kFilter: {
            Jv v;
            if (!eval_expr(*e.kids[0], env, v, err)) return false;
            return eval_filter(e.s, v, e.args, env, out, err);
        }
    }
    err = "未实现的表达式节点";
    return false;
}

// ---- 解释执行 -------------------------------------------------------------
bool run(const std::vector<Op>& ops, std::size_t from, std::size_t to, Env& env,
         std::string& out, std::string& err) {
    std::size_t i = from;
    while (i < to) {
        const Op& op = ops[i];
        const std::string pos_str = "模板第 " + std::to_string(op.pos) + " 字符处：";
        switch (op.kind) {
            case Kind::kText: out += op.a; ++i; break;
            case Kind::kExpr: {
                Jv v;
                if (!eval_expr(*op.expr, env, v, err)) {
                    err = pos_str + "求值失败：" + op.a + "（" + err + "）";
                    return false;
                }
                if (v.k == Jv::K::kUndef) {
                    err = pos_str + "求值失败：" + op.a + "（未定义值落入输出位置——unknown variable）";
                    return false;
                }
                std::string s;
                if (!jv_to_str(v, s, err)) {
                    err = pos_str + "求值失败：" + op.a + "（" + err + "）";
                    return false;
                }
                out += s;
                ++i;
                break;
            }
            case Kind::kIf: {
                Jv c;
                if (!eval_expr(*op.expr, env, c, err)) {
                    err = pos_str + "条件求值失败：" + op.a + "（" + err + "）";
                    return false;
                }
                env.chain.push_back(env.taken);
                if (jv_truthy(c)) { ++i; env.taken = true; }
                else { env.taken = false; i = op.jump; }
                break;
            }
            case Kind::kElif: {
                if (env.taken) { i = op.jump; break; }
                Jv c;
                if (!eval_expr(*op.expr, env, c, err)) {
                    err = pos_str + "条件求值失败：" + op.a + "（" + err + "）";
                    return false;
                }
                if (jv_truthy(c)) { ++i; env.taken = true; }
                else { i = op.jump; }
                break;
            }
            case Kind::kElse:
                if (env.taken) { i = op.jump; break; }
                env.taken = true;
                ++i;
                break;
            case Kind::kEndif:
                ++i;
                env.taken = env.chain.back();
                env.chain.pop_back();
                break;
            case Kind::kSet: {
                Jv v;
                if (!eval_expr(*op.expr, env, v, err)) {
                    err = pos_str + "set 求值失败：" + op.b + "（" + err + "）";
                    return false;
                }
                if (!op.b.empty()) {
                    Jv* obj = env.find(op.a);
                    if (!obj || obj->k != Jv::K::kDict) {
                        err = pos_str + "set 目标不是 namespace：" + op.a + "." + op.b;
                        return false;
                    }
                    for (auto& kv : *obj->dict) {
                        if (kv.first == op.b) { kv.second = v; break; }
                    }
                    obj->dict->emplace_back(op.b, std::move(v));
                } else {
                    env.scopes.back()[op.a] = std::move(v);
                }
                ++i;
                break;
            }
            case Kind::kFor: {
                Jv it;
                if (!eval_expr(*op.expr, env, it, err)) {
                    err = pos_str + "for 迭代式求值失败（" + err + "）";
                    return false;
                }
                // 未定义可迭代对象按空列表（工具/视觉分支缺失变量，文本输入不进入其实体）。
                if (it.k == Jv::K::kUndef || (it.k == Jv::K::kList && (!it.list || it.list->empty()))) {
                    i = op.end + 1;
                    break;
                }
                if (it.k != Jv::K::kList) {
                    err = pos_str + "for 只能迭代列表";
                    return false;
                }
                const std::size_t n = it.list->size();
                env.scopes.push_back({});
                bool body_ok = true;
                for (std::size_t k = 0; k < n; ++k) {
                    const Jv& item = (*it.list)[k];
                    auto& scope = env.scopes.back();
                    if (op.v2.empty()) {
                        scope[op.a] = item;
                    } else {
                        if (item.k != Jv::K::kList || !item.list || item.list->size() != 2) {
                            err = pos_str + "for 解包失败（需要元组对）";
                            body_ok = false;
                            break;
                        }
                        scope[op.a] = (*item.list)[0];
                        scope[op.v2] = (*item.list)[1];
                    }
                    Jv loop = jv_dict();
                    loop.dict->emplace_back("index0", jv_num(static_cast<long long>(k)));
                    loop.dict->emplace_back("index1", jv_num(static_cast<long long>(k + 1)));
                    loop.dict->emplace_back("first", jv_bool(k == 0));
                    loop.dict->emplace_back("last", jv_bool(k + 1 == n));
                    loop.dict->emplace_back("previtem", k > 0 ? (*it.list)[k - 1] : Jv());
                    loop.dict->emplace_back("nextitem", k + 1 < n ? (*it.list)[k + 1] : Jv());
                    scope["loop"] = std::move(loop);
                    if (!run(ops, i + 1, op.end, env, out, err)) {
                        err = "循环体内：" + err;
                        body_ok = false;
                        break;
                    }
                }
                env.scopes.pop_back();
                if (!body_ok) return false;
                i = op.end + 1;
                break;
            }
            case Kind::kMacro: i = op.end + 1; break;   // 宏体仅在调用时执行（编译期已登记）
            case Kind::kEndmacro: i = to; break;        // 防御：正常到不了
            case Kind::kEndfor: i = to; break;          // 防御：正常到不了
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
    env.ops = &ops;
    std::map<std::string, Jv> global;
    Jv ml = jv_list();
    for (const Message& m : msgs) ml.list->push_back(jv_msg(&m));
    global["messages"] = std::move(ml);
    global["add_generation_prompt"] = jv_bool(add_generation_prompt);
    global["bos_token"] = jv_str("");
    global["eos_token"] = jv_str("");
    // 宏登记到全局作用域（定义位置之后调用，编译期一次性收起）。
    for (std::size_t idx = 0; idx < ops.size(); ++idx) {
        if (ops[idx].kind != Kind::kMacro) continue;
        Jv m;
        m.k = Jv::K::kMacro;
        m.macro = std::make_shared<Jv::Macro>();
        m.macro->name = ops[idx].a;
        m.macro->params = ops[idx].params;
        m.macro->defaults = ops[idx].defaults;
        m.macro->begin = idx;
        m.macro->end = ops[idx].end;
        global[ops[idx].a] = std::move(m);
    }
    env.scopes.push_back(std::move(global));
    out.clear();
    return run(ops, 0, ops.size(), env, out, err);
}

}  // namespace qinfer::serve