// 手写 JSON 子集的解析与序列化实现。
// 解析为单趟递归下降：跳过空白 → 一个值 → 再看尾部，任何越界输入都返回假并给出带位置的错误；
// 字符串内 \uXXXX 支持代理对（高位必须紧跟低位，孤立低位报错），裸控制字符报错；
// 数字语法 -? (0 | [1-9][0-9]*) (. [0-9]+)? ([eE][+-]?[0-9]+)?，用 from_chars 转 double（与 locale 无关）。
// 序列化紧凑：数字整数值且 |x|<9e15 走整数字面量，否则 %.9g；字符串经 json_escape。
#include "serve/json.hpp"

#include <charconv>
#include <cmath>
#include <cstdio>
#include <string>

namespace qinfer::serve {

namespace {

bool is_ws(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}

// 把码点编码为 UTF-8 追加进 out。
void append_utf8(std::string& out, std::uint32_t cp) {
    if (cp < 0x80) {
        out.push_back(static_cast<char>(cp));
    } else if (cp < 0x800) {
        out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
        out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
}

int hex_digit(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

struct Parser {
    const std::string& s;
    std::size_t pos = 0;
    std::string& err;

    bool fail(const char* msg) {
        err = "第 " + std::to_string(pos + 1) + " 字符处：" + msg;
        return false;
    }

    char peek() const { return pos < s.size() ? s[pos] : '\0'; }

    void skip_ws() {
        while (pos < s.size() && is_ws(s[pos])) ++pos;
    }

    bool match_literal(const char* lit) {
        const std::size_t len = std::char_traits<char>::length(lit);
        if (pos + len > s.size() || s.compare(pos, len, lit) != 0) return false;
        pos += len;
        return true;
    }

    bool parse_value(Json& out);

    bool parse_number(Json& out) {
        const std::size_t start = pos;
        if (peek() == '-') ++pos;
        if (pos >= s.size()) return fail("数字缺整数部分");
        if (s[pos] == '0') {
            ++pos;   // 0 后不再吃数字
        } else if (s[pos] >= '1' && s[pos] <= '9') {
            while (pos < s.size() && s[pos] >= '0' && s[pos] <= '9') ++pos;
        } else {
            return fail("数字缺整数部分");
        }
        if (pos < s.size() && s[pos] == '.') {
            ++pos;
            if (pos >= s.size() || s[pos] < '0' || s[pos] > '9') return fail("小数部分缺数字");
            while (pos < s.size() && s[pos] >= '0' && s[pos] <= '9') ++pos;
        }
        if (pos < s.size() && (s[pos] == 'e' || s[pos] == 'E')) {
            ++pos;
            if (pos < s.size() && (s[pos] == '+' || s[pos] == '-')) ++pos;
            if (pos >= s.size() || s[pos] < '0' || s[pos] > '9') return fail("指数缺数字");
            while (pos < s.size() && s[pos] >= '0' && s[pos] <= '9') ++pos;
        }
        const std::string tok = s.substr(start, pos - start);
        double v = 0.0;
        const auto res = std::from_chars(tok.data(), tok.data() + tok.size(), v);
        if (res.ec != std::errc()) return fail("数字超范围");
        out = Json::numv(v);
        return true;
    }

    // pos 已指向 \u 的 'u'，读取 4 位十六进制；高位代理后必须紧跟 \uXXXX 低位代理。
    bool parse_unicode_escape(std::string& str) {
        ++pos;
        std::uint32_t cp = 0;
        for (int i = 0; i < 4; ++i) {
            if (pos >= s.size()) return fail("\\u 缺十六进制位");
            const int d = hex_digit(s[pos]);
            if (d < 0) return fail("\\u 非法十六进制位");
            cp = (cp << 4) | static_cast<std::uint32_t>(d);
            ++pos;
        }
        if (cp >= 0xD800 && cp <= 0xDBFF) {
            if (pos + 1 >= s.size() || s[pos] != '\\' || s[pos + 1] != 'u')
                return fail("高位代理后缺 \\u 低位代理");
            pos += 2;
            std::uint32_t lo = 0;
            for (int i = 0; i < 4; ++i) {
                if (pos >= s.size()) return fail("\\u 缺十六进制位");
                const int d = hex_digit(s[pos]);
                if (d < 0) return fail("低位代理的 \\u 非法十六进制位");
                lo = (lo << 4) | static_cast<std::uint32_t>(d);
                ++pos;
            }
            if (lo < 0xDC00 || lo > 0xDFFF) return fail("高位代理后不是低位代理");
            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
        } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
            return fail("孤立低位代理");
        }
        append_utf8(str, cp);
        return true;
    }

    bool parse_string(Json& out) {
        ++pos;   // 跳过开引号
        std::string str;
        while (true) {
            if (pos >= s.size()) return fail("字符串未闭合");
            const unsigned char c = static_cast<unsigned char>(s[pos]);
            if (c == '"') {
                ++pos;
                out = Json::strv(std::move(str));
                return true;
            }
            if (c == '\\') {
                ++pos;
                if (pos >= s.size()) return fail("字符串未闭合");
                switch (s[pos]) {
                    case '"': str.push_back('"'); break;
                    case '\\': str.push_back('\\'); break;
                    case '/': str.push_back('/'); break;
                    case 'b': str.push_back('\b'); break;
                    case 'f': str.push_back('\f'); break;
                    case 'n': str.push_back('\n'); break;
                    case 'r': str.push_back('\r'); break;
                    case 't': str.push_back('\t'); break;
                    case 'u':
                        if (!parse_unicode_escape(str)) return false;
                        continue;
                    default:
                        return fail("非法转义");
                }
                ++pos;
                continue;
            }
            if (c < 0x20) return fail("字符串内出现裸控制字符");
            // 其余字节原样进字符串（含 UTF-8 多字节序列，此处不做再解码）。
            str.push_back(s[pos]);
            ++pos;
        }
    }

    bool parse_array(Json& out) {
        ++pos;   // 跳过 '['
        Json a = Json::arrv();
        skip_ws();
        if (peek() == ']') {
            ++pos;
            out = std::move(a);
            return true;
        }
        while (true) {
            skip_ws();
            Json v;
            if (!parse_value(v)) return false;
            a.push(std::move(v));
            skip_ws();
            if (pos >= s.size()) return fail("数组未闭合");
            if (s[pos] == ',') {
                ++pos;
                continue;
            }
            if (s[pos] == ']') {
                ++pos;
                out = std::move(a);
                return true;
            }
            return fail("数组元素间缺逗号或右括号");
        }
    }

    bool parse_object(Json& out) {
        ++pos;   // 跳过 '{'
        Json o = Json::objv();
        skip_ws();
        if (peek() == '}') {
            ++pos;
            out = std::move(o);
            return true;
        }
        while (true) {
            skip_ws();
            if (peek() != '"') return fail("对象键必须是字符串");
            Json key;
            if (!parse_string(key)) return false;
            skip_ws();
            if (pos >= s.size() || s[pos] != ':') return fail("对象键后缺冒号");
            ++pos;
            skip_ws();
            Json v;
            if (!parse_value(v)) return false;
            o.set(key.str, std::move(v));
            skip_ws();
            if (pos >= s.size()) return fail("对象未闭合");
            if (s[pos] == ',') {
                ++pos;
                continue;
            }
            if (s[pos] == '}') {
                ++pos;
                out = std::move(o);
                return true;
            }
            return fail("对象成员间缺逗号或右花括号");
        }
    }
};

bool Parser::parse_value(Json& out) {
    if (pos >= s.size()) return fail("意外结束");
    switch (s[pos]) {
        case '{':
            return parse_object(out);
        case '[':
            return parse_array(out);
        case '"':
            return parse_string(out);
        case 't':
            if (match_literal("true")) { out = Json::boolv(true); return true; }
            return fail("期望 true");
        case 'f':
            if (match_literal("false")) { out = Json::boolv(false); return true; }
            return fail("期望 false");
        case 'n':
            if (match_literal("null")) { out = Json::nullv(); return true; }
            return fail("期望 null");
        case '-':
        case '0': case '1': case '2': case '3': case '4':
        case '5': case '6': case '7': case '8': case '9':
            return parse_number(out);
        default:
            return fail("非法字符");
    }
}

}  // namespace

const Json* Json::find(const std::string& key) const {
    if (kind != kObj) return nullptr;
    for (const auto& kv : obj) {
        if (kv.first == key) return &kv.second;
    }
    return nullptr;
}

Json* Json::find(const std::string& key) {
    return const_cast<Json*>(static_cast<const Json*>(this)->find(key));
}

bool json_parse(const std::string& s, Json& out, std::string& err) {
    Parser p{s, 0, err};
    p.skip_ws();
    if (!p.parse_value(out)) return false;
    p.skip_ws();
    if (p.pos != s.size()) return p.fail("尾随内容");
    return true;
}

std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (const char ch : s) {
        const unsigned char c = static_cast<unsigned char>(ch);
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out.push_back(ch);   // UTF-8 字节原样
                }
        }
    }
    return out;
}

std::string json_dump(const Json& j) {
    switch (j.kind) {
        case Json::kNull:
            return "null";
        case Json::kBool:
            return j.b ? "true" : "false";
        case Json::kNum: {
            // 整数值且 |x|<9e15 走整数字面量，否则 %.9g（两条路径都被测试钉住）。
            const double x = j.num;
            if (x == std::trunc(x) && std::abs(x) < 9e15) {
                char buf[32];
                std::snprintf(buf, sizeof(buf), "%lld", static_cast<long long>(x));
                return buf;
            }
            char buf[32];
            std::snprintf(buf, sizeof(buf), "%.9g", x);
            return buf;
        }
        case Json::kStr: {
            std::string out;
            out.reserve(j.str.size() + 2);
            out.push_back('"');
            out += json_escape(j.str);
            out.push_back('"');
            return out;
        }
        case Json::kArr: {
            std::string out = "[";
            for (std::size_t i = 0; i < j.arr.size(); ++i) {
                if (i != 0) out.push_back(',');
                out += json_dump(j.arr[i]);
            }
            out.push_back(']');
            return out;
        }
        case Json::kObj: {
            std::string out = "{";
            for (std::size_t i = 0; i < j.obj.size(); ++i) {
                if (i != 0) out.push_back(',');
                out.push_back('"');
                out += json_escape(j.obj[i].first);
                out += "\":";
                out += json_dump(j.obj[i].second);
            }
            out.push_back('}');
            return out;
        }
    }
    return "null";   // 枚举已全覆盖，仅为满足返回值要求
}

}  // namespace qinfer::serve