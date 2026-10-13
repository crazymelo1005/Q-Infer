#include "serve/toolcall.hpp"

#include <algorithm>
#include <cctype>
#include <utility>

namespace qinfer::serve {

namespace {

constexpr const char* kCallStart = "<tool_call>";
constexpr const char* kCallEnd = "</tool_call>";
constexpr const char* kFuncStart = "<function=";
constexpr const char* kFuncEnd = "</function>";
constexpr const char* kParamStart = "<parameter=";
constexpr const char* kParamEnd = "</parameter>";

constexpr std::size_t kCallStartLen = 11;   // strlen("<tool_call>")
constexpr std::size_t kCallEndLen = 12;     // strlen("</tool_call>")
constexpr std::size_t kFuncStartLen = 10;   // strlen("<function=")
constexpr std::size_t kFuncEndLen = 11;     // strlen("</function>")
constexpr std::size_t kParamStartLen = 11;  // strlen("<parameter=")
constexpr std::size_t kParamEndLen = 12;    // strlen("</parameter>")

// 首尾剥空白（对拍 Python str.strip()：空白含空格/制表/换行/回车）。
std::string trim(const std::string& s) {
    std::size_t b = 0;
    while (b < s.size() && std::isspace(static_cast<unsigned char>(s[b]))) ++b;
    std::size_t e = s.size();
    while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) --e;
    return s.substr(b, e - b);
}

// 前导空白计数（对拍 Python str.lstrip() 的去除量）。
std::size_t lstrip_count(const std::string& s) {
    std::size_t i = 0;
    while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i]))) ++i;
    return i;
}

bool starts_with(const std::string& s, const char* p) { return s.rfind(p, 0) == 0; }

// s 是 p 的前缀（含 s 空）：留半截标记等齐所用。
bool is_prefix_of(const std::string& s, const char* p) { return std::string(p).rfind(s, 0) == 0; }

// 参数值在 text 里的结束偏移（对拍 frontend.py param_end）：第一个 `</parameter>` 且其后
// （剥空白）跟着 `<parameter=` 或 `</function>`——值里同名标签不算（#210）。后随未到齐
// （空/半截）：final=true（整块解析）当结束，否则 -2。找不到 → -1。
std::ptrdiff_t param_end(const std::string& text, bool final = false) {
    std::size_t at = text.find(kParamEnd);
    while (at != std::string::npos) {
        std::size_t i = at + kParamEndLen;
        while (i < text.size() && std::isspace(static_cast<unsigned char>(text[i]))) ++i;
        const std::string after = text.substr(i);
        if (starts_with(after, kParamStart) || starts_with(after, kFuncEnd))
            return static_cast<std::ptrdiff_t>(at);
        if (after.empty() || is_prefix_of(after, kParamStart) || is_prefix_of(after, kFuncEnd))
            return final ? static_cast<std::ptrdiff_t>(at) : -2;
        at = text.find(kParamEnd, at + 1);
    }
    return -1;
}

// 调用体在 text 里的结束偏移（对拍 frontend.py call_end）：返回 `</tool_call>` 开始处，
// -1 = 未齐。体按 <function=…> 开头并在参数链里步进，值里的 </tool_call> 字面不截块
// （#210）；不在调用格式的体在第一处 `</tool_call>` 收尾（散文里的例子不算调用）。
std::ptrdiff_t call_end(const std::string& text) {
    const std::size_t pos = lstrip_count(text);
    const std::string s = text.substr(pos);
    if (!starts_with(s, kFuncStart)) {
        if (is_prefix_of(s, kFuncStart)) return -1;   // 半截 <function=
        const std::size_t at = text.find(kCallEnd);
        return at == std::string::npos ? -1 : static_cast<std::ptrdiff_t>(at);
    }
    std::size_t gt = text.find('>', pos);
    if (gt == std::string::npos) return -1;
    std::size_t p = gt + 1;
    for (;;) {
        const std::string rest = text.substr(p);
        const std::size_t strip = lstrip_count(rest);
        p += strip;
        const std::string rs = rest.substr(strip);
        if (starts_with(rs, kParamStart)) {
            gt = text.find('>', p);
            if (gt == std::string::npos) return -1;
            const std::ptrdiff_t end = param_end(text.substr(gt + 1));
            if (end < 0) return -1;
            p = gt + 1 + static_cast<std::size_t>(end) + kParamEndLen;
        } else if (starts_with(rs, kFuncEnd)) {
            const std::string rrest = text.substr(p + kFuncEndLen);
            const std::size_t rstrip = lstrip_count(rrest);
            const std::string rs2 = rrest.substr(rstrip);
            if (starts_with(rs2, kCallEnd))
                return static_cast<std::ptrdiff_t>(p + kFuncEndLen + rstrip);
            if (is_prefix_of(rs2, kCallEnd)) return -1;   // 半截 </tool_call>
            const std::size_t at = text.find(kCallEnd, p);
            return at == std::string::npos ? -1 : static_cast<std::ptrdiff_t>(at);
        } else if (rs.empty() || is_prefix_of(rs, kParamStart) || is_prefix_of(rs, kFuncEnd)) {
            return -1;
        } else {
            const std::size_t at = text.find(kCallEnd, p);
            return at == std::string::npos ? -1 : static_cast<std::ptrdiff_t>(at);
        }
    }
}

// 全空白串？
bool is_blank(const std::string& s) {
    for (unsigned char c : s)
        if (!std::isspace(c)) return false;
    return true;
}

}  // namespace

bool normalize_tools(const Json& value, bool anthropic, std::vector<Json>& out, std::string& err) {
    if (value.is_null() || (value.is_arr() && value.arr.empty()) ||
        (value.is_str() && value.str.empty()) || (value.is_obj() && value.obj.empty()))
        return true;
    if (!value.is_arr()) {
        err = "tools 必须是工具对象的数组（OpenAI 形状 {type, function} 或裸 {name}）";
        return false;
    }
    for (std::size_t i = 0; i < value.arr.size(); ++i) {
        const Json& t = value.arr[i];
        if (!t.is_obj()) {
            err = "tools[" + std::to_string(i) + "] 不是对象";
            return false;
        }
        const Json* fn = &t;
        if (!anthropic) {
            const Json* ty = t.find("type");
            if (ty != nullptr && ty->is_str() && ty->str == "function") {
                fn = t.find("function");   // {"type":"function","function":{...}}：取 function 体
                if (fn == nullptr) fn = &t;
            }
        }
        if (!fn->is_obj()) {
            err = "tools[" + std::to_string(i) + "] 的 function 不是对象";
            return false;
        }
        const Json* nm = fn->find("name");
        if (nm == nullptr || !nm->is_str() || nm->str.empty()) {
            err = "tools[" + std::to_string(i) + "] 缺名字";
            return false;
        }
        const char* key = anthropic ? "input_schema" : "parameters";
        const Json* schema = fn->find(key);
        if (schema != nullptr && !schema->is_obj()) {
            err = "tools[" + std::to_string(i) + "] 的 " + key + " 必须是对象（参数 JSON schema 或缺省）";
            return false;
        }
        Json o = *fn;   // 解包后的工具本体
        if (anthropic) {
            // input_schema 改名 parameters、旧键不残留；description 缺省补空串
            const Json* is = o.find("input_schema");
            if (is != nullptr) {
                o.obj.erase(std::remove_if(o.obj.begin(), o.obj.end(),
                                           [](const std::pair<std::string, Json>& kv) {
                                               return kv.first == "input_schema";
                                           }),
                            o.obj.end());
                o.set("parameters", *is);
            }
            if (o.find("description") == nullptr) o.set("description", Json::strv(""));
        }
        out.push_back(std::move(o));
    }
    return true;
}

Json tool_arguments(const Json* raw) {
    if (raw == nullptr || raw->is_null()) return Json::objv();
    if (raw->is_obj()) return *raw;
    if (raw->is_str()) {
        if (is_blank(raw->str)) return Json::objv();
        Json j;
        std::string jerr;
        if (json_parse(raw->str, j, jerr) && j.is_obj()) return j;
        Json w = Json::objv();
        w.set("arguments", *raw);
        return w;
    }
    Json w = Json::objv();
    w.set("arguments", Json::strv(json_dump(*raw)));
    return w;
}

bool parse_openai_tool_calls(const Json& value, std::vector<Json>& out, std::string& err) {
    if (!value.is_arr()) {
        err = "tool_calls 必须是数组";
        return false;
    }
    for (const Json& c : value.arr) {
        const Json* fn = c.find("function");
        if (fn == nullptr) fn = &c;   // 无 function 包装：元素本身
        if (!fn->is_obj()) {
            err = "tool_calls 元素缺 function 对象";
            return false;
        }
        const Json* nm = fn->find("name");
        if (nm == nullptr || !nm->is_str() || nm->str.empty()) {
            err = "tool_calls 元素缺名字";
            return false;
        }
        const Json* raw_args = fn->find("arguments");
        Json f = Json::objv();
        f.set("name", *nm);
        f.set("arguments", tool_arguments(raw_args));
        Json o = Json::objv();
        o.set("function", std::move(f));
        out.push_back(std::move(o));
    }
    return true;
}

bool parse_tool_block(const std::string& raw, const std::vector<Json>& tools, ToolCall& out,
                      std::string& err) {
    std::string body = trim(raw);
    if (!starts_with(body, kCallStart)) {
        err = "工具块不以 <tool_call> 开头（不在调用格式）";
        return false;
    }
    body = trim(body.substr(kCallStartLen));
    const std::size_t ng = body.find('>');
    if (!starts_with(body, kFuncStart) || ng == std::string::npos) {
        err = "工具块缺 <function=…>";
        return false;
    }
    std::string name = trim(body.substr(kFuncStartLen, ng - kFuncStartLen));
    if (name.empty()) {
        err = "工具块函数名为空";
        return false;
    }
    // schema 查表：按名字找工具，取 parameters.properties（缺省空——不声明就 JSON 自动）
    const Json* props = nullptr;
    for (const Json& t : tools) {
        const Json* tn = t.find("name");
        if (tn == nullptr || !tn->is_str() || tn->str != name) continue;
        const Json* params = t.find("parameters");
        if (params != nullptr && params->is_obj()) {
            const Json* p = params->find("properties");
            if (p != nullptr && p->is_obj()) props = p;
        }
        break;
    }
    Json args = Json::objv();
    std::string rest = body.substr(ng + 1);
    for (;;) {
        const std::size_t at = rest.find(kParamStart);
        if (at == std::string::npos) break;
        rest = rest.substr(at + kParamStartLen);
        const std::size_t gt = rest.find('>');
        if (gt == std::string::npos) {
            err = "工具块参数 <parameter= 后无 >（截断）";
            return false;
        }
        const std::string pname = rest.substr(0, gt);   // 名字原样（对拍 Strata，不剥空白）
        rest = rest.substr(gt + 1);
        const std::ptrdiff_t end = param_end(rest, /*final=*/true);
        std::string value =
            end >= 0 ? rest.substr(0, static_cast<std::size_t>(end)) : rest;
        rest = end >= 0 ? rest.substr(static_cast<std::size_t>(end) + kParamEndLen)
                        : std::string();
        if (!value.empty() && value.front() == '\n') value.erase(value.begin());
        if (!value.empty() && value.back() == '\n') value.pop_back();
        bool keep_raw = false;
        if (props != nullptr) {
            const Json* pv = props->find(pname);
            if (pv != nullptr && pv->is_obj()) {
                const Json* ty = pv->find("type");
                keep_raw = ty != nullptr && ty->is_str() && ty->str == "string";
            }
        }
        if (keep_raw) {
            args.set(pname, Json::strv(value));
        } else {
            Json j;
            std::string jerr;
            if (json_parse(value, j, jerr))
                args.set(pname, std::move(j));
            else
                args.set(pname, Json::strv(value));
        }
    }
    out.name = std::move(name);
    out.arguments = std::move(args);
    return true;
}

void ToolCallStreamer::feed(const std::string& piece, std::vector<Event>& out) {
    buf_ += piece;
    for (;;) {
        if (in_block_) {
            // 调用状态：身体（剥掉 <tool_call> 前缀）在 call_end 眼里步进
            const std::string walk =
                buf_.size() >= kCallStartLen ? buf_.substr(kCallStartLen) : std::string();
            const std::ptrdiff_t i = call_end(walk);
            if (i < 0) return;
            const std::size_t ev_end =
                kCallStartLen + static_cast<std::size_t>(i) + kCallEndLen;
            out.push_back(Event{Kind::kBlock, buf_.substr(0, ev_end)});
            buf_ = buf_.substr(ev_end);
            in_block_ = false;
            lead_ = true;
            continue;
        }
        if (lead_) {
            std::size_t j = 0;
            while (j < buf_.size() && buf_[j] == '\n') ++j;
            if (j == buf_.size()) {
                buf_.clear();
                return;
            }
            if (j > 0) buf_.erase(0, j);
            lead_ = false;
        }
        const std::size_t i = buf_.find(kCallStart);
        if (i != std::string::npos) {
            const std::string before = buf_.substr(0, i);
            if (!is_blank(before)) {
                std::size_t e = before.size();
                while (e > 0 && before[e - 1] == '\n') --e;
                if (e > 0) out.push_back(Event{Kind::kText, before.substr(0, e)});
            }
            buf_ = buf_.substr(i);
            in_block_ = true;
            continue;
        }
        // 无完整标记：留半截 <tool_call> 与它前面的换行（分片=整段不变量）
        std::size_t hold = 0;
        {
            std::size_t n = std::min<std::size_t>(buf_.size(), kCallStartLen - 1);
            while (n > 0) {
                if (std::string(kCallStart).rfind(buf_.substr(buf_.size() - n), 0) == 0) break;
                --n;
            }
            hold = n;
        }
        std::size_t j = buf_.size() - hold;
        while (j > 0 && buf_[j - 1] == '\n') --j;
        if (j > 0) {
            out.push_back(Event{Kind::kText, buf_.substr(0, j)});
            buf_.erase(0, j);
        }
        return;
    }
}

void ToolCallStreamer::finish(std::vector<Event>& out) {
    if (in_block_) {
        out.push_back(Event{Kind::kText, std::move(buf_)});   // 未闭合调用按原样文本释放
        buf_.clear();
        in_block_ = false;
    } else if (!buf_.empty()) {
        out.push_back(Event{Kind::kText, std::move(buf_)});
        buf_.clear();
    }
    lead_ = false;
}

void split_generation(const std::string& text, const std::vector<Json>& tools,
                      std::vector<GenEvent>& out) {
    ToolCallStreamer st;
    std::vector<ToolCallStreamer::Event> evs;
    st.feed(text, evs);
    st.finish(evs);
    std::string pending;
    const auto flush = [&out, &pending]() {
        if (pending.empty()) return;
        out.push_back(GenEvent{std::move(pending), std::nullopt});
        pending.clear();
    };
    for (const auto& e : evs) {
        if (e.kind == ToolCallStreamer::Kind::kText) {
            pending += e.text;
            continue;
        }
        ToolCall tc;
        std::string err;
        if (parse_tool_block(e.text, tools, tc, err)) {
            flush();
            out.push_back(GenEvent{"", std::move(tc)});
        } else {
            pending += e.text;   // 幻觉出的坏标记：并回文本，不当错误毁掉请求
        }
    }
    flush();
}

Json openai_tool_call(const std::string& id, const std::string& name, const Json& arguments) {
    Json fn = Json::objv();
    fn.set("name", Json::strv(name));
    fn.set("arguments", Json::strv(json_dump(arguments)));
    Json c = Json::objv();
    c.set("id", Json::strv(id));
    c.set("type", Json::strv("function"));
    c.set("function", std::move(fn));
    return c;
}

Json anthropic_tool_use(const std::string& id, const std::string& name, const Json& arguments) {
    Json u = Json::objv();
    u.set("type", Json::strv("tool_use"));
    u.set("id", Json::strv(id));
    u.set("name", Json::strv(name));
    u.set("input", arguments);
    return u;
}

}  // namespace qinfer::serve