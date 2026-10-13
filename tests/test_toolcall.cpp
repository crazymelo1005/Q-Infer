// 工具调用编解码（M1 工具回合）：tools 参数归一、历史 tool_calls 解析、生成文本里的
// <tool_call>…</tool_call> 块切分（ToolCallStreamer）与解析（parse_tool_block）、
// OpenAI/Anthropic 响应形状。语义以 Strata 的 serve/frontend.py（_tool_list / tool_arguments /
// parse_tool_call / param_end / OutputParser）为对拍。
#include "serve/toolcall.hpp"

#include <cstdint>
#include <string>
#include <vector>

#include "check.hpp"

using namespace qinfer::serve;

// 工具定义构建：{name, parameters: {type: object, properties: {名: {type}}}}。
static Json tool_def(const char* name, const std::vector<std::pair<const char*, const char*>>& typed) {
    Json t = Json::objv();
    t.set("name", Json::strv(name));
    Json params = Json::objv();
    params.set("type", Json::strv("object"));
    Json props = Json::objv();
    for (const auto& [p, ty] : typed) {
        Json v = Json::objv();
        v.set("type", Json::strv(ty));
        props.set(p, std::move(v));
    }
    params.set("properties", std::move(props));
    t.set("parameters", std::move(params));
    return t;
}

static std::vector<ToolCallStreamer::Event> feed_all(ToolCallStreamer& st,
                                                     const std::vector<std::string>& pieces) {
    std::vector<ToolCallStreamer::Event> evs;
    for (const auto& p : pieces) st.feed(p, evs);
    return evs;
}

static const Json* arg(const ToolCall& c, const char* key) { return c.arguments.find(key); }

int main() {
    // === normalize_tools：OpenAI 形状解包 ===
    {
        Json tools = Json::arrv();
        Json t = Json::objv();
        t.set("type", Json::strv("function"));
        Json fn = Json::objv();
        fn.set("name", Json::strv("get_weather"));
        Json params = Json::objv();
        params.set("type", Json::strv("object"));
        fn.set("parameters", std::move(params));
        t.set("function", std::move(fn));
        tools.push(std::move(t));
        std::vector<Json> out;
        std::string err;
        CHECK(normalize_tools(tools, /*anthropic=*/false, out, err));
        CHECK(out.size() == 1);
        const Json* n = out[0].find("name");
        CHECK(n != nullptr && n->is_str() && n->str == "get_weather");
        CHECK(out[0].find("parameters") != nullptr && out[0].find("parameters")->is_obj());
        std::printf("tools 归一 OpenAI：解包 function 体、name/parameters 保留\n");
    }
    // === normalize_tools：裸元素（无 type）原样；空值 → 空列表 ===
    {
        Json tools = Json::arrv();
        Json bare = Json::objv();
        bare.set("name", Json::strv("f"));
        tools.push(std::move(bare));
        std::vector<Json> out;
        std::string err;
        CHECK(normalize_tools(tools, false, out, err));
        CHECK(out.size() == 1 && out[0].find("name")->str == "f");
        Json empty = Json::arrv();
        out.clear();
        CHECK(normalize_tools(empty, false, out, err));
        CHECK(out.empty());
        std::printf("tools 归一：裸元素原样、空列表空结果\n");
    }
    // === normalize_tools：非法形状 → false + err ===
    {
        std::vector<Json> out;
        std::string err;
        Json auto_v = Json::strv("auto");
        CHECK(!normalize_tools(auto_v, false, out, err) && !err.empty());          // 字符串不是数组
        Json bad = Json::arrv();
        Json no_name = Json::objv();
        no_name.set("name", Json::strv(""));
        bad.push(no_name);
        CHECK(!normalize_tools(bad, false, out, err) && !err.empty());             // 空名字
        Json bad2 = Json::arrv();
        Json nonobj = Json::objv();
        nonobj.set("type", Json::strv("function"));
        nonobj.set("function", Json::strv("x"));
        bad2.push(nonobj);
        CHECK(!normalize_tools(bad2, false, out, err) && !err.empty());            // function 非对象
        Json bad3 = Json::arrv();
        Json sch = Json::objv();
        sch.set("name", Json::strv("f"));
        sch.set("parameters", Json::strv("obj"));
        bad3.push(sch);
        CHECK(!normalize_tools(bad3, false, out, err) && !err.empty());            // parameters 非对象
        std::printf("tools 归一：非法形状（auto/空名/非对象/坏 schema）→ 报错\n");
    }
    // === normalize_tools：Anthropic 形状 input_schema → parameters，description 缺省空串 ===
    {
        Json tools = Json::arrv();
        Json t = Json::objv();
        t.set("name", Json::strv("f"));
        Json sch = Json::objv();
        sch.set("type", Json::strv("object"));
        t.set("input_schema", std::move(sch));
        tools.push(std::move(t));
        std::vector<Json> out;
        std::string err;
        CHECK(normalize_tools(tools, true, out, err));
        CHECK(out.size() == 1);
        CHECK(out[0].find("name")->str == "f");
        const Json* d = out[0].find("description");
        CHECK(d != nullptr && d->is_str() && d->str.empty());
        CHECK(out[0].find("parameters") != nullptr && out[0].find("parameters")->is_obj());
        CHECK(out[0].find("input_schema") == nullptr);                             // 改名后原键不残留
        std::printf("tools 归一 Anthropic：input_schema 改名 parameters、description 缺省空\n");
    }
    // === tool_arguments：空/对象/可解析串/原文串/其他类型 ===
    {
        Json r = tool_arguments(nullptr);
        CHECK(r.is_obj() && r.obj.empty());
        Json ws = Json::strv("   ");
        r = tool_arguments(&ws);
        CHECK(r.is_obj() && r.obj.empty());
        Json obj = Json::objv();
        obj.set("city", Json::strv("北京"));
        r = tool_arguments(&obj);
        CHECK(r.obj.size() == 1 && r.find("city")->str == "北京");
        Json js = Json::strv(R"({"a": 1})");
        r = tool_arguments(&js);
        CHECK(r.is_obj() && r.obj.size() == 1 && r.find("a")->is_num() && r.find("a")->num == 1.0);
        Json raw = Json::strv("nope");
        r = tool_arguments(&raw);
        CHECK(r.is_obj() && r.obj.size() == 1 && r.find("arguments")->is_str() && r.find("arguments")->str == "nope");
        Json arr = Json::strv("[1,2]");            // 解析成功但非对象 → 原文入 arguments
        r = tool_arguments(&arr);
        CHECK(r.obj.size() == 1 && r.find("arguments")->is_str() && r.find("arguments")->str == "[1,2]");
        Json num = Json::numv(3.0);                // 非字符串类型 → 序列化入 arguments
        r = tool_arguments(&num);
        CHECK(r.obj.size() == 1 && r.find("arguments")->is_str() && r.find("arguments")->str == "3");
        std::printf("tool_arguments：空/对象/解析/原文/其他五路都对\n");
    }
    // === parse_openai_tool_calls：历史 assistant 消息的 tool_calls ===
    {
        Json arr = Json::arrv();
        Json c = Json::objv();
        Json fn = Json::objv();
        fn.set("name", Json::strv("f"));
        fn.set("arguments", Json::strv(R"({"x": 1})"));
        c.set("function", std::move(fn));
        arr.push(std::move(c));
        std::vector<Json> out;
        std::string err;
        CHECK(parse_openai_tool_calls(arr, out, err));
        CHECK(out.size() == 1);
        const Json* f2 = out[0].find("function");
        CHECK(f2 != nullptr && f2->find("name")->str == "f");
        CHECK(f2->find("arguments")->is_obj() && f2->find("arguments")->find("x")->num == 1.0);
        Json bare = Json::arrv();                  // 无 function 包装 → 元素本身
        Json b = Json::objv();
        b.set("name", Json::strv("g"));
        b.set("arguments", Json::strv("bad json"));
        bare.push(std::move(b));
        out.clear();
        CHECK(parse_openai_tool_calls(bare, out, err));
        CHECK(out[0].find("function")->find("arguments")->find("arguments")->str == "bad json");
        Json notarr = Json::objv();
        CHECK(!parse_openai_tool_calls(notarr, out, err) && !err.empty());
        Json nname = Json::arrv();                 // 缺名字 → 报错
        nname.push(Json::objv());
        CHECK(!parse_openai_tool_calls(nname, out, err) && !err.empty());
        std::printf("parse_openai_tool_calls：包装/裸元素/非法输入全录\n");
    }
    // === parse_tool_block：基本块 + 声明 string 参数原文 ===
    {
        std::vector<Json> tools = {tool_def("get_weather", {{"city", "string"}})};
        const std::string raw =
            "<tool_call>\n<function=get_weather>\n<parameter=city>\n北京\n</parameter>\n"
            "</function>\n</tool_call>";
        ToolCall tc;
        std::string err;
        CHECK(parse_tool_block(raw, tools, tc, err));
        CHECK(tc.name == "get_weather");
        const Json* v = arg(tc, "city");
        CHECK(v != nullptr && v->is_str() && v->str == "北京");
        std::printf("parse_tool_block：基本块、声明 string 原样\n");
    }
    // === parse_tool_block：单侧换行剥一个；未声明参数走 JSON 自动 ===
    {
        std::vector<Json> tools = {tool_def("f", {{"p", "number"}})};
        const std::string raw =
            "<tool_call>\n<function=f>\n<parameter=p>\n\n42\n\n</parameter>\n</function>\n</tool_call>";
        ToolCall tc;
        std::string err;
        CHECK(parse_tool_block(raw, tools, tc, err));          // 值 "\n42\n"：各剥一个 → "42"
        CHECK(arg(tc, "p") != nullptr && arg(tc, "p")->is_num() && arg(tc, "p")->num == 42.0);
        const std::string raw2 =
            "<tool_call>\n<function=f>\n<parameter=q>\n3.5\n</parameter>\n</function>\n</tool_call>";
        CHECK(parse_tool_block(raw2, tools, tc, err));
        CHECK(arg(tc, "q")->is_num() && arg(tc, "q")->num == 3.5);
        const std::string raw3 =
            "<tool_call>\n<function=f>\n<parameter=r>\nnope\n</parameter>\n</function>\n</tool_call>";
        CHECK(parse_tool_block(raw3, tools, tc, err));
        CHECK(arg(tc, "r")->is_str() && arg(tc, "r")->str == "nope");   // 解析失败 → 原文
        std::printf("parse_tool_block：换行剥一个、JSON 自动（数/浮点/失败原文）\n");
    }
    // === parse_tool_block：多参数；函数名带空格宽容 ===
    {
        std::vector<Json> tools = {tool_def("f", {{"a", "string"}, {"b", "number"}})};
        const std::string raw =
            "<tool_call>\n<function= f >\n<parameter=a>\nx\n</parameter>\n<parameter=b>\n7\n</parameter>\n"
            "</function>\n</tool_call>";
        ToolCall tc;
        std::string err;
        CHECK(parse_tool_block(raw, tools, tc, err));
        CHECK(tc.name == "f");
        CHECK(arg(tc, "a")->str == "x" && arg(tc, "b")->num == 7.0);
        std::printf("parse_tool_block：多参数、<function= 后空格宽容\n");
    }
    // === parse_tool_block：值与字面重名标签（</parameter> 在值里） ===
    {
        std::vector<Json> tools = {tool_def("f", {})};
        const std::string raw =
            "<tool_call>\n<function=f>\n<parameter=p>\na</parameter>\nb\n</parameter>\n</function>\n</tool_call>";
        ToolCall tc;
        std::string err;
        CHECK(parse_tool_block(raw, tools, tc, err));
        CHECK(arg(tc, "p")->is_str() && arg(tc, "p")->str == "a</parameter>\nb");
        std::printf("parse_tool_block：值里的 </parameter> 字面不进参数边界\n");
    }
    // === parse_tool_block：值里含 </tool_call> 字面（收尾取最后一个） ===
    {
        std::vector<Json> tools = {tool_def("f", {})};
        const std::string raw =
            "<tool_call>\n<function=f>\n<parameter=p>\n</tool_call>\n</parameter>\n</function>\n</tool_call>";
        ToolCall tc;
        std::string err;
        CHECK(parse_tool_block(raw, tools, tc, err));
        CHECK(arg(tc, "p")->is_str() && arg(tc, "p")->str == "</tool_call>");
        std::printf("parse_tool_block：值里的 </tool_call> 字面不提前截断\n");
    }
    // === parse_tool_block：残缺块 → 报错 ===
    {
        std::vector<Json> tools = {tool_def("f", {})};
        ToolCall tc;
        std::string err;
        CHECK(!parse_tool_block("随便说点什么</tool_call>", tools, tc, err) && !err.empty());
        CHECK(!parse_tool_block("<tool_call>\n<parameter=p\n42\n</parameter>\n</function>\n</tool_call>",
                                tools, tc, err) && !err.empty());     // <parameter= 后无 >（截断）
        CHECK(!parse_tool_block("<tool_call>\n<function=>\n</function>\n</tool_call>",
                                tools, tc, err) && !err.empty());     // 空函数名
        std::printf("parse_tool_block：无 <tool_call>/截断 <parameter=/空函数名 → 报错\n");
    }
    // === ToolCallStreamer：标记跨增量切碎，块整体在收尾后出现 ===
    {
        ToolCallStreamer st;
        auto evs = feed_all(st, {"<tool_", "call>\n<function=f>\n<param", "eter=p>\n42\n</parameter>\n</function>\n</tool_",
                                 "call>\n"});
        CHECK(evs.size() == 1 && evs[0].kind == ToolCallStreamer::Kind::kBlock);
        CHECK(evs[0].text ==
              "<tool_call>\n<function=f>\n<parameter=p>\n42\n</parameter>\n</function>\n</tool_call>");
        CHECK(!st.in_block());
        std::printf("ToolCallStreamer：切碎标记整块拼回、收尾换行不泄漏\n");
    }
    // === ToolCallStreamer：调用前/后的换行按 Strata 丢弃 ===
    {
        ToolCallStreamer st;
        auto evs = feed_all(st, {"hello\n<tool_call>\n<function=f>\n<parameter=p>\n1\n</parameter>\n</function>\n"
                                 "</tool_call>\nworld"});
        CHECK(evs.size() == 3);
        CHECK(evs[0].kind == ToolCallStreamer::Kind::kText && evs[0].text == "hello");
        CHECK(evs[1].kind == ToolCallStreamer::Kind::kBlock);
        CHECK(evs[2].kind == ToolCallStreamer::Kind::kText && evs[2].text == "world");
        std::printf("ToolCallStreamer：调用前/后换行丢弃、内容分三段\n");
    }
    // === ToolCallStreamer：调用前的纯空白段整体丢弃 ===
    {
        ToolCallStreamer st;
        auto evs = feed_all(st, {"  \n<tool_call>\n<function=f>\n<parameter=p>\n1\n</parameter>\n</function>\n"
                                 "</tool_call>"});
        CHECK(evs.size() == 1 && evs[0].kind == ToolCallStreamer::Kind::kBlock);
        std::printf("ToolCallStreamer：调用前纯空白不输出\n");
    }
    // === ToolCallStreamer：同批多个调用按序出；内容事件不含半个标记 ===
    {
        ToolCallStreamer st;
        auto evs = feed_all(st, {"a<tool_call>1</tool_call>b<tool_call>2</tool_call>c"});
        CHECK(evs.size() == 5);
        CHECK(evs[0].kind == ToolCallStreamer::Kind::kText && evs[0].text == "a");
        CHECK(evs[1].kind == ToolCallStreamer::Kind::kBlock && evs[1].text == "<tool_call>1</tool_call>");
        CHECK(evs[2].kind == ToolCallStreamer::Kind::kText && evs[2].text == "b");
        CHECK(evs[3].kind == ToolCallStreamer::Kind::kBlock && evs[3].text == "<tool_call>2</tool_call>");
        CHECK(evs[4].kind == ToolCallStreamer::Kind::kText && evs[4].text == "c");
        for (const auto& e : evs)
            if (e.kind == ToolCallStreamer::Kind::kText)
                CHECK(e.text.find("<tool_") == std::string::npos);
        std::printf("ToolCallStreamer：多个调用按序、内容无半截标记\n");
    }
    // === ToolCallStreamer：分片喂与整段喂事件序一致（流式=整体不变量） ===
    {
        const std::string full =
            "think a<b><tool_call>\n<function=f>\n<parameter=p>\n9\n</parameter>\n</function>\n</tool_call>"
            "\nzz<tool_call>\n<function=g>\n<parameter=q>\n8\n</parameter>\n</function>\n</tool_call>tail";
        ToolCallStreamer s1;
        std::vector<ToolCallStreamer::Event> whole;
        s1.feed(full, whole);
        s1.finish(whole);
        ToolCallStreamer s2;
        const std::vector<std::string> cuts = {"think a<b><tool_", "call>\n<function=f>\n<parameter=p>\n9\n</par",
                                               "ameter>\n</function>\n</tool_call>", "\nzz<tool_call>\n<function=g>",
                                               "\n<parameter=q>\n8\n</parameter>\n</function>\n</tool_call>tail"};
        std::vector<ToolCallStreamer::Event> split;
        for (const auto& c : cuts) s2.feed(c, split);
        s2.finish(split);
        CHECK(split.size() == whole.size());
        for (std::size_t i = 0; i < split.size(); ++i) {
            CHECK(split[i].kind == whole[i].kind);
            CHECK(split[i].text == whole[i].text);
        }
        std::printf("ToolCallStreamer：分片与整段事件逐条一致（流式=整体）\n");
    }
    // === ToolCallStreamer：值里的 </tool_call> 字面不提前收尾（call_end 沿参数走） ===
    {
        ToolCallStreamer st;
        auto evs = feed_all(st, {"<tool_call>\n<function=f>\n<parameter=p>\n</tool_call>\n</parameter>\n"
                                 "</function>\n</tool_call>"});
        CHECK(evs.size() == 1 && evs[0].kind == ToolCallStreamer::Kind::kBlock);
        CHECK(evs[0].text ==
              "<tool_call>\n<function=f>\n<parameter=p>\n</tool_call>\n</parameter>\n</function>\n</tool_call>");
        std::printf("ToolCallStreamer：块扫描沿参数走，值里 </tool_call> 字面不截块\n");
    }
    // === ToolCallStreamer：生成中断在调用里 → 整段作文本（#211） ===
    {
        ToolCallStreamer st;
        auto evs = feed_all(st, {"done<tool_call>\n<function=f>\n<param"});
        CHECK(evs.size() == 1 && evs[0].kind == ToolCallStreamer::Kind::kText && evs[0].text == "done");
        st.finish(evs);
        CHECK(evs.size() == 2 && evs[1].kind == ToolCallStreamer::Kind::kText);
        CHECK(evs[1].text == "<tool_call>\n<function=f>\n<param");
        CHECK(!st.in_block());
        ToolCallStreamer s2;
        auto evs2 = feed_all(s2, {"abc"});
        s2.finish(evs2);
        CHECK(evs2.size() == 1 && evs2[0].kind == ToolCallStreamer::Kind::kText && evs2[0].text == "abc");
        std::printf("ToolCallStreamer：未闭合块 finish 时按原样文本释放\n");
    }
    // === split_generation：文本/调用/文本顺序保留 ===
    {
        std::vector<Json> tools = {tool_def("f", {{"p", "number"}})};
        std::vector<GenEvent> gevs;
        split_generation(
            "前\n<tool_call>\n<function=f>\n<parameter=p>\n42\n</parameter>\n</function>\n</tool_call>\n后",
            tools, gevs);
        CHECK(gevs.size() == 3);
        CHECK(!gevs[0].call.has_value() && gevs[0].text == "前");
        CHECK(gevs[1].call.has_value() && gevs[1].call->name == "f");
        CHECK(arg(*gevs[1].call, "p") != nullptr && arg(*gevs[1].call, "p")->num == 42.0);
        CHECK(!gevs[2].call.has_value() && gevs[2].text == "后");
        // 失败块并入文本（模型幻觉标记不当错误毁掉整次请求）
        std::vector<GenEvent> gevs2;
        split_generation("x<tool_call>没闭合", tools, gevs2);
        CHECK(gevs2.size() == 1 && !gevs2[0].call.has_value());
        CHECK(gevs2[0].text == "x<tool_call>没闭合");
        std::printf("split_generation：事件序保留、失败块回退文本\n");
    }
    // === 响应形状：OpenAI 调用对象与 Anthropic tool_use 块 ===
    {
        Json args = Json::objv();
        args.set("city", Json::strv("北京"));
        Json c = openai_tool_call("call_1_0", "get_weather", args);
        CHECK(c.find("id")->str == "call_1_0");
        CHECK(c.find("type")->str == "function");
        const Json* fn = c.find("function");
        CHECK(fn != nullptr && fn->find("name")->str == "get_weather");
        CHECK(fn->find("arguments")->is_str() && fn->find("arguments")->str == R"({"city":"北京"})");
        Json u = anthropic_tool_use("call_1_0", "get_weather", args);
        CHECK(u.find("type")->str == "tool_use");
        CHECK(u.find("id")->str == "call_1_0" && u.find("name")->str == "get_weather");
        const Json* inp = u.find("input");
        CHECK(inp != nullptr && inp->is_obj() && inp->find("city")->str == "北京");
        std::printf("openai_tool_call / anthropic_tool_use：形状与字段正确\n");
    }

    std::puts("toolcall: 通过");
    return 0;
}