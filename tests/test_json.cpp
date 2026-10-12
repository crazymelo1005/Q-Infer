#include "check.hpp"
#include "serve/json.hpp"
#include <cstddef>
#include <cstdio>
#include <string>

using namespace qinfer::serve;

namespace {

// 结构相等：逐 kind 比较。对象按插入序逐成员比——序列化保序，故 dump→parse 可对拍。
bool eq(const Json& a, const Json& b) {
    if (a.kind != b.kind) return false;
    switch (a.kind) {
        case Json::kNull:
            return true;
        case Json::kBool:
            return a.b == b.b;
        case Json::kNum:
            return a.num == b.num;
        case Json::kStr:
            return a.str == b.str;
        case Json::kArr: {
            if (a.arr.size() != b.arr.size()) return false;
            for (std::size_t i = 0; i < a.arr.size(); ++i)
                if (!eq(a.arr[i], b.arr[i])) return false;
            return true;
        }
        case Json::kObj: {
            if (a.obj.size() != b.obj.size()) return false;
            for (std::size_t i = 0; i < a.obj.size(); ++i) {
                if (a.obj[i].first != b.obj[i].first) return false;
                if (!eq(a.obj[i].second, b.obj[i].second)) return false;
            }
            return true;
        }
    }
    return false;
}

// 序列化往返：dump 后能再解析，且结构相等。
bool roundtrip(const Json& x) {
    Json y;
    std::string err;
    return json_parse(json_dump(x), y, err) && eq(x, y);
}

}  // namespace

int main() {
    // 解析嵌套对象/数组：结构、类型、取值逐层核对。
    {
        const std::string s = "{"
            "\"menu\":{"
            "\"id\":5,"
            "\"items\":[{\"name\":\"a\",\"on\":true},{\"name\":\"b\",\"on\":false}],"
            "\"empty\":[]"
            "}"
            "}";
        Json j;
        std::string err;
        CHECK(json_parse(s, j, err));
        CHECK(j.is_obj());
        const Json* menu = j.find("menu");
        CHECK(menu != nullptr && menu->is_obj());
        const Json* id = menu->find("id");
        CHECK(id != nullptr && id->is_num() && id->num == 5.0);
        const Json* items = menu->find("items");
        CHECK(items != nullptr && items->is_arr() && items->arr.size() == 2);
        CHECK(items->arr[0].find("name")->str == "a");
        CHECK(items->arr[0].find("on")->b == true);
        CHECK(items->arr[1].find("name")->str == "b");
        CHECK(items->arr[1].find("on")->b == false);
        const Json* empt = menu->find("empty");
        CHECK(empt != nullptr && empt->is_arr() && empt->arr.empty());
        // 非对象或缺键 → find 返回 nullptr。
        CHECK(items->find("x") == nullptr);
        CHECK(Json::numv(1).find("x") == nullptr);
        std::printf("嵌套对象/数组：两层结构与取值正确\n");
    }
    // 字符串转义：\" \\ \/ \b \f \n \r \t 全部还原；空串、非代理 \uXXXX。
    {
        Json j;
        std::string err;
        CHECK(json_parse(R"("a\"b\\c\/d\b\f\n\r\t")", j, err));
        std::string want = "a\"b\\c/d";
        want += '\b';
        want += '\f';
        want += '\n';
        want += '\r';
        want += '\t';
        CHECK(j.is_str() && j.str == want);
        CHECK(json_parse(R"("")", j, err) && j.is_str() && j.str.empty());
        CHECK(json_parse(R"("\u0041")", j, err) && j.is_str() && j.str == "A");
        std::printf("字符串转义：八种转义逐一还原，\\uXXXX 非代理直通\n");
    }
    // \uXXXX：\u4E2D → 中；代理对 \uD83D\uDE00 → U+1F600。
    {
        Json j;
        std::string err;
        CHECK(json_parse(R"("\u4E2D")", j, err));
        CHECK(j.is_str() && j.str == "中");
        CHECK(json_parse(R"("\uD83D\uDE00")", j, err));
        CHECK(j.is_str() && j.str == "\xF0\x9F\x98\x80");
        CHECK(j.str.size() == 4);
        std::printf("\\u4E2D → 中；代理对 → U+1F600\n");
    }
    // 数字：0、-3.5、1e10、1.5E-2 与整数形态。
    {
        Json j;
        std::string err;
        CHECK(json_parse("0", j, err) && j.is_num() && j.num == 0.0);
        CHECK(json_parse("-3.5", j, err) && j.is_num() && j.num == -3.5);
        CHECK(json_parse("1e10", j, err) && j.is_num() && j.num == 1e10);
        CHECK(json_parse("1.5E-2", j, err) && j.is_num() && j.num == 0.015);
        CHECK(json_parse("42", j, err) && j.is_num() && j.num == 42.0);
        std::printf("数字：0 / -3.5 / 1e10 / 1.5E-2 / 42\n");
    }
    // 布尔与空。
    {
        Json j;
        std::string err;
        CHECK(json_parse("true", j, err) && j.kind == Json::kBool && j.b == true);
        CHECK(json_parse("false", j, err) && j.kind == Json::kBool && j.b == false);
        CHECK(json_parse("null", j, err) && j.is_null());
        std::printf("布尔与空：true / false / null\n");
    }
    // 错误：每类越界输入都必须返回假且带错误消息；消息含位置。
    {
        Json j;
        std::string err;
        CHECK(!json_parse("1 2", j, err) && !err.empty());          // 尾随垃圾
        CHECK(!json_parse("\"abc", j, err) && !err.empty());        // 未闭合字符串
        CHECK(!json_parse("{\"a\":1", j, err) && !err.empty());     // 未闭合对象
        CHECK(!json_parse("[1,2", j, err) && !err.empty());         // 未闭合数组
        CHECK(!json_parse("\"\\x\"", j, err) && !err.empty());      // 坏转义
        CHECK(!json_parse("\"\\uDE00\"", j, err) && !err.empty());  // 孤立低位代理
        CHECK(!json_parse("\"\\uD83D\"", j, err) && !err.empty());  // 高位代理缺低位
        CHECK(!json_parse("01", j, err) && !err.empty());           // 前导零：0 后留 1 → 尾随内容
        CHECK(!json_parse("", j, err) && !err.empty());
        CHECK(!json_parse("nul", j, err) && !err.empty());
        CHECK(!json_parse("{\"a\":}", j, err) && !err.empty());
        CHECK(!json_parse("[1,]", j, err) && !err.empty());
        CHECK(!json_parse("1.", j, err) && !err.empty());           // 小数部分缺数字
        CHECK(!json_parse("1e", j, err) && !err.empty());           // 指数缺数字
        CHECK(!json_parse("\"ab", j, err) && err.find("第") != std::string::npos);
        std::printf("错误：尾随/未闭合/坏转义/孤立代理/前导零等 15 类全拒\n");
    }
    // 数字序列化两路径：整数值且 |x|<9e15 → 整数字面量；否则 %.9g。
    {
        CHECK(json_dump(Json::numv(0)) == "0");
        CHECK(json_dump(Json::numv(-123456789012345.0)) == "-123456789012345");
        CHECK(json_dump(Json::numv(3.14159265)) == "3.14159265");
        CHECK(json_dump(Json::numv(1.5E-2)) == "0.015");
        std::printf("数字序列化：整数字面量与 %%.9g 两路径钉住\n");
    }
    // 紧凑序列化：空白剔除、键序保持。
    {
        Json j;
        std::string err;
        CHECK(json_parse(" { \"a\" : [ 1 , 2 ] , \"b\" : \"x\" } ", j, err));
        CHECK(json_dump(j) == R"({"a":[1,2],"b":"x"})");
        std::printf("紧凑序列化：空白剔除、键序保持\n");
    }
    // json_escape：引号/反斜杠/六个控制字符名与 <0x20 的 \u00xx，UTF-8 字节原样。
    {
        std::string s = "\"\\";
        s += '\n';
        s += '\r';
        s += '\t';
        s += '\b';
        s += '\f';
        s += '\x01';
        s += '\x1f';
        s += "中";
        const std::string e = json_escape(s);
        CHECK(e == R"(\"\\\n\r\t\b\f\u0001\u001f中)");
        CHECK(json_escape("hello world") == "hello world");
        std::printf("json_escape：控制字符出 \\u00xx、其余原样\n");
    }
    // 序列化往返：混合结构 dump 后再解析结构相等。
    {
        Json x = Json::objv();
        x.set("s", Json::strv("中\n\"x\""));
        x.set("n", Json::numv(-3.5));
        x.set("big", Json::numv(1e10));
        x.set("tiny", Json::numv(1.5E-2));
        x.set("t", Json::boolv(true));
        x.set("f", Json::boolv(false));
        x.set("z", Json::nullv());
        Json sub = Json::arrv();
        sub.push(Json::strv("a"));
        sub.push(Json::numv(0));
        Json nested = Json::objv();
        nested.set("k", Json::arrv());
        sub.push(std::move(nested));
        x.set("arr", std::move(sub));
        CHECK(roundtrip(x));
        // 纯数组与含控制字符的字符串也往返。
        Json a = Json::arrv();
        a.push(Json::strv(""));
        a.push(Json::numv(0.015));
        CHECK(roundtrip(a));
        std::string ctl;
        ctl += '\x01';
        CHECK(roundtrip(Json::strv(ctl)));
        std::printf("序列化往返：对象/数组/转义/数字多形态结构相等\n");
    }
    std::puts("json: 通过");
    return 0;
}