// 手写 JSON 子集：对象、数组、字符串（含转义与 \uXXXX、代理对）、数字、布尔、空。
// 零依赖（ADR-008 在协议层同样成立）；解析严格（越界输入返回假，上层回 400）；序列化紧凑。
#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace qinfer::serve {

class Json {
public:
    enum Kind { kNull, kBool, kNum, kStr, kArr, kObj };
    Kind kind = kNull;
    bool b = false;
    double num = 0.0;
    std::string str;
    std::vector<Json> arr;
    std::vector<std::pair<std::string, Json>> obj;   // 保序（响应按固定顺序拼）

    static Json nullv() { return {}; }
    static Json boolv(bool v) { Json j; j.kind = kBool; j.b = v; return j; }
    static Json numv(double v) { Json j; j.kind = kNum; j.num = v; return j; }
    static Json strv(std::string v) { Json j; j.kind = kStr; j.str = std::move(v); return j; }
    static Json arrv() { Json j; j.kind = kArr; return j; }
    static Json objv() { Json j; j.kind = kObj; return j; }

    bool is_null() const { return kind == kNull; }
    bool is_str() const { return kind == kStr; }
    bool is_num() const { return kind == kNum; }
    bool is_arr() const { return kind == kArr; }
    bool is_obj() const { return kind == kObj; }

    const Json* find(const std::string& key) const;   // 非对象/缺键 → nullptr
    Json* find(const std::string& key);
    void push(Json v) { if (kind != kArr) kind = kArr; arr.push_back(std::move(v)); }
    void set(const std::string& key, Json v) {
        if (kind != kObj) kind = kObj;
        obj.emplace_back(key, std::move(v));
    }
};

bool json_parse(const std::string& s, Json& out, std::string& err);   // 语法错 → false + err（含位置）
std::string json_dump(const Json& j);                                 // 紧凑
std::string json_escape(const std::string& s);                        // JSON 字符串字面量转义

}  // namespace qinfer::serve