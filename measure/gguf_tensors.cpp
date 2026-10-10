// 列一个 GGUF 里的张力：名字、维度、类型码与块几何、字节数。给「实现某个稠密/注意力部件之前先把
// 真模型的张力清单与档位取出来」用。不是测量工具，不进 ctest。
//
// 用法：gguf_tensors <model.gguf> [名字子串] [--limit N]
//   不带子串则只打印汇总结论（每个类型各多少个张力、共多少字节）。
// 类型码的名字表取自 S-42 登记的那张 GGML_QUANT_SIZES（块元素数 / 块字节数），只列本仓库用得到的档。
#include "artifact/gguf_table.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace {

// 码与块几何都取自**已核对过的来源**，不要凭记忆重打：
//   * 码：llama.cpp 的 `enum ggml_type`（ggml/include/ggml.h，提交 3cf03257；注意 41 = Q1_0、42 = Q2_0
//     是较新的追加，6 = Q5_0、15 = Q8_K、16/17/18 = IQ2_XXS/IQ2_XS/IQ3_XXS 这一段的错位最容易记岔）；
//   * 块几何：本仓库 `src/experts/expert_formats.cpp` 的那张表（与 gguf-py 的 GGML_QUANT_SIZES 核对过，
//     见 S-42）。
// 这份表第一版是凭记忆写的，把 IQ3_XXS 认成了码 22、IQ2_S 认成 21、Q8_K 认成 16……于是列出来的清单
// 名字全错、看上去却完全可信（计数还对得上）——「码→名字」和「格式事实」一样，必须从已核对的来源取。
struct QkInfo {
    const char* name;
    std::uint64_t elems;  // 每块元素数
    std::uint64_t bytes;  // 每块字节数
};

// 只列见过的档；未列出的码按「未知」打印，不猜。
const QkInfo* quant_info(std::uint32_t code) {
    static const std::map<std::uint32_t, QkInfo> k = {
        {0, {"F32", 1, 4}},        {1, {"F16", 1, 2}},        {2, {"Q4_0", 32, 18}},
        {3, {"Q4_1", 32, 20}},     {6, {"Q5_0", 32, 22}},     {7, {"Q5_1", 32, 24}},
        {8, {"Q8_0", 32, 34}},     {10, {"Q2_K", 256, 84}},   {11, {"Q3_K", 256, 110}},
        {12, {"Q4_K", 256, 144}},  {13, {"Q5_K", 256, 176}},  {14, {"Q6_K", 256, 210}},
        {15, {"Q8_K", 256, 292}},  {16, {"IQ2_XXS", 256, 66}}, {17, {"IQ2_XS", 256, 74}},
        {18, {"IQ3_XXS", 256, 98}}, {19, {"IQ1_S", 256, 50}}, {20, {"IQ4_NL", 32, 18}},
        {21, {"IQ3_S", 256, 110}}, {22, {"IQ2_S", 256, 82}},  {23, {"IQ4_XS", 256, 136}},
        {29, {"IQ1_M", 256, 56}},  {30, {"BF16", 1, 2}},      {42, {"Q2_0", 64, 18}},
    };
    const auto it = k.find(code);
    return it == k.end() ? nullptr : &it->second;
}

std::string type_name(std::uint32_t code) {
    const QkInfo* q = quant_info(code);
    if (q == nullptr) return "码 " + std::to_string(code) + "（未知）";
    return std::string(q->name) + "（码 " + std::to_string(code) + "，" + std::to_string(q->elems) +
           " 值/" + std::to_string(q->bytes) + " 字节每块）";
}

std::uint64_t bytes_of(const qinfer::artifact::GgufTensorInfo& t) {
    const QkInfo* q = quant_info(t.type);
    if (q == nullptr || t.elements() == 0) return 0;
    const std::uint64_t nblk = (t.elements() + q->elems - 1) / q->elems;
    return nblk * q->bytes;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("用法：gguf_tensors <model.gguf> [名字子串] [--limit N]\n");
        return 2;
    }
    const char* filter = nullptr;
    std::uint64_t limit = 60;
    long type_filter = -1;
    for (int i = 2; i < argc; ++i) {
        if (std::strcmp(argv[i], "--limit") == 0 && i + 1 < argc) {
            limit = static_cast<std::uint64_t>(std::atoi(argv[++i]));
        } else if (std::strcmp(argv[i], "--type") == 0 && i + 1 < argc) {
            type_filter = std::atol(argv[++i]);
        } else if (filter == nullptr) {
            filter = argv[i];
        }
    }
    qinfer::artifact::GgufFile g;
    std::string err;
    if (!g.open(argv[1], err)) {
        std::printf("打不开模型：%s（%s）\n", argv[1], err.c_str());
        return 1;
    }
    std::printf("张力总数 %zu，文件大小 %llu 字节\n", g.tensors().size(),
                static_cast<unsigned long long>(g.file_size()));
    std::uint64_t shown = 0;
    std::map<std::uint32_t, std::pair<std::uint64_t, std::uint64_t>> by_type;  // 码 -> (个数, 字节)
    for (const auto& t : g.tensors()) {
        const std::uint64_t b = bytes_of(t);
        auto& e = by_type[t.type];
        ++e.first;
        e.second += b;
        if (filter != nullptr && t.name.find(filter) == std::string::npos) continue;
        if (type_filter >= 0 && t.type != static_cast<std::uint32_t>(type_filter)) continue;
        if (shown++ >= limit) continue;
        std::string dims;
        for (std::size_t i = 0; i < t.dims.size(); ++i) {
            dims += (i == 0 ? "" : "×") + std::to_string(t.dims[i]);
        }
        std::printf("  %-52s [%s]  %s  %llu 字节\n", t.name.c_str(), dims.c_str(),
                    type_name(t.type).c_str(), static_cast<unsigned long long>(b));
    }
    if (filter != nullptr) {
        std::printf("匹配「%s」的张力：共 %llu 个（上面最多列 %llu 个）\n", filter,
                    static_cast<unsigned long long>(shown),
                    static_cast<unsigned long long>(limit));
    }
    std::printf("--- 按类型汇总 ---\n");
    for (const auto& [code, cnt] : by_type) {
        std::printf("  %-40s %5llu 个  %.3f GiB\n", type_name(code).c_str(),
                    static_cast<unsigned long long>(cnt.first),
                    static_cast<double>(cnt.second) / (1024.0 * 1024.0 * 1024.0));
    }
    return 0;
}