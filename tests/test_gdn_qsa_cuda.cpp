// 阶段 A 设备路径对照载体：GDN/QSA 稠密段各内核的真字节夹具与判据。
// 本文件由 measure/ 的 oracle 工具生成的 .inc 提供真字节（部署 GGUF 的投影权重前若干块），
// 目标判据照 ADR-009 分布级口径：位相同比例 + L1 相对差小（h 量化在舍入边界可差一步）。
// 后续内核（gdn_proj_gemv / gdn_preprocess_cuda / gdn_step_cuda / qsa_*_cuda）逐个在这里对照。
#include <bit>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <string>
#include <vector>

// oracle .inc 用 std::uint32_t 等类型，须在 <cstdint> 之后引入。
// 这些是阶段 A 的稠密投影档夹具（gdn_ 前缀，避免与主机同档测试的 .inc 撞名）。
#include "gdn_iq3s_oracle_data.inc"
#include "gdn_iq4xs_oracle_data.inc"
#include "gdn_q6k_oracle_data.inc"
#include "check.hpp"

namespace {

// hex 字符串 → 字节（oracle .inc 的 kXxxRealHex 是纯 hex 串）。
std::vector<std::uint8_t> from_hex(const std::string& hex) {
    CHECK(hex.size() % 2 == 0);
    std::vector<std::uint8_t> out;
    out.reserve(hex.size() / 2);
    auto nib = [](char c) -> int {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        CHECK(false && "bad hex digit");
        return 0;
    };
    for (std::size_t i = 0; i < hex.size(); i += 2) {
        out.push_back(static_cast<std::uint8_t>(nib(hex[i]) * 16 + nib(hex[i + 1])));
    }
    return out;
}

// 分发级判据（ADR-009）：位相同比例 ≥ 0.99 且 L1 相对差 ≤ 1e-4。返回 0 = 通过。
int compare_rows(const std::vector<float>& a, const std::vector<float>& b, double l1_threshold,
                 double bit_ratio_threshold) {
    CHECK(a.size() == b.size());
    std::uint64_t same = 0, total = a.size();
    double diff_sum = 0.0, abs_sum = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (std::bit_cast<std::uint32_t>(a[i]) == std::bit_cast<std::uint32_t>(b[i])) ++same;
        diff_sum += std::abs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
        abs_sum += std::abs(static_cast<double>(a[i]));
    }
    const double l1 = abs_sum > 0.0 ? diff_sum / abs_sum : 0.0;
    const double bit_ratio = total > 0 ? static_cast<double>(same) / static_cast<double>(total) : 1.0;
    std::printf("  bit-same %llu/%llu (%.3f), L1 rel %.3e\n",
                static_cast<unsigned long long>(same), static_cast<unsigned long long>(total),
                bit_ratio, l1);
    return bit_ratio >= bit_ratio_threshold && l1 <= l1_threshold ? 0 : 1;
}

// 夹具可用性：三个新档的 oracle .inc 能解码、块数与字节数自洽（Task 1 RED 基线，只钉数据本身）。
void test_oracle_fixtures_load() {
    const auto iq4xs = from_hex(kIq4xsRealHex);
    const auto iq3s = from_hex(kIq3sRealHex);
    const auto q6k = from_hex(kQ6kRealHex);
    // 每块字节数（oracle 头注释写明）：IQ4_XS 136、IQ3_S 110、Q6_K 210；各取 64 块。
    CHECK(iq4xs.size() == 64 * 136);
    CHECK(iq3s.size() == 64 * 110);
    CHECK(q6k.size() == 64 * 210);
    std::printf("oracle 夹具自洽：IQ4_XS %zu B、IQ3_S %zu B、Q6_K %zu B\n", iq4xs.size(),
                iq3s.size(), q6k.size());
}

}  // namespace

int main() {
    test_oracle_fixtures_load();
    std::puts("gdn_qsa_cuda: 真字节夹具工程就绪（IQ4_XS/IQ3_S/Q6_K oracle 自洽）");
    return 0;
}