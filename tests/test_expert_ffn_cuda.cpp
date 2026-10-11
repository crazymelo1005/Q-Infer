// 设备侧的一层专家前馈（权重常驻）与主机 `expert_ffn` 的对照。只在 -DQINFER_ENABLE_CUDA=ON 时构建。
//
// 夹具的几何与 test_iq2s_cuda / test_q2_0_cuda 同一套：ffn 64、hidden 256，gate/up 用 IQ2_S
// （每行一块 256 值）、down 用 Q2_0（每行一块 64 值），字节取自两份真模型 oracle。
//
// 五路证据：
//   1. 每个专家的一次前馈，设备结果与主机 `expert_ffn` 对齐（逐元素，报告位相同的个数与 L1 相对差）。
//      一层的三档在设备侧都有内核，故这条覆盖的是「四个 kernel 串起来 + 设备侧 h 量化」。
//   2. 常驻：`uploaded_bytes` 在第一次上传后不再增长——第二个 `run` 不该再搬权重。
//   3. 槽位：同一批里两个专家用同一个槽时，两次结果都等于该槽里那份权重（不串）。
//   4. 越出支持范围明确失败：down 的激活搭档不是 Q8_0、或三矩阵里有设备侧没有内核的档。
//   5. 入参不合法明确失败。
#include "experts/cuda/expert_ffn_cuda.hpp"

#include "check.hpp"
#include "kernels/q8_0.hpp"
#include "kernels/q8k.hpp"
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace qinfer::experts;

namespace {

#include "iq2s_oracle_data.inc"
#include "q2_0_oracle_data.inc"

constexpr int kHidden = 256;
constexpr int kFfn = 64;

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

// 层的规格：gate/up = IQ2_S（64 行 × 256 列 = 每行 1 块），down = Q2_0（256 行 × 64 列 = 每行 1 块）。
LayerSpec make_spec() {
    LayerSpec s;
    s.layer = 0;
    s.gate.format = Format::kIq2S;
    s.gate.rows = kFfn;
    s.gate.cols = kHidden;
    s.up = s.gate;
    s.down.format = Format::kQ2_0;
    s.down.rows = kHidden;
    s.down.cols = kFfn;
    return s;
}

// 一个专家的三矩阵字节：从真模型 oracle 里 tile 到该有的长度。
struct ExpertBytes {
    std::vector<std::uint8_t> gate, up, down;
    ExpertWeights view() const {
        return ExpertWeights{gate.data(), up.data(), down.data()};
    }
};

std::vector<std::uint8_t> tile(const std::vector<std::uint8_t>& src, int block_bytes,
                               std::size_t want_bytes) {
    CHECK(src.size() % static_cast<std::size_t>(block_bytes) == 0);
    const std::size_t n_real = src.size() / static_cast<std::size_t>(block_bytes);
    CHECK(n_real > 0 && want_bytes % static_cast<std::size_t>(block_bytes) == 0);
    std::vector<std::uint8_t> out(want_bytes);
    for (std::size_t i = 0; i < want_bytes / static_cast<std::size_t>(block_bytes); ++i) {
        std::memcpy(out.data() + i * static_cast<std::size_t>(block_bytes),
                    src.data() + (i % n_real) * static_cast<std::size_t>(block_bytes),
                    static_cast<std::size_t>(block_bytes));
    }
    return out;
}

ExpertBytes make_expert(int salt) {
    const std::vector<std::uint8_t> iq2s = from_hex(kIq2sRealHex);
    const std::vector<std::uint8_t> q20 = from_hex(kQ20RealHex);
    ExpertBytes e;
    e.gate = tile(iq2s, 82, static_cast<std::size_t>(kFfn) * 82);
    e.up = tile(iq2s, 82, static_cast<std::size_t>(kFfn) * 82);
    e.down = tile(q20, 18, static_cast<std::size_t>(kHidden) * 18);
    // 每个专家的字节不同：把第一个块的第 2 个字节扰动一下（仍落在合法字节上，故仍是真格式的字节）。
    e.gate[2] = static_cast<std::uint8_t>(e.gate[2] ^ (salt & 0x0Fu));
    e.up[3] = static_cast<std::uint8_t>(e.up[3] ^ ((salt * 3) & 0x0Fu));
    e.down[2] = static_cast<std::uint8_t>(e.down[2] ^ ((salt * 5) & 0x0Fu));
    return e;
}

std::vector<float> make_x() {
    std::vector<float> x(kHidden);
    for (int i = 0; i < kHidden; ++i) {
        x[static_cast<std::size_t>(i)] = static_cast<float>((i % 23) - 11) * 0.09375f;
    }
    return x;
}

// 断言带上原来那条 err：不然失败时只知道「返回了假」，看不出为什么（这一片就吃过一次亏）。
void must(bool ok, const std::string& err, const char* what) {
    if (!ok) std::printf("    失败：%s（%s）\n", what, err.c_str());
    CHECK(ok);
}

void test_device_matches_host() {
    const LayerSpec spec = make_spec();
    const std::vector<ExpertBytes> experts{make_expert(1), make_expert(2)};
    const std::vector<float> x = make_x();

    // 主机参考：逐个专家走 expert_ffn。
    std::vector<float> host(experts.size() * kHidden, 0.0f);
    for (std::size_t e = 0; e < experts.size(); ++e) {
        FfnScratch sc;
        std::string err;
        CHECK(expert_ffn(spec, experts[e].view(), x.data(), sc,
                         host.data() + e * kHidden, err));
    }

    ExpertFfnDevice dev;
    std::string err;
    must(dev.init(0, spec, static_cast<int>(experts.size()), err), err, "init");
    for (std::size_t e = 0; e < experts.size(); ++e) {
        must(dev.upload(static_cast<int>(e), experts[e].view(), err), err, "upload");
    }
    const std::uint64_t per_expert =
        static_cast<std::uint64_t>(kFfn) * 82 * 2 + static_cast<std::uint64_t>(kHidden) * 18;
    CHECK(dev.resident_bytes() == per_expert * experts.size());
    CHECK(dev.uploaded_bytes() == dev.resident_bytes());

    std::vector<int> slots{0, 1};
    std::vector<float> got(experts.size() * kHidden, 0.0f);
    must(dev.run(slots.data(), static_cast<int>(experts.size()), x.data(), got.data(), err), err,
         "第一次 run");

    // 常驻：第二个 run 不再搬权重。
    const std::uint64_t after_first = dev.uploaded_bytes();
    must(dev.run(slots.data(), static_cast<int>(experts.size()), x.data(), got.data(), err), err,
         "第二次 run");
    CHECK(dev.uploaded_bytes() == after_first);

    int same = 0;
    double num = 0.0, den = 0.0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        if (std::bit_cast<std::uint32_t>(got[i]) == std::bit_cast<std::uint32_t>(host[i])) ++same;
        num += std::fabs(static_cast<double>(got[i]) - static_cast<double>(host[i]));
        den += std::fabs(static_cast<double>(host[i]));
    }
    const double rel = den > 0.0 ? num / den : num;
    std::printf("  设备 vs 主机：逐元素位相同 %d/%zu，L1 相对差 %.3g\n", same, got.size(), rel);
    CHECK(den > 0.0);
    CHECK(rel < 1e-6);
    CHECK(same > 0);
}

void test_same_slot_twice_gives_that_slot() {
    // 一批里两个位置用同一个槽：两次结果都该等于该槽里那份权重（不串到别的槽）。
    const LayerSpec spec = make_spec();
    const ExpertBytes e0 = make_expert(3);
    const std::vector<float> x = make_x();
    FfnScratch sc;
    std::string err;
    std::vector<float> want(kHidden, 0.0f);
    CHECK(expert_ffn(spec, e0.view(), x.data(), sc, want.data(), err));

    // 槽位给两个（一批的专家数不能超过槽位数：结果缓冲是按槽位数开的），但只往 0 号槽放权重，
    // 再让这一批的两个位置都用 0 号槽——这才是那条「同一槽用两次不串」的用例。
    ExpertFfnDevice dev;
    must(dev.init(0, spec, 2, err), err, "init");
    must(dev.upload(0, e0.view(), err), err, "upload");
    const std::vector<int> slots{0, 0};
    std::vector<float> got(2 * kHidden, 0.0f);
    must(dev.run(slots.data(), 2, x.data(), got.data(), err), err, "同一槽两次");
    for (int e = 0; e < 2; ++e) {
        for (int i = 0; i < kHidden; ++i) {
            CHECK(std::bit_cast<std::uint32_t>(got[static_cast<std::size_t>(e * kHidden + i)]) ==
                  std::bit_cast<std::uint32_t>(want[static_cast<std::size_t>(i)]));
        }
    }
}

void test_unsupported_layers_fail_loudly() {
    ExpertFfnDevice dev;
    std::string err;
    // down 换成 IQ2_S（激活搭档是 Q8_K，设备侧只有 Q8_0 的 h 量化器）：明确失败。
    LayerSpec bad = make_spec();
    bad.down.format = Format::kIq2S;
    bad.down.cols = kFfn;
    CHECK(!dev.init(0, bad, 1, err));
    CHECK(!err.empty());
    err.clear();
    // gate 换成设备侧没有内核的档（IQ2_XS 本仓库有主机内核、设备侧还没做）：明确失败。
    LayerSpec bad2 = make_spec();
    bad2.gate.format = Format::kIq2Xs;
    CHECK(!dev.init(0, bad2, 1, err));
    CHECK(!err.empty());
    err.clear();
    // 几何不成立：down 的列数不是 64 的整数倍。
    LayerSpec bad3 = make_spec();
    bad3.down.cols = kFfn + 1;
    CHECK(!dev.init(0, bad3, 1, err));
    CHECK(!err.empty());
}

void test_bad_inputs_fail_reply() {
    const LayerSpec spec = make_spec();
    ExpertFfnDevice dev;
    std::string err;
    // 没 init 就用。
    CHECK(!dev.upload(0, make_expert(1).view(), err));
    CHECK(!err.empty());
    err.clear();
    std::vector<float> x = make_x();
    std::vector<float> out(kHidden, 0.0f);
    const std::vector<int> slots{0};
    CHECK(!dev.run(slots.data(), 1, x.data(), out.data(), err));
    CHECK(!err.empty());

    CHECK(dev.init(0, spec, 2, err));
    err.clear();
    const int oob = 2;
    CHECK(!dev.upload(oob, make_expert(1).view(), err));
    CHECK(!err.empty());
    err.clear();
    CHECK(!dev.run(slots.data(), 0, x.data(), out.data(), err));
    CHECK(!err.empty());
    err.clear();
    CHECK(!dev.run(slots.data(), 1, nullptr, out.data(), err));
    CHECK(!err.empty());
    err.clear();
    const std::vector<int> bad_slots{5};
    CHECK(!dev.run(bad_slots.data(), 1, x.data(), out.data(), err));
    CHECK(!err.empty());
}

}  // namespace

int main() {
    test_device_matches_host();
    test_same_slot_twice_gives_that_slot();
    test_unsupported_layers_fail_loudly();
    test_bad_inputs_fail_reply();
    std::puts("expert_ffn_cuda: 设备侧一层专家前馈（权重常驻）与主机 expert_ffn 一致");
    return 0;
}
