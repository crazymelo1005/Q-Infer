// 路由门控的回归。不引第三方框架。
//
// 五路证据：
//   1. 全等 logits：softmax 均匀，前 10 个概率相同 —— 此时**只有「同值取小索引」这条规则**能决定
//      ids，故断言 ids 恰为 0..9、每个权重恰为 0.1。这是最尖的一条。
//   2. 边界同值：前 3 个 logits 严格更大、其余全等，则剩下 7 个名额必须落在索引最小的那 7 个上。
//   3. 不变量：ids 互不相同、范围合法、按权重降序、权重之和为 1。
//   4. 归一化分母的下限（2^-14）在本模型几何下取不到：k/n_expert = 10/512 ≈ 0.0195 远大于 6.1e-5，
//      故不为它造用例，而是把「取不到」这件事断言出来（与引擎自身 parity 测试的处理一致）。
//   5. 真模型手工入口：读 blk.<L>.ffn_gate_inp.weight（BF16），跑一次完整路由并与双精度参考
//      （引擎 reference() 的转写，含钳位）逐条比对 ids 与权重。
//
// bf16 转换另测：已知值 + 就近舍入到偶数 + 非有限值的处理。
#include "experts/router.hpp"

#include "artifact/gguf_table.hpp"
#include "check.hpp"
#include "kernels/bf16.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <string>
#include <vector>

using namespace qinfer;
using namespace qinfer::experts;

namespace {

constexpr int kNExpert = 512;
constexpr int kTopK = 10;

// 引擎 reference() 的转写（含 2^-14 钳位）：全专家 softmax（双精度、减最大值）、稳定降序取前 k、
// 按前 k 之和归一化。它的保真度是另做一次性核对确认的（在环境2 上编译引擎自己的那份函数对拍）。
void reference(const float* logits, int n_expert, int k, int* ids, float* w) {
    double mx = logits[0];
    for (int e = 1; e < n_expert; ++e) mx = std::max(mx, static_cast<double>(logits[e]));
    double sum = 0.0;
    std::vector<double> p(static_cast<std::size_t>(n_expert));
    for (int e = 0; e < n_expert; ++e) {
        p[static_cast<std::size_t>(e)] = std::exp(static_cast<double>(logits[e]) - mx);
        sum += p[static_cast<std::size_t>(e)];
    }
    for (int e = 0; e < n_expert; ++e) p[static_cast<std::size_t>(e)] /= sum;
    std::vector<int> idx(static_cast<std::size_t>(n_expert));
    std::iota(idx.begin(), idx.end(), 0);
    std::stable_sort(idx.begin(), idx.end(),
                     [&](int a, int b) { return p[static_cast<std::size_t>(a)] > p[static_cast<std::size_t>(b)]; });
    double s = 0.0;
    for (int i = 0; i < k; ++i) {
        ids[i] = idx[static_cast<std::size_t>(i)];
        w[i] = static_cast<float>(p[static_cast<std::size_t>(idx[static_cast<std::size_t>(i)])]);
        s += p[static_cast<std::size_t>(idx[static_cast<std::size_t>(i)])];
    }
    s = std::max(s, kRenormClamp);
    for (int i = 0; i < k; ++i) w[i] = static_cast<float>(static_cast<double>(w[i]) / s);
}

int compare_ids(const int* got, const int* want, int k, const char* what) {
    for (int i = 0; i < k; ++i) {
        if (got[i] != want[i]) {
            std::printf("FAIL %s[%d]: got %d want %d\n", what, i, got[i], want[i]);
            return 1;
        }
    }
    return 0;
}

void test_bf16_conversion() {
    CHECK(kernels::bf16_bits_to_f32(0x3F80) == 1.0f);        // 1.0
    CHECK(kernels::bf16_bits_to_f32(0xC000) == -2.0f);
    CHECK(kernels::bf16_bits_to_f32(0x0000) == 0.0f);
    CHECK(kernels::f32_to_bf16_bits(1.0f) == 0x3F80);
    CHECK(kernels::f32_to_bf16_bits(-2.0f) == 0xC000);
    CHECK(kernels::f32_to_bf16_bits(0.0f) == 0x0000);
    CHECK(kernels::f32_to_bf16_bits(-0.0f) == 0x8000);
    // bf16 在 1.0 处的间隔是 2^-7（尾数 7 位）：下一个可表示值是 1+2^-7；
    // 1+2^-8 正好是中间点、舍到偶数（尾数 0）；1+2^-7+2^-8 的尾数是 1.5 个单位、舍到偶数 2。
    CHECK(kernels::f32_to_bf16_bits(1.0f + 0x1p-7f) == 0x3F81);
    CHECK(kernels::f32_to_bf16_bits(1.0f + 0x1p-8f) == 0x3F80);
    CHECK(kernels::f32_to_bf16_bits(1.0f + 0x1p-7f + 0x1p-8f) == 0x3F82);
    // 非有限值：Inf 原样、NaN 强制静默。
    CHECK(kernels::f32_to_bf16_bits(INFINITY) == 0x7F80);
    CHECK(kernels::f32_to_bf16_bits(-INFINITY) == 0xFF80);
    const std::uint32_t nan_bits = 0x7FC00000u;
    float nan_v;
    std::memcpy(&nan_v, &nan_bits, 4);
    CHECK((kernels::f32_to_bf16_bits(nan_v) & 0x0040u) != 0);
    // 往返：bf16 -> f32 -> bf16 必须逐位回到原值（bf16 是 f32 的高半）。
    for (std::uint32_t bits = 0; bits < 0x10000u; ++bits) {
        const std::uint16_t h = static_cast<std::uint16_t>(bits);
        const bool is_nan = ((h >> 7) & 0xFFu) == 0xFFu && (h & 0x7Fu) != 0;
        if (is_nan) continue;  // NaN 会被强制成静默，不做往返断言
        if (kernels::f32_to_bf16_bits(kernels::bf16_bits_to_f32(h)) != h) {
            std::printf("FAIL bf16 roundtrip 0x%04x\n", h);
            CHECK(false);
        }
    }
}

void test_all_equal_logits_pins_the_tie_rule() {
    std::vector<float> logits(kNExpert, 0.0f);
    RouterScratch s;
    int ids[kTopK];
    float w[kTopK];
    router_topk(logits.data(), kNExpert, kTopK, s, ids, w);
    for (int i = 0; i < kTopK; ++i) {
        if (ids[i] != i) {
            std::printf("FAIL tie ids[%d] = %d（同值必须取小索引）\n", i, ids[i]);
            CHECK(false);
        }
        if (std::fabs(w[i] - 0.1f) > 1e-7f) {
            std::printf("FAIL tie weight[%d] = %.9g\n", i, static_cast<double>(w[i]));
            CHECK(false);
        }
    }
}

void test_boundary_tie() {
    // 索引 0/1/2 严格更大，其余全等：剩下 7 个名额必须给索引 3..9。
    std::vector<float> logits(kNExpert, 0.0f);
    logits[0] = logits[1] = logits[2] = 1.0f;
    RouterScratch s;
    int ids[kTopK];
    float w[kTopK];
    router_topk(logits.data(), kNExpert, kTopK, s, ids, w);
    const int want[kTopK] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    CHECK(compare_ids(ids, want, kTopK, "boundary tie") == 0);
    double sum = 0.0;
    for (int i = 0; i < kTopK; ++i) sum += w[i];
    CHECK(std::fabs(sum - 1.0) < 1e-6);
}

void test_invariants_and_reference() {
    std::vector<float> logits(kNExpert);
    for (int e = 0; e < kNExpert; ++e) {
        // 刻意做出「两头并列」的分布：用整数化的小范围值制造大量同值。
        logits[static_cast<std::size_t>(e)] = static_cast<float>((e * 37) % 23) - 11.0f;
    }
    RouterScratch s;
    int ids[kTopK];
    float w[kTopK];
    router_topk(logits.data(), kNExpert, kTopK, s, ids, w);

    int ref_ids[kTopK];
    float ref_w[kTopK];
    reference(logits.data(), kNExpert, kTopK, ref_ids, ref_w);
    CHECK(compare_ids(ids, ref_ids, kTopK, "ids vs reference") == 0);
    for (int i = 0; i < kTopK; ++i) CHECK(w[i] == ref_w[i]);

    // 不变量。
    double sum = 0.0;
    for (int i = 0; i < kTopK; ++i) {
        sum += w[i];
        CHECK(ids[i] >= 0 && ids[i] < kNExpert);
        CHECK(w[i] > 0.0f);
        if (i > 0) CHECK(w[i - 1] >= w[i]);  // 按权重降序
    }
    CHECK(std::fabs(sum - 1.0) < 1e-6);
    for (int i = 0; i < kTopK; ++i) {
        for (int j = i + 1; j < kTopK; ++j) CHECK(ids[i] != ids[j]);
    }

    // 归一化分母的下限取不到：最小的可能分母是均匀分布下的 k/n_expert。
    CHECK(static_cast<double>(kTopK) / kNExpert > kRenormClamp);

    // 大 logits 必须仍然给出有限且正确的权重：softmax 要减最大值，否则 exp 会溢出。
    // 本例的 logits 幅值只有几，减不减最大值在数学上等价，所以必须另造一组大的才测得出这条；
    // 而且要大过 double 的 exp 溢出阈值（约 709），否则 double 下照样算得出来。
    {
        std::vector<float> big(kNExpert, -800.0f);
        big[7] = 800.0f;
        big[9] = 799.0f;
        RouterScratch sb;
        int bid[kTopK];
        float bw[kTopK];
        router_topk(big.data(), kNExpert, kTopK, sb, bid, bw);
        CHECK(bid[0] == 7);
        CHECK(bid[1] == 9);
        double bsum = 0.0;
        for (int i = 0; i < kTopK; ++i) {
            CHECK(std::isfinite(bw[i]));
            bsum += bw[i];
        }
        CHECK(std::fabs(bsum - 1.0) < 1e-6);
    }
}

void test_logits_use_bf16_activation() {
    // 路由的投影必须把「权重与激活都按 bf16 舍入」——引擎自注：这里改用 fp16 激活会引入 8.100e-03
    // 的误差并翻掉选择。参考实现按同样的 bf16 舍入在双精度下累加，容差取相对量级，故换错舍入会超差。
    const int n_expert = 64;
    const int hidden = 256;
    std::vector<std::uint16_t> w(static_cast<std::size_t>(n_expert * hidden));
    std::vector<float> x(static_cast<std::size_t>(hidden));
    for (std::size_t i = 0; i < w.size(); ++i) {
        w[i] = kernels::f32_to_bf16_bits(static_cast<float>((static_cast<int>(i % 13) - 6)) * 0.0125f);
    }
    for (std::size_t j = 0; j < x.size(); ++j) {
        // 刻意让 x 需要超过 8 位有效位：这样「按 bf16 舍入」与「按 fp16 舍入」才会给出不同结果
        // （bf16 只有 8 位尾数精度，fp16 有 11 位）。若 x 在两种格式下都精确可表示，换错舍入也测不出来。
        x[j] = static_cast<float>(static_cast<int>(j % 17) - 8) * 0.03125f +
               static_cast<float>(j % 7) * 0.0001220703125f;  // 2^-13
    }

    std::vector<float> logits(static_cast<std::size_t>(n_expert));
    router_logits(w.data(), n_expert, hidden, x.data(), logits.data());

    double mag = 0.0;
    for (int e = 0; e < n_expert; ++e) {
        double acc = 0.0;
        for (int j = 0; j < hidden; ++j) {
            const double wv = kernels::bf16_bits_to_f32(w[static_cast<std::size_t>(e) * hidden + j]);
            const double xv = kernels::bf16_bits_to_f32(
                kernels::f32_to_bf16_bits(x[static_cast<std::size_t>(j)]));
            acc += wv * xv;
            mag += std::fabs(wv * xv);
        }
        const double rel = std::fabs(static_cast<double>(logits[static_cast<std::size_t>(e)]) - acc) /
                           (mag > 0.0 ? mag : 1.0);
        if (!(rel < 1e-6)) {
            std::printf("FAIL logits[%d]: got %.9g want %.9g rel %.3g\n", e,
                        static_cast<double>(logits[static_cast<std::size_t>(e)]), acc, rel);
            CHECK(false);
        }
    }
}

void test_route_argument_checks() {
    RouterScratch s;
    int ids[kTopK];
    float w[kTopK];
    std::string err;
    std::vector<float> x(2560, 0.5f);
    std::vector<std::uint16_t> wt(static_cast<std::size_t>(kNExpert) * 2560, 0x3F80);
    // k 超过专家数必须失败。
    err.clear();
    CHECK(!route(wt.data(), kNExpert, 2560, kNExpert + 1, x.data(), s, ids, w, err));
    CHECK(!err.empty());
    // 空指针必须失败。
    err.clear();
    CHECK(!route(nullptr, kNExpert, 2560, kTopK, x.data(), s, ids, w, err));
    CHECK(!err.empty());
}

// ---- 真模型手工入口 ----

int manual_model_entry(const std::string& path, int layer, int k, const std::string& dump_path) {
    artifact::GgufFile g;
    std::string err;
    if (!g.open(path, err)) {
        std::printf("打不开：%s\n", err.c_str());
        return 1;
    }
    RouterSpec spec;
    if (!read_router_spec(g, layer, spec, err)) {
        std::printf("路由矩阵不可用：%s\n", err.c_str());
        return 1;
    }
    const artifact::GgufTensorInfo* t =
        g.find("blk." + std::to_string(layer) + ".ffn_gate_inp.weight");
    std::vector<std::uint16_t> wb(static_cast<std::size_t>(spec.hidden * spec.n_expert));
    if (!g.read_at(t->offset, reinterpret_cast<std::uint8_t*>(wb.data()), wb.size() * 2)) {
        std::printf("读路由权重失败\n");
        return 1;
    }
    std::vector<float> x(static_cast<std::size_t>(spec.hidden));
    for (std::size_t i = 0; i < x.size(); ++i) {
        x[i] = static_cast<float>((static_cast<int>(i % 31) - 15) * 0.03125);
    }
    RouterScratch s;
    std::vector<int> ids(static_cast<std::size_t>(k));
    std::vector<float> w(static_cast<std::size_t>(k));
    if (!route(wb.data(), static_cast<int>(spec.n_expert), static_cast<int>(spec.hidden), k,
               x.data(), s, ids.data(), w.data(), err)) {
        std::printf("路由失败：%s\n", err.c_str());
        return 1;
    }
    std::printf("层 %d 路由：hidden=%llu n_expert=%llu k=%d\n", layer,
                static_cast<unsigned long long>(spec.hidden),
                static_cast<unsigned long long>(spec.n_expert), k);
    std::printf("  ids    ");
    for (int i = 0; i < k; ++i) std::printf(" %d", ids[i]);
    std::printf("\n  weights");
    for (int i = 0; i < k; ++i) std::printf(" %.6f", w[i]);
    std::printf("\n");
    double lmax = s.logits[0], lmin = s.logits[0], lsum = 0.0;
    for (float v : s.logits) {
        lmax = std::max(lmax, static_cast<double>(v));
        lmin = std::min(lmin, static_cast<double>(v));
        lsum += static_cast<double>(v);
    }
    std::printf("  logits min %.6f max %.6f sum %.6f\n", lmin, lmax, lsum);
    if (!dump_path.empty()) {
        // 把 logits 原样落盘：便于用车库外的实现（引擎自身的 reference()）在同样的输入上对拍。
        std::FILE* f = std::fopen(dump_path.c_str(), "w");
        if (f == nullptr) {
            std::printf("写 logits 失败：%s\n", dump_path.c_str());
            return 1;
        }
        std::fprintf(f, "%llu\n", static_cast<unsigned long long>(spec.n_expert));
        for (float v : s.logits) std::fprintf(f, "%.9g\n", static_cast<double>(v));
        std::fclose(f);
        std::printf("  logits 已写入 %s\n", dump_path.c_str());
    }

    std::vector<int> rid(static_cast<std::size_t>(k));
    std::vector<float> rw(static_cast<std::size_t>(k));
    reference(s.logits.data(), static_cast<int>(spec.n_expert), k, rid.data(), rw.data());
    const int bad_ids = compare_ids(ids.data(), rid.data(), k, "真模型 ids vs 参考");
    int bad_w = 0;
    for (int i = 0; i < k; ++i) {
        if (w[i] != rw[i]) {
            std::printf("FAIL weight[%d]: got %.9g want %.9g\n", i, static_cast<double>(w[i]),
                        static_cast<double>(rw[i]));
            bad_w = 1;
        }
    }
    std::printf("与双精度参考比对：ids %s，weights %s\n", bad_ids ? "不一致" : "一致",
                bad_w ? "不一致" : "一致");
    return bad_ids || bad_w;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc >= 2 && std::string(argv[1]) == "--model") {
        int layer = 0, k = kTopK;
        std::string dump;
        for (int i = 2; i + 1 < argc; ++i) {
            if (std::string(argv[i]) == "--layer") layer = std::atoi(argv[i + 1]);
            if (std::string(argv[i]) == "--k") k = std::atoi(argv[i + 1]);
            if (std::string(argv[i]) == "--dump-logits") dump = argv[i + 1];
        }
        return manual_model_entry(argv[2], layer, k, dump);
    }
    test_bf16_conversion();
    test_all_equal_logits_pins_the_tie_rule();
    test_boundary_tie();
    test_invariants_and_reference();
    test_logits_use_bf16_activation();
    test_route_argument_checks();
    std::puts("router: tie rule, bf16 conversion, invariants and double-precision reference hold");
    return 0;
}