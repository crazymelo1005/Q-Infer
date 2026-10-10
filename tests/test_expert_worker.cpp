// 专家作业与工人的回归（interfaces.md 第 8 节）。不引第三方框架。
//
// 合成几何：hidden = ffn = 64，三个矩阵都用 Q2_0（一块 64 值 / 18 字节），权重全为 −d（码 0）。
// 两个专家的 d 不同，故「谁的结果写到了哪一段」可被断言；这也顺带抓「两个专家写同一段缓冲」。
//
// 五条：CPU 工人逐专家结果与直接调 expert_ffn 一致；作业指针/规模不合法时明确失败并计数；
// 核显占位不可用且报得出原因；重派语义（不可用或提交失败都改派给 CPU 且不改结果）；
// 两边都失败时错误向上抛。
#include "experts/expert_worker.hpp"

#include "check.hpp"
#include "kernels/fp16.hpp"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

using namespace qinfer;
using namespace qinfer::experts;

namespace {

constexpr std::uint64_t kN = 64;  // hidden 与 ffn 都是 64

// 一个 64×64 的 Q2_0 矩阵：每行一块（64 值 / 18 字节），块内 d 给定、16 个码字节全 0
// （码 0 -> 值 = −d）。总字节数必须是 rows × 18——只给一块会被 expert_ffn 越界读到后面的内存。
std::vector<std::uint8_t> q20_matrix(float d) {
    std::vector<std::uint8_t> m(static_cast<std::size_t>(kN) * 18, 0);
    const std::uint16_t bits = kernels::f32_to_f16_bits(d);
    for (std::uint64_t r = 0; r < kN; ++r) {
        m[r * 18] = static_cast<std::uint8_t>(bits & 0xFF);
        m[r * 18 + 1] = static_cast<std::uint8_t>(bits >> 8);
    }
    return m;  // 码字节保持 0
}

LayerSpec make_spec() {
    LayerSpec s;
    s.layer = 0;
    s.gate.format = Format::kQ2_0;
    s.gate.cols = kN;
    s.gate.rows = kN;
    s.gate.experts = 1;
    s.up = s.gate;
    s.down.format = Format::kQ2_0;
    s.down.cols = kN;
    s.down.rows = kN;
    s.down.experts = 1;
    return s;
}

std::vector<float> make_x() {
    std::vector<float> x(kN);
    for (std::size_t i = 0; i < x.size(); ++i) {
        x[i] = static_cast<float>((static_cast<int>(i % 7) - 3)) * 0.25f;
    }
    return x;
}

void test_cpu_worker_matches_direct_call() {
    const LayerSpec spec = make_spec();
    const std::vector<std::uint8_t> g0 = q20_matrix(1.0f), g1 = q20_matrix(0.5f);
    const std::vector<std::uint8_t> u0 = q20_matrix(2.0f), u1 = q20_matrix(1.0f);
    const std::vector<std::uint8_t> d0 = q20_matrix(1.0f), d1 = q20_matrix(0.25f);
    const ExpertWeights ex[2] = {{g0.data(), u0.data(), d0.data()},
                                {g1.data(), u1.data(), d1.data()}};
    const std::vector<float> x = make_x();
    std::vector<float> out(2 * kN, 0.0f);

    ExpertJob job;
    job.spec = &spec;
    job.experts = ex;
    job.n = 2;
    job.x = x.data();
    job.hidden = kN;
    job.out = out.data();

    CpuExpertWorker cpu;
    CHECK(cpu.available());
    std::string err;
    CHECK(cpu.submit(job, err));
    CHECK(err.empty());
    CHECK(cpu.wait(err));  // 幂等：没有待等的作业也返回真
    CHECK(cpu.wait(err));
    CHECK(cpu.stats().jobs == 1 && cpu.stats().experts == 2 && cpu.stats().failed == 0);

    // 逐专家与直接调 expert_ffn 一致。
    FfnScratch scratch;
    for (int e = 0; e < 2; ++e) {
        std::vector<float> direct(kN, 0.0f);
        CHECK(expert_ffn(spec, ex[e], x.data(), scratch, direct.data(), err));
        for (std::size_t i = 0; i < kN; ++i) {
            const float got = out[static_cast<std::size_t>(e) * kN + i];
            if (std::bit_cast<std::uint32_t>(got) != std::bit_cast<std::uint32_t>(direct[i])) {
                std::printf("FAIL 专家 %d 第 %zu 个值：得 %.9g 期 %.9g\n", e, i,
                            static_cast<double>(got), static_cast<double>(direct[i]));
                CHECK(false);
            }
        }
    }
    // 两个专家的 d 不同，故两段结果必须不同（抓「写同一段缓冲」）。
    bool differs = false;
    for (std::size_t i = 0; i < kN; ++i) {
        if (std::bit_cast<std::uint32_t>(out[i]) !=
            std::bit_cast<std::uint32_t>(out[kN + i])) {
            differs = true;
        }
    }
    CHECK(differs);
}

void test_bad_jobs_fail_loudly() {
    CpuExpertWorker cpu;
    std::string err;
    ExpertJob job;  // 全空
    CHECK(!cpu.submit(job, err));
    CHECK(!err.empty());
    CHECK(cpu.stats().failed == 1);

    const LayerSpec spec = make_spec();
    const std::vector<std::uint8_t> b = q20_matrix(1.0f);
    const ExpertWeights ex[1] = {{b.data(), b.data(), b.data()}};
    const std::vector<float> x = make_x();
    std::vector<float> out(kN, 0.0f);
    job.spec = &spec;
    job.experts = ex;
    job.x = x.data();
    job.hidden = kN;
    job.out = out.data();

    job.n = 0;  // 没有专家
    err.clear();
    CHECK(!cpu.submit(job, err));
    CHECK(!err.empty());

    job.n = 1;
    job.hidden = 0;  // hidden 为 0
    err.clear();
    CHECK(!cpu.submit(job, err));
    CHECK(!err.empty());
    CHECK(cpu.stats().failed == 3);
    CHECK(cpu.stats().jobs == 0);
}

void test_igpu_placeholder_is_unavailable_with_reason() {
    IgpuExpertWorker igpu(/*device_usable=*/false);
    CHECK(!igpu.available());
    const LayerSpec spec = make_spec();
    const std::vector<std::uint8_t> b = q20_matrix(1.0f);
    const ExpertWeights ex[1] = {{b.data(), b.data(), b.data()}};
    const std::vector<float> x = make_x();
    std::vector<float> out(kN, 0.0f);
    ExpertJob job;
    job.spec = &spec;
    job.experts = ex;
    job.n = 1;
    job.x = x.data();
    job.hidden = kN;
    job.out = out.data();

    std::string err;
    CHECK(!igpu.submit(job, err));
    // 原因里要能看出「本机没有计算运行时」这件事，而不是一句泛泛的失败。
    CHECK(err.find("无计算运行时") != std::string::npos);
    CHECK(err.find("integer_dot_product") != std::string::npos);
    CHECK(igpu.stats().failed == 1);
}

void test_reroute_keeps_results_and_counts() {
    const LayerSpec spec = make_spec();
    const std::vector<std::uint8_t> b = q20_matrix(1.0f);
    const ExpertWeights ex[1] = {{b.data(), b.data(), b.data()}};
    const std::vector<float> x = make_x();

    ExpertJob job;
    job.spec = &spec;
    job.experts = ex;
    job.n = 1;
    job.x = x.data();
    job.hidden = kN;

    // (1) 核显不可用 -> 改派给 CPU，结果与纯 CPU 一致。
    IgpuExpertWorker igpu(false);
    CpuExpertWorker cpu;
    std::vector<float> via_reroute(kN, 0.0f);
    std::vector<float> via_cpu(kN, 0.0f);
    job.out = via_reroute.data();
    RerouteStats rs{};
    std::string err;
    CHECK(run_job_with_fallback(igpu, cpu, job, rs, err));
    CHECK(err.empty());
    CHECK(rs.rerouted == 1 && rs.routed == 0);
    // 不可用的 worker 不该被调用：它的失败计数必须保持 0（否则观测里会全是假失败）。
    CHECK(igpu.stats().failed == 0);
    job.out = via_cpu.data();
    CHECK(cpu.submit(job, err));
    CHECK(cpu.wait(err));
    for (std::size_t i = 0; i < kN; ++i) {
        CHECK(std::bit_cast<std::uint32_t>(via_reroute[i]) ==
              std::bit_cast<std::uint32_t>(via_cpu[i]));
    }

    // (2) 可用时走首选 worker，不计重派。
    RerouteStats rs2{};
    job.out = via_reroute.data();
    CHECK(run_job_with_fallback(cpu, cpu, job, rs2, err));
    CHECK(rs2.routed == 1 && rs2.rerouted == 0);

    // (3) 两边都失败（作业不合法）时错误向上抛。
    ExpertJob bad;
    RerouteStats rs3{};
    IgpuExpertWorker igpu2(false);
    std::string err3;
    CHECK(!run_job_with_fallback(igpu2, cpu, bad, rs3, err3));
    CHECK(!err3.empty());
    CHECK(rs3.routed == 0 && rs3.rerouted == 0);
}

}  // namespace

int main() {
    test_cpu_worker_matches_direct_call();
    test_bad_jobs_fail_loudly();
    test_igpu_placeholder_is_unavailable_with_reason();
    test_reroute_keeps_results_and_counts();
    std::puts("expert_worker: CPU 工人与直接调用一致、坏作业明确失败、核显占位与重派语义成立");
    return 0;
}