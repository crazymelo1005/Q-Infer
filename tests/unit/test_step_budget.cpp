// 计划层到仲裁器的缝：预算推导、需求转换、每步观测。不引第三方框架。
//
// 四路证据：
//   1. 预算推导与画像档案逐位对上：环境2 的 `measure/results/20261009T234327-calibrate-环境2.json`
//      记的 pcie_h2d_gbps 28.64 / step_p50_ms 12.74 / io_share 0.5 推出的 pcie_step_budget_mib
//      必须等于档案里的 186.8；并钉住画像的换算口径（GB 按 1024 MiB），以及「按十进制 GB 会差 7.4%」
//      这条口径提醒。
//   2. 表行需求与档案对上：档案的 demand_table_mib 0.00155 反推 positions_per_token ≈ 1.13。
//   3. 三类需求转成请求后，仲裁的优先级与放弃语义仍成立：KV 先于表行、必需项永不丢、预测性按
//      概率÷字节排序、放不下的记迟到。
//   4. 观测记录的每一格都能从仲裁结果复算，且预算不足时「必需项 overspend、预测性迟到」两笔账
//      不会互相污染。
#include "scheduling/step_budget.hpp"

#include "check.hpp"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

using namespace qinfer::scheduling;

namespace {

constexpr std::uint64_t kMib = 1024ull * 1024ull;

// 环境2 画像档案里的三项（measure/results/20261009T234327-calibrate-环境2.json）。
const ProfileInputs kEnv2{/*pcie_h2d_gbps=*/28.64, /*step_ms=*/12.74, /*io_share=*/0.5};

double to_mib(std::uint64_t bytes) { return static_cast<double>(bytes) / static_cast<double>(kMib); }

void test_budget_matches_profile() {
    const std::uint64_t bytes = step_budget_bytes(kEnv2);
    const double mib = to_mib(bytes);
    // 档案记的是 186.8（一位小数）。
    if (std::fabs(mib - 186.8) > 0.05) {
        std::printf("FAIL 每步预算：得 %.3f MiB，档案记 186.8 MiB\n", mib);
        CHECK(false);
    }
    // 口径提醒：若把 28.64 当十进制 GB 换算，同一条记录会得 174.0 MiB（不是 186.8），两者之比
    // 恰好是 2^30/1e9 = 1.0737。这条断言是故意钉住差异，免得日后有人「顺手修正」换算方式而
    // 悄悄改掉全部下游数字。
    const double decimal = 28.64 * 1e9 * (12.74 / 1000.0) * 0.5 / static_cast<double>(kMib);
    if (std::fabs(decimal - 174.0) > 0.05) {
        std::printf("FAIL 十进制换算口径：得 %.3f MiB，应为 174.0\n", decimal);
        CHECK(false);
    }
    if (std::fabs(mib / decimal - 1.073741824) > 1e-6) {
        std::printf("FAIL 两种换算口径之比应为 2^30/1e9：实得 %.9f（%.3f 对 %.3f）\n", mib / decimal,
                    mib, decimal);
        CHECK(false);
    }
    // 零输入不产生预算，也不出 NaN。
    CHECK(step_budget_bytes(ProfileInputs{}) == 0);
    CHECK(step_budget_bytes(ProfileInputs{28.64, 0.0, 0.5}) == 0);
    CHECK(step_budget_bytes(ProfileInputs{28.64, 12.74, 0.0}) == 0);
}

void test_table_demand_matches_profile() {
    // 本模型：一个 token 16 行、IQ4_NL 一行 90 字节（engine §4.1）。一个 token 的表行读合计 1440 字节。
    const TableDemand d{/*rows_per_token=*/16, /*row_bytes=*/90, /*positions_per_token=*/1.13};
    const std::uint64_t bytes = table_demand_bytes(d);
    if (bytes != 1627) {  // llround(16 × 90 × 1.13) = llround(1627.2)
        std::printf("FAIL 表行需求：得 %llu 字节，应 1627\n",
                    static_cast<unsigned long long>(bytes));
        CHECK(false);
    }
    // 档案记 demand_table_mib = 0.00155（三位小数）-> 反推 positions_per_token ∈ [1.125, 1.132]。
    const double mib = to_mib(bytes);
    if (std::fabs(mib - 0.00155) > 0.000005) {
        std::printf("FAIL 表行需求与档案不一致：得 %.6f MiB，档案 0.00155\n", mib);
        CHECK(false);
    }
    bool lo_ok = false, hi_ok = false;
    for (int i = 0; i <= 200; ++i) {
        const double p = 1.120 + 0.0001 * i;
        const double m = to_mib(table_demand_bytes(TableDemand{16, 90, p}));
        if (std::fabs(m - 0.00155) <= 0.000005) {
            if (p < 1.13) lo_ok = true;
            if (p > 1.13) hi_ok = true;
        }
    }
    CHECK(lo_ok && hi_ok);  // 1.13 落在能复现该档案值的区间内部，不是边界
    // 零输入不产生需求。
    CHECK(table_demand_bytes(TableDemand{0, 90, 1.13}) == 0);
    CHECK(table_demand_bytes(TableDemand{16, 0, 1.13}) == 0);
}

void test_demand_to_requests_and_late_count() {
    // 预算取画像里的 186.8 MiB。表行与 KV 都远小于预算，专家需求取画像的静态口径 106.9 MiB。
    const std::uint64_t budget = step_budget_bytes(kEnv2);
    StepDemand d;
    d.step = 7;
    d.kv_bytes = 24 * kMib;  // 与「KV 内存侧读 24.6 MiB（4K 提示）」同量级
    d.table = TableDemand{16, 90, 1.13};
    // 三个候选：概率不同、字节相同（同一层里逐专家档相同）。按概率÷字节排序即按概率排序。
    d.experts = {{/*id=*/11, /*p=*/0.02, /*bytes=*/40 * kMib},
                 {/*id=*/12, /*p=*/0.05, /*bytes=*/40 * kMib},
                 {/*id=*/13, /*p=*/0.01, /*bytes=*/40 * kMib}};

    const std::vector<Request> reqs = build_requests(d);
    CHECK(reqs.size() == 5u);
    CHECK(reqs[0].cls == FlowClass::kKv && reqs[0].must);
    CHECK(reqs[1].cls == FlowClass::kTableRow && reqs[1].must);
    for (std::size_t i = 2; i < reqs.size(); ++i) {
        CHECK(reqs[i].cls == FlowClass::kExpert && !reqs[i].must);
        CHECK(reqs[i].consume_at_step == 7);
    }
    // build_requests 保持候选的输入序（排序由仲裁器做，这里不替它排），键是概率 ÷ 字节。
    CHECK(reqs[2].tag == 11 && reqs[3].tag == 12 && reqs[4].tag == 13);
    CHECK(reqs[2].benefit_per_byte < reqs[3].benefit_per_byte);   // 输入序 p=0.02 < 0.05
    CHECK(reqs[3].benefit_per_byte > reqs[4].benefit_per_byte);   // p=0.05 > 0.01
    const double want_key = 0.05 / static_cast<double>(40 * kMib);
    CHECK(std::fabs(reqs[3].benefit_per_byte - want_key) < 1e-30);

    const Result r = arbitrate(budget, reqs);
    const StepObservation o = observe(r, budget, 7);
    CHECK(o.step == 7 && o.budget_bytes == budget);
    CHECK(o.kv_admitted == 1 && o.table_rows_admitted == 1 && o.experts_admitted == 3);
    CHECK(o.late_requests == 0 && o.late_bytes == 0);
    CHECK(o.overspend_bytes == 0);
    CHECK(o.admitted_bytes == r.admitted_bytes);
    CHECK(std::fabs(o.budget_used - static_cast<double>(o.admitted_bytes) /
                                        static_cast<double>(budget)) < 1e-12);
    CHECK(o.budget_used > 0.5 && o.budget_used < 0.9);  // 24 + 120 MiB 对 186.8 MiB
    // admit_order：必需的两项按类别先后，预测性按概率降序。
    CHECK(r.admit_order.size() == 5u);
    CHECK(r.admit_order[0] == 0 && r.admit_order[1] == 0);  // 两个必需项的 tag 都是 0
    CHECK(r.admit_order[2] == 12 && r.admit_order[3] == 11 && r.admit_order[4] == 13);

    // 预算砍到只装得下两个专家：必需项照进，被放弃的那个记迟到，且**不**影响必需项。
    const std::uint64_t tight = 24 * kMib + 90 * kMib;
    const Result r2 = arbitrate(tight, reqs);
    const StepObservation o2 = observe(r2, tight, 8);
    CHECK(o2.kv_admitted == 1 && o2.table_rows_admitted == 1);
    CHECK(o2.experts_admitted == 2);       // 40×2 + 24 + 0.0016 ≤ 114 MiB < 90+24... 见下
    CHECK(o2.late_requests == 1);
    CHECK(o2.late_bytes == 40 * kMib);
    CHECK(o2.overspend_bytes == 0);
}

void test_must_overspend_does_not_become_late() {
    // 预算远小于必需项：必需项一律准入并记 overspend；预测性全迟到；两笔账不混。
    const std::uint64_t budget = 8 * kMib;
    StepDemand d;
    d.step = 3;
    d.kv_bytes = 24 * kMib;  // 单独就超预算
    d.table = TableDemand{16, 90, 1.13};
    d.experts = {{/*id=*/5, /*p=*/0.9, /*bytes=*/4 * kMib}};

    const std::vector<Request> reqs = build_requests(d);
    const Result r = arbitrate(budget, reqs);
    const StepObservation o = observe(r, budget, 3);
    CHECK(o.kv_admitted == 1 && o.table_rows_admitted == 1);
    CHECK(o.experts_admitted == 0 && o.late_requests == 1 && o.late_bytes == 4 * kMib);
    CHECK(o.overspend_bytes == 24 * kMib + table_demand_bytes(d.table) - budget);
    // 必需项的两笔字节都在 admitted_bytes 里；overspend 只是其中超出预算的那部分。
    CHECK(o.admitted_bytes == 24 * kMib + table_demand_bytes(d.table));
    CHECK(o.admitted_bytes - o.overspend_bytes == budget);
    CHECK(o.budget_used > 1.0);  // 超支时该比值大于 1，是预期的可观测信号
}

void test_ratio_key_and_token_scaling() {
    // 位置放大要真的乘进去：投机窗口下一步要读多个 token 的表行。
    StepDemand d;
    d.step = 2;
    d.table = TableDemand{16, 90, 1.13};
    d.table_tokens = 3;
    const std::vector<Request> reqs = build_requests(d);
    CHECK(reqs.size() == 1u);
    CHECK(reqs[0].bytes == 3 * 1627);

    // 排序键是概率 ÷ 字节，不是概率：字节小的低概率候选应当先于字节大的高概率候选被准入。
    // 这条钉住 /bytes 那一项——若退化成按概率排，本用例会翻。
    StepDemand e;
    e.step = 2;
    e.experts = {{/*id=*/1, /*p=*/0.05, /*bytes=*/50 * kMib},
                 {/*id=*/2, /*p=*/0.04, /*bytes=*/10 * kMib}};
    const std::vector<Request> er = build_requests(e);
    CHECK(er.size() == 2u);
    CHECK(er[0].benefit_per_byte < er[1].benefit_per_byte);  // 0.05/50 < 0.04/10
    const Result r = arbitrate(10 * kMib, er);               // 只装得下小字节的那个
    CHECK(r.admit_order.size() == 1u && r.admit_order[0] == 2);
    CHECK(r.dropped == 1 && r.dropped_bytes == 50 * kMib);
}

void test_empty_and_zero_byte_candidates() {
    StepDemand empty;
    CHECK(build_requests(empty).empty());
    CHECK(observe(Result{}, 0, 0).budget_used == 0.0);  // 预算为 0 记 0，不出 NaN

    // 字节为 0 的候选不占决策位（否则绕过预算的「零字节请求永远准入」会被误用）。
    StepDemand d;
    d.step = 1;
    d.experts = {{1, 0.5, 0 /* bytes = 0 */}, {2, 0.4, 4 * kMib}};
    const std::vector<Request> reqs = build_requests(d);
    CHECK(reqs.size() == 1u);
    CHECK(reqs[0].tag == 2);
}

}  // namespace

int main() {
    test_budget_matches_profile();
    test_table_demand_matches_profile();
    test_demand_to_requests_and_late_count();
    test_must_overspend_does_not_become_late();
    test_ratio_key_and_token_scaling();
    test_empty_and_zero_byte_candidates();
    std::puts("step_budget: profile derivation, demand conversion and per-step observation hold");
    return 0;
}