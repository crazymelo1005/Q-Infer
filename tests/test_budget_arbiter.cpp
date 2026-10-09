// 字节预算仲裁器的行为回归。不引第三方框架：assert 失败即中止，ctest 据此判成败。
//
// 规则出自 interfaces.md 第 3 节与 engine.md §3.2：必需请求按类别优先（KV 先于表行）占用预算，
// 但绝不因为预算不足被丢弃——它在消费点阻塞，仲裁器只把超出部分记成 overspend；预测性请求按
// 「收益 ÷ 字节」降序贪心准入，放不下的弃掉并计入迟到，且不因为前一个放不下就停止尝试。
#include "scheduling/budget_arbiter.hpp"

#include <cassert>
#include <cstdio>
#include <vector>

using namespace qinfer::scheduling;

namespace {

Request kv(std::uint64_t bytes, std::uint32_t tag) {
    Request r;
    r.cls = FlowClass::kKv;
    r.bytes = bytes;
    r.must = true;
    r.tag = tag;
    return r;
}

Request table_row(std::uint64_t bytes, std::uint32_t tag) {
    Request r;
    r.cls = FlowClass::kTableRow;
    r.bytes = bytes;
    r.must = true;
    r.tag = tag;
    return r;
}

Request expert(std::uint64_t bytes, double benefit_per_byte, std::uint32_t tag) {
    Request r;
    r.cls = FlowClass::kExpert;
    r.bytes = bytes;
    r.must = false;
    r.benefit_per_byte = benefit_per_byte;
    r.tag = tag;
    return r;
}

const Decision& decision_of(const Result& res, std::uint32_t tag) {
    for (const Decision& d : res.decisions) {
        if (d.tag == tag) return d;
    }
    assert(false && "tag not found in decisions");
    return res.decisions.front();
}

bool admitted(const Result& res, std::uint32_t tag) {
    return decision_of(res, tag).verdict == Verdict::kAdmitted;
}

void test_all_fit() {
    std::vector<Request> reqs{kv(100'000, 1), table_row(90, 2),
                              expert(200'000, 0.001, 3), expert(200'000, 0.001, 4)};
    const Result res = arbitrate(1'000'000, reqs);

    assert(res.decisions.size() == reqs.size());
    assert(admitted(res, 1) && admitted(res, 2) && admitted(res, 3) && admitted(res, 4));
    assert(res.admitted_bytes == 500'090);
    assert(res.overspend_bytes == 0);
    assert(res.dropped == 0 && res.dropped_bytes == 0);
    assert(res.admit_order.size() == 4);
}

void test_must_is_never_dropped_even_over_budget() {
    std::vector<Request> reqs{kv(8'000, 1), table_row(5'000, 2), expert(1, 1.0, 3)};
    const Result res = arbitrate(10'000, reqs);

    assert(admitted(res, 1) && admitted(res, 2));       // 必需的照进，超出部分记账
    assert(res.overspend_bytes == 3'000);
    assert(!admitted(res, 3));                          // 预算已被必需项用尽，预测性全弃
    assert(res.dropped == 1 && res.dropped_bytes == 1);
}

void test_required_priority_is_kv_then_table_row() {
    // 输入顺序刻意让表行在前；准入顺序必须是 KV 先。
    std::vector<Request> reqs{table_row(40, 7), kv(60, 9)};
    const Result res = arbitrate(1'000, reqs);

    assert(res.admit_order.size() == 2);
    assert(res.admit_order[0] == 9);
    assert(res.admit_order[1] == 7);
}

void test_predictive_sorted_by_benefit_per_byte() {
    // 预算 50：小的高比值请求先拿，大而低比值的放不下就被弃。
    std::vector<Request> reqs{expert(60, 0.01, 1), expert(30, 0.03, 2)};
    const Result res = arbitrate(50, reqs);

    assert(admitted(res, 2));
    assert(!admitted(res, 1));
    assert(res.admit_order.size() == 1 && res.admit_order[0] == 2);
    assert(res.admitted_bytes == 30);
    assert(res.dropped == 1 && res.dropped_bytes == 60);
}

void test_greedy_keeps_trying_after_a_drop() {
    // 第一个（比值最高）放不下，不能因此停止；后面更小的要能进。
    std::vector<Request> reqs{expert(100, 1.0, 1), expert(20, 0.5, 2)};
    const Result res = arbitrate(50, reqs);

    assert(!admitted(res, 1));
    assert(admitted(res, 2));
    assert(res.admitted_bytes == 20);
    assert(res.dropped == 1);
}

void test_equal_ratio_breaks_ties_by_input_order() {
    std::vector<Request> reqs{expert(20, 0.5, 11), expert(20, 0.5, 12), expert(20, 0.5, 13)};
    const Result res = arbitrate(40, reqs);

    assert(res.admit_order.size() == 2);
    assert(res.admit_order[0] == 11 && res.admit_order[1] == 12);
    assert(!admitted(res, 13));
}

void test_zero_budget_admits_only_zero_byte_requests() {
    std::vector<Request> reqs{kv(0, 1), expert(1, 1.0, 2)};
    const Result res = arbitrate(0, reqs);

    assert(admitted(res, 1));
    assert(!admitted(res, 2));
    assert(res.overspend_bytes == 0);
}

}  // namespace

int main() {
    test_all_fit();
    test_must_is_never_dropped_even_over_budget();
    test_required_priority_is_kv_then_table_row();
    test_predictive_sorted_by_benefit_per_byte();
    test_greedy_keeps_trying_after_a_drop();
    test_equal_ratio_breaks_ties_by_input_order();
    test_zero_budget_admits_only_zero_byte_requests();
    std::puts("budget_arbiter: all rules hold");
    return 0;
}
