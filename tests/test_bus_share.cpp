// DDR 总线份额限流器的回归（interfaces.md 第 8.2 节）。不引第三方框架。
//
// 四条：默认 0 即关闭、关闭时一律不受理（含 0 字节）；份额内可以累计用完、刚好等量时仍可；
// 单次超过份额直接拒（不做部分准入）；每步清零后可以重新用。
#include "scheduling/bus_share.hpp"

#include "check.hpp"
#include <cstdint>
#include <cstdio>

using namespace qinfer::scheduling;

namespace {

constexpr std::uint64_t kMib = 1024ull * 1024ull;

void test_disabled_by_default() {
    BusShareLimiter off(0);
    CHECK(!off.enabled());
    CHECK(off.share() == 0);
    CHECK(!off.try_acquire(1));
    CHECK(!off.try_acquire(0));
    CHECK(off.used() == 0);
    CHECK(off.remaining() == 0);
    CHECK(off.refusals() == 2);  // 关闭时两次都记拒收
}

void test_accumulates_up_to_share() {
    BusShareLimiter lim(10 * kMib);
    CHECK(lim.enabled());
    CHECK(lim.try_acquire(4 * kMib));
    CHECK(lim.used() == 4 * kMib);
    CHECK(lim.remaining() == 6 * kMib);
    // 刚好等量：仍应受理（份额是上界，不是排他）。
    CHECK(lim.try_acquire(6 * kMib));
    CHECK(lim.used() == 10 * kMib);
    CHECK(lim.remaining() == 0);
    // 再用 1 字节就超了。
    CHECK(!lim.try_acquire(1));
    CHECK(lim.used() == 10 * kMib);
    CHECK(lim.refusals() == 1);
}

void test_oversized_single_request_is_refused_whole() {
    BusShareLimiter lim(10 * kMib);
    // 单次请求就超过整个份额：不做部分准入，一个字节都不放。
    CHECK(!lim.try_acquire(11 * kMib));
    CHECK(lim.used() == 0);
    CHECK(lim.remaining() == 10 * kMib);
    CHECK(lim.refusals() == 1);
    // 同一份额下，小请求仍然可以走。
    CHECK(lim.try_acquire(8 * kMib));
    CHECK(lim.used() == 8 * kMib);
}

void test_zero_byte_request_does_not_consume_nor_count() {
    BusShareLimiter lim(10 * kMib);
    CHECK(lim.try_acquire(0));  // 没东西要搬，受理
    CHECK(lim.used() == 0);
    CHECK(lim.refusals() == 0);  // 不计入拒收
}

void test_start_step_resets_usage() {
    BusShareLimiter lim(10 * kMib);
    CHECK(lim.try_acquire(10 * kMib));
    CHECK(!lim.try_acquire(1));
    lim.start_step();
    CHECK(lim.used() == 0);
    CHECK(lim.remaining() == 10 * kMib);
    CHECK(lim.try_acquire(10 * kMib));
    CHECK(lim.refusals() == 1);  // 拒收计数不随步清零（它是累计观测量）
}

}  // namespace

int main() {
    test_disabled_by_default();
    test_accumulates_up_to_share();
    test_oversized_single_request_is_refused_whole();
    test_zero_byte_request_does_not_consume_nor_count();
    test_start_step_resets_usage();
    std::puts("bus_share: 份额为 0 即关闭、份额内累计、超量整拒、每步清零");
    return 0;
}