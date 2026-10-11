// 计算组的回归（interfaces.md 第 8 节，决策见 ADR-011）。不引第三方框架。
//
// 三件事：能力表的档掩码与内核表一致；一批的权重字节算得对；选组的三道筛子与两遍扫描。
// 每条筛子都写明「去掉它就会选错」的那个变异体，那正是各自的判据。
#include "experts/compute_group.hpp"

#include "check.hpp"
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

using namespace qinfer::experts;

namespace {

constexpr int kHidden = 256;
constexpr int kFfn = 64;

// 与 test_layer_dispatch 同几何：gate/up 用 IQ2_S（一块 256 值）、down 用 Q2_0（一块 64 值）。
MatrixSpec mat(Format f, std::uint64_t rows, std::uint64_t cols) {
    MatrixSpec m;
    m.format = f;
    m.rows = rows;
    m.cols = cols;
    return m;
}

LayerSpec make_spec() {
    LayerSpec s;
    s.layer = 0;
    s.gate = mat(Format::kIq2S, kFfn, kHidden);
    s.up = mat(Format::kIq2S, kFfn, kHidden);
    s.down = mat(Format::kQ2_0, kHidden, kFfn);
    return s;
}

// 每专家 64×82 + 64×82 + 256×18 = 15104 字节（块字节数按 S-42）。
constexpr std::uint64_t kPerExpert = 64ull * 82 + 64ull * 82 + 256ull * 18;
constexpr std::uint64_t kBatchBytes = 3 * kPerExpert;

ExpertJob make_job(const LayerSpec& spec) {
    ExpertJob j;
    j.spec = &spec;
    j.n = 3;
    j.hidden = kHidden;
    return j;
}

// 档齐、可用、但按 kind 不是 CPU 的组：用它当「非兜底组」。
ComputeGroup other_group(ExpertWorker& w, std::uint32_t formats, qinfer::scheduling::BusShareLimiter* bus) {
    ComputeGroup g;
    g.kind = GroupKind::kIgpu;
    g.tier = MemoryTier::kSharedDdr;
    g.caps.formats = formats;
    g.worker = &w;
    g.bus = bus;
    g.fallback = false;
    return g;
}

void test_formats_mask_matches_kernel_table() {
    const GroupCaps caps{supported_format_mask(), false};
    // 逐档核对：掩码就是「本仓库有内核」这个判据本身，不是另抄一份清单。
    for (int i = 1; i <= static_cast<int>(Format::kIq4Xs); ++i) {
        const Format f = static_cast<Format>(i);
        CHECK(has_format(caps, f) == has_kernel(f));
    }
    // kUnknown 不进任何能力表：把它算进去会让「没有档」的组看起来什么都能接。
    CHECK(!has_format(caps, Format::kUnknown));
    CHECK(has_format(caps, Format::kIq2S));
    CHECK(has_format(caps, Format::kQ2_0));
    CHECK(has_format(caps, Format::kIq4Xs));
    const GroupCaps empty{0, false};
    CHECK(!has_format(empty, Format::kIq2S));
    CHECK(!has_format(empty, Format::kUnknown));
}

void test_unknown_format_is_taken_by_nobody() {
    // 认不出来的档不能被悄悄塞给某个组：连一个声明「什么档都有」的组也得跳过它，否则那一层会拿着
    // 一份含义不明的字节去算，错得还很可信。这条同时钉住 format_bit(kUnknown) == 0。
    LayerSpec spec = make_spec();
    spec.gate = mat(Format::kUnknown, kFfn, kHidden);
    const ExpertJob job = make_job(spec);
    CpuExpertWorker cpu;
    IgpuExpertWorker any(/*device_usable=*/true);
    std::vector<ComputeGroup> groups{other_group(any, ~0u, nullptr), cpu_group(cpu)};

    std::string why;
    const GroupPick pick = select_group(groups, job, why);
    CHECK(pick.group == nullptr);
    CHECK(!why.empty());
}

void test_job_weight_bytes() {
    const LayerSpec spec = make_spec();
    std::uint64_t bytes = 0;
    CHECK(job_weight_bytes(make_job(spec), bytes));
    CHECK(bytes == kBatchBytes);

    // 没有层规格、没有专家：算不出来（持份额限流器的组因此不能被选中）。
    ExpertJob bare;
    bare.n = 3;
    CHECK(!job_weight_bytes(bare, bytes));
    ExpertJob none = make_job(spec);
    none.n = 0;
    CHECK(!job_weight_bytes(none, bytes));

    // 几何不成立（cols 不是块元素数的整数倍）：同样算不出来，不猜。
    LayerSpec bad = make_spec();
    bad.down.cols = kFfn + 1;
    CHECK(!job_weight_bytes(make_job(bad), bytes));
}

void test_skips_unavailable_group() {
    // 变异体：去掉 available() 那道筛子 -> 选中这个组（它档齐、份额也够，只是不可用）。
    const LayerSpec spec = make_spec();
    const ExpertJob job = make_job(spec);
    CpuExpertWorker cpu;
    IgpuExpertWorker igpu(/*device_usable=*/false);
    qinfer::scheduling::BusShareLimiter bus(64ull * 1024 * 1024);
    std::vector<ComputeGroup> groups{other_group(igpu, supported_format_mask(), &bus), cpu_group(cpu)};

    std::string why;
    const GroupPick pick = select_group(groups, job, why);
    CHECK(pick.group != nullptr);
    CHECK(pick.group->kind == GroupKind::kCpu);
    CHECK(pick.booked_bytes == 0);  // CPU 组不持份额限流器
    CHECK(!why.empty());            // 逐组的理由写下来了
    CHECK(bus.used() == 0);
}

void test_skips_group_without_the_format() {
    // 变异体：去掉档那道筛子 -> 选中这个组，它没有 down 那个档的内核。
    const LayerSpec spec = make_spec();
    const ExpertJob job = make_job(spec);
    CpuExpertWorker cpu;
    IgpuExpertWorker partial(/*device_usable=*/true);
    const std::uint32_t only_gate_up = format_bit(Format::kIq2S);
    std::vector<ComputeGroup> groups{other_group(partial, only_gate_up, nullptr), cpu_group(cpu)};

    std::string why;
    const GroupPick pick = select_group(groups, job, why);
    CHECK(pick.group != nullptr);
    CHECK(pick.group->kind == GroupKind::kCpu);
    CHECK(why.find(format_name(Format::kQ2_0)) != std::string::npos);
}

void test_share_refuses_and_books() {
    // 变异体：去掉份额那道筛子 -> 份额装不下也照样选中它。
    const LayerSpec spec = make_spec();
    const ExpertJob job = make_job(spec);
    CpuExpertWorker cpu;
    IgpuExpertWorker igpu(/*device_usable=*/true);

    qinfer::scheduling::BusShareLimiter tight(kBatchBytes - 1);
    std::vector<ComputeGroup> g1{other_group(igpu, supported_format_mask(), &tight), cpu_group(cpu)};
    std::string why;
    const GroupPick p1 = select_group(g1, job, why);
    CHECK(p1.group != nullptr && p1.group->kind == GroupKind::kCpu);
    CHECK(tight.used() == 0);  // 被拒就不记账

    qinfer::scheduling::BusShareLimiter roomy(kBatchBytes);
    std::vector<ComputeGroup> g2{other_group(igpu, supported_format_mask(), &roomy), cpu_group(cpu)};
    const GroupPick p2 = select_group(g2, job, why);
    CHECK(p2.group != nullptr && p2.group->kind == GroupKind::kIgpu);
    CHECK(p2.booked_bytes == kBatchBytes);
    CHECK(roomy.used() == kBatchBytes);  // 选中即记账；退回是调用方的事（layer_dispatch）
}

void test_fallback_is_last_resort() {
    // 变异体：只扫一遍、按列表顺序取 -> 列表里 CPU 组在前时会选中它，而不是先试非兜底组。
    const LayerSpec spec = make_spec();
    const ExpertJob job = make_job(spec);
    CpuExpertWorker cpu;
    IgpuExpertWorker igpu(/*device_usable=*/true);
    std::vector<ComputeGroup> groups{cpu_group(cpu), other_group(igpu, supported_format_mask(), nullptr)};

    std::string why;
    const GroupPick pick = select_group(groups, job, why);
    CHECK(pick.group != nullptr);
    CHECK(pick.group->kind == GroupKind::kIgpu);

    // 只有兜底组时，仍然接得住。
    std::vector<ComputeGroup> only_fallback{cpu_group(cpu)};
    const GroupPick p2 = select_group(only_fallback, job, why);
    CHECK(p2.group != nullptr && p2.group->kind == GroupKind::kCpu);

    // 一组都接不住：返回空，且理由非空（不静默）。
    IgpuExpertWorker no_kernel(/*device_usable=*/true);
    std::vector<ComputeGroup> nothing{other_group(no_kernel, 0, nullptr)};
    const GroupPick p3 = select_group(nothing, job, why);
    CHECK(p3.group == nullptr);
    CHECK(!why.empty());
    CHECK(why.find("igpu") != std::string::npos);
}

void test_cpu_and_igpu_group_defaults() {
    CpuExpertWorker cpu;
    const ComputeGroup c = cpu_group(cpu);
    CHECK(c.kind == GroupKind::kCpu);
    CHECK(c.fallback);
    CHECK(c.bus == nullptr);  // 那条总线上的余量是它的，不需要限自己
    CHECK(c.caps.formats == supported_format_mask());

    IgpuExpertWorker igpu(true);
    qinfer::scheduling::BusShareLimiter bus(1024);
    const ComputeGroup g = igpu_group(igpu, bus);
    CHECK(g.kind == GroupKind::kIgpu);
    CHECK(!g.fallback);
    CHECK(g.bus == &bus);
    CHECK(g.caps.formats == 0);      // 核显内核尚未接入
    CHECK(!g.caps.integer_dot);      // 192 项扩展里没有整数点积指令
    // 份额为 0 即不启用：这一个不进候选（可用性也据此返回假）。
    qinfer::scheduling::BusShareLimiter off(0);
    const ComputeGroup g_off = igpu_group(igpu, off);
    std::string why;
    CHECK(!g_off.available(why));
    CHECK(!why.empty());
    CHECK(g.available(why));  // 份额给了就可用（能不能算另说，那是档掩码的事）
}

}  // namespace

int main() {
    test_formats_mask_matches_kernel_table();
    test_unknown_format_is_taken_by_nobody();
    test_job_weight_bytes();
    test_skips_unavailable_group();
    test_skips_group_without_the_format();
    test_share_refuses_and_books();
    test_fallback_is_last_resort();
    test_cpu_and_igpu_group_defaults();
    std::puts("compute_group: 档掩码与内核表一致、字节算得对、三道筛子与两遍扫描都成立");
    return 0;
}
