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
#include "artifact/gguf_table.hpp"
#include "dense/step_loop.hpp"
#include "experts/cuda/cuda_worker.hpp"
#include "experts/cuda/expert_ffn_cuda.hpp"
#include "experts/layer_dispatch.hpp"
#include "experts/moe_layer.hpp"
#include "scheduling/thread_pool.hpp"

#include "check.hpp"
#include "kernels/q8_0.hpp"
#include "kernels/q8k.hpp"
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace qinfer::experts;

namespace {

#include "iq1m_oracle_data.inc"
#include "iq2s_oracle_data.inc"
#include "iq2xxs_oracle_data.inc"
#include "q2_0_oracle_data.inc"

constexpr int kHidden = 256;
constexpr int kFfn = 64;

// 一个专家的三矩阵字节数（gate 64×82 + up 64×82 + down 256×18）。
constexpr std::uint64_t kPerExpertBytes =
    static_cast<std::uint64_t>(kFfn) * 82 * 2 + static_cast<std::uint64_t>(kHidden) * 18;

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

// gate/up 的档可换（部署那份 gate/up 是三档混的，故三档都要在「整条前馈」里过一遍）。
struct GuFormat {
    Format fmt;
    int block_bytes;
    const char* hex;
};

// 三个档各自的块字节数（256 值一块）与真模型 oracle。
const GuFormat kGuFormats[3] = {{Format::kIq2S, 82, kIq2sRealHex},
                                {Format::kIq2Xxs, 66, kIq2xxsRealHex},
                                {Format::kIq1M, 56, kIq1mRealHex}};

ExpertBytes make_expert(int salt, const GuFormat& gu = kGuFormats[0]) {
    const std::vector<std::uint8_t> wsrc = from_hex(gu.hex);
    const std::vector<std::uint8_t> q20 = from_hex(kQ20RealHex);
    ExpertBytes e;
    e.gate = tile(wsrc, gu.block_bytes, static_cast<std::size_t>(kFfn) * gu.block_bytes);
    e.up = tile(wsrc, gu.block_bytes, static_cast<std::size_t>(kFfn) * gu.block_bytes);
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

void test_device_matches_host(GuFormat gu) {
    LayerSpec spec = make_spec();
    spec.gate.format = gu.fmt;
    spec.up.format = gu.fmt;
    const std::vector<ExpertBytes> experts{make_expert(1, gu), make_expert(2, gu)};
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
    const std::uint64_t per_expert = static_cast<std::uint64_t>(kFfn) *
                                         static_cast<std::uint64_t>(gu.block_bytes) * 2 +
                                     static_cast<std::uint64_t>(kHidden) * 18;
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
    std::printf("  gate/up = %s：设备 vs 主机逐元素位相同 %d/%zu，L1 相对差 %.3g\n",
                format_name(gu.fmt), same, got.size(), rel);
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

void test_worker_computes_a_job_and_stays_resident() {
    const LayerSpec spec = make_spec();
    const std::vector<ExpertBytes> experts{make_expert(7), make_expert(8)};
    const std::vector<float> x = make_x();
    const std::vector<int> ids{5, 4};  // 刻意与下标不同序：设备侧按 id 认常驻，不看位置

    std::vector<ExpertWeights> views{experts[0].view(), experts[1].view()};
    std::vector<float> host(experts.size() * kHidden, 0.0f);
    for (std::size_t e = 0; e < experts.size(); ++e) {
        FfnScratch sc;
        std::string err;
        must(expert_ffn(spec, experts[e].view(), x.data(), sc, host.data() + e * kHidden, err), err,
             "主机参考");
    }

    CudaExpertWorker worker(0, /*slots_per_layer=*/4);
    CHECK(worker.available());
    ExpertJob job;
    job.spec = &spec;
    job.experts = views.data();
    job.ids = ids.data();
    job.n = 2;
    job.x = x.data();
    job.hidden = kHidden;
    std::vector<float> got(experts.size() * kHidden, 0.0f);
    job.out = got.data();

    std::string err;
    must(worker.submit(job, err), err, "submit");
    CHECK(worker.wait(err));
    const std::uint64_t after = worker.uploaded_bytes();
    CHECK(after == 2 * kPerExpertBytes);          // 只搬了两个专家
    CHECK(worker.resident_bytes() == 4 * kPerExpertBytes);  // 常驻区是按槽位数开的（4 个）
    must(worker.submit(job, err), err, "第二次 submit（两个专家都已在常驻区）");
    CHECK(worker.uploaded_bytes() == after);
    CHECK(worker.stats().jobs == 2 && worker.stats().experts == 4 && worker.stats().failed == 0);

    int same = 0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        if (std::bit_cast<std::uint32_t>(got[i]) == std::bit_cast<std::uint32_t>(host[i])) ++same;
    }
    std::printf("  工人算出来的一批：位相同 %d/%zu\n", same, got.size());
    CHECK(same == static_cast<int>(got.size()));

    // 第三个：换一个没见过的专家 id——必须真的把它搬上来。这条钉的是「按 id 认常驻，不是按位置」：
    // 若按位置认，id 6 落在第 0 个位置，工人会以为 0 号槽就是它而不搬，于是算出 5 号专家的结果。
    const ExpertBytes extra = make_expert(11);
    const ExpertWeights extra_view = extra.view();
    const std::vector<int> ids3{6};
    std::vector<float> got3(kHidden, 0.0f);
    ExpertJob job3;
    job3.spec = &spec;
    job3.experts = &extra_view;
    job3.ids = ids3.data();
    job3.n = 1;
    job3.x = x.data();
    job3.hidden = kHidden;
    job3.out = got3.data();
    must(worker.submit(job3, err), err, "第三个专家（新 id）");
    CHECK(worker.uploaded_bytes() > after);

    FfnScratch sc3;
    std::vector<float> want3(kHidden, 0.0f);
    must(expert_ffn(spec, extra_view, x.data(), sc3, want3.data(), err), err, "主机参考（第三个）");
    for (int i = 0; i < kHidden; ++i) {
        CHECK(std::bit_cast<std::uint32_t>(got3[static_cast<std::size_t>(i)]) ==
              std::bit_cast<std::uint32_t>(want3[static_cast<std::size_t>(i)]));
    }
}

void test_select_group_prefers_the_cuda_group() {
    const LayerSpec spec = make_spec();
    CudaExpertWorker worker(0, 4);
    IgpuExpertWorker igpu(/*device_usable=*/false);
    qinfer::scheduling::BusShareLimiter off(0);
    CpuExpertWorker cpu;
    std::vector<ComputeGroup> groups{igpu_group(igpu, off), cuda_group(worker), cpu_group(cpu)};

    ExpertJob job;
    job.spec = &spec;
    job.n = 2;
    job.hidden = kHidden;
    std::string why;
    const GroupPick pick = select_group(groups, job, why);
    CHECK(pick.group != nullptr);
    CHECK(pick.group->kind == GroupKind::kCuda);
}

void test_unsupported_role_is_refused_by_the_worker() {
    // down 换成 IQ2_S：角色不对（它当 down 时的激活搭档是 Q8_K，设备侧只做了 Q8_0 的 h 量化器）。
    // 组的档掩码会放它进来（IQ2_S 确实是设备侧有内核的档），真正挡住它的是工人的 init——
    // 这正是 §8.5 那条「改派给兜底组」的触发路径。
    LayerSpec bad = make_spec();
    bad.down.format = Format::kIq2S;
    bad.down.cols = kFfn;
    const std::vector<ExpertBytes> experts{make_expert(9)};
    std::vector<ExpertWeights> views{experts[0].view()};
    const std::vector<int> ids{1};
    const std::vector<float> x = make_x();
    std::vector<float> out(kHidden, 0.0f);

    CudaExpertWorker worker(0, 2);
    ExpertJob job;
    job.spec = &bad;
    job.experts = views.data();
    job.ids = ids.data();
    job.n = 1;
    job.x = x.data();
    job.hidden = kHidden;
    job.out = out.data();
    std::string err;
    CHECK(!worker.submit(job, err));
    CHECK(!err.empty());
    CHECK(worker.stats().failed == 1);
}

int count_cards() {
    int n = 0;
    for (int d = 0; d < 4; ++d) {
        CudaExpertWorker probe(d, 1);
        if (probe.available()) ++n;
    }
    return n;
}

void test_owner_split() {
    // 连续切分：48 层两张卡 → 24/24；10 层三张卡 → 4/3/3；一张卡全给 0。越界按最后一段算。
    CHECK(cuda_owner_of(2, 48, 0) == 0);
    CHECK(cuda_owner_of(2, 48, 23) == 0);
    CHECK(cuda_owner_of(2, 48, 24) == 1);
    CHECK(cuda_owner_of(2, 48, 47) == 1);
    CHECK(cuda_owner_of(1, 48, 47) == 0);
    CHECK(cuda_owner_of(3, 10, 0) == 0 && cuda_owner_of(3, 10, 3) == 0);
    CHECK(cuda_owner_of(3, 10, 4) == 1 && cuda_owner_of(3, 10, 6) == 1);
    CHECK(cuda_owner_of(3, 10, 7) == 2 && cuda_owner_of(3, 10, 9) == 2);
    CHECK(cuda_owner_of(2, 48, -5) == 0);   // 越界按边界算，不返回 -1
    CHECK(cuda_owner_of(2, 48, 99) == 1);
    std::printf("  层切分：48 层/2 卡 → 0..23 / 24..47；10 层/3 卡 → 4/3/3\n");
}

void test_fleet_and_each_card_runs() {
    const int cards = count_cards();
    std::printf("  这台机器可见的卡：%d\n", cards);
    if (cards < 2) {
        std::printf("  只有一张卡，跳过按卡分工的用例\n");
        return;
    }
    const LayerSpec spec = make_spec();
    const std::vector<float> x = make_x();
    CpuExpertWorker cpu;
    CudaGroupFleet fleet;
    std::string err;
    must(fleet.init({0, 1}, /*slots_per_layer=*/4, cpu, err), err, "fleet.init");
    CHECK(fleet.card_count() == 2);
    CHECK(fleet.device_of(0) == 0 && fleet.device_of(1) == 1);
    for (int owner = 0; owner < 2; ++owner) {
        const std::vector<ComputeGroup>& cand = fleet.candidates_for(owner);
        CHECK(cand.size() == 2);
        CHECK(cand[0].kind == GroupKind::kCuda);      // 本卡的组在前
        CHECK(!cand[0].fallback);
        CHECK(cand[1].fallback);                      // CPU 兜底组在后
        CHECK(cand[0].worker == &fleet.worker(owner));  // 各卡一份，不共用
    }
    CHECK(fleet.candidates_for(0)[0].worker != fleet.candidates_for(1)[0].worker);

    // 每张卡都要真的算得对：同一层同一批专家分别交给两张卡，结果都该与主机逐位一致。
    for (int owner = 0; owner < 2; ++owner) {
        const ExpertBytes e = make_expert(20 + owner);
        const ExpertWeights view = e.view();
        const std::vector<int> ids{3};
        std::vector<float> got(kHidden, 0.0f);
        ExpertJob job;
        job.spec = &spec;
        job.experts = &view;
        job.ids = ids.data();
        job.n = 1;
        job.x = x.data();
        job.hidden = kHidden;
        job.out = got.data();
        ExpertWorker& w = fleet.worker(owner);
        must(w.submit(job, err), err, "submit（按卡）");
        FfnScratch sc;
        std::vector<float> want(kHidden, 0.0f);
        must(expert_ffn(spec, view, x.data(), sc, want.data(), err), err, "主机参考（按卡）");
        for (int i = 0; i < kHidden; ++i) {
            CHECK(std::bit_cast<std::uint32_t>(got[static_cast<std::size_t>(i)]) ==
                  std::bit_cast<std::uint32_t>(want[static_cast<std::size_t>(i)]));
        }
        std::printf("  卡 %d：一个专家的前馈与主机逐位一致（它算了 %llu 个专家）\n",
                    fleet.device_of(owner), (unsigned long long) w.stats().experts);
    }
}

// ---- 真模型入口 ----
//
// 一层 MoE 的两条路：不交出去（逐专家主机路径）与交出去（候选组 + 批量分派 → 设备侧工人）。
// 两条路的归并顺序相同，故结果该逐位一样。同时要求 `rs.routed == 1`——真的由选中的组做完，
// 而不是被静默改派给 CPU；不然这条对照是空转。
int manual_model_entry(const char* model, int layer) {
    qinfer::artifact::GgufFile g;
    std::string err;
    if (!g.open(model, err)) {
        std::printf("打不开模型：%s（%s）\n", model, err.c_str());
        return 1;
    }
    MoeLayer ml;
    if (!build_moe_layer(g, layer, /*top_k=*/10, ml, err)) {
        std::printf("构建层 %d 的 MoE 失败：%s\n", layer, err.c_str());
        return 1;
    }
    const std::size_t hidden = static_cast<std::size_t>(ml.hidden);
    std::vector<float> x(hidden);
    for (std::size_t i = 0; i < hidden; ++i) {
        x[i] = static_cast<float>(static_cast<int>(i % 31) - 15) * 0.0625f;
    }

    MoeScratch s_host;
    std::vector<float> out_host(hidden, 0.0f);
    std::vector<int> ids(static_cast<std::size_t>(ml.top_k), 0);
    std::vector<float> weights(static_cast<std::size_t>(ml.top_k), 0.0f);
    if (!run_moe_layer(g, ml, x.data(), out_host.data(), s_host, err, ids.data(), weights.data())) {
        std::printf("主机路径失败：%s\n", err.c_str());
        return 1;
    }
    // 主机那条 MoE 的用时（先热身一遍再计，免得把读盘算进去）——把「一步 54 ms/层」拆开要用它。
    double host_moe_ms = 0.0;
    for (int rep = 0; rep < 2; ++rep) {
        const auto a = std::chrono::steady_clock::now();
        if (!run_moe_layer(g, ml, x.data(), out_host.data(), s_host, err, nullptr, nullptr)) {
            std::printf("主机路径热身失败：%s\n", err.c_str());
            return 1;
        }
        const auto b = std::chrono::steady_clock::now();
        host_moe_ms = std::chrono::duration<double, std::milli>(b - a).count();
    }

    CudaExpertWorker worker(0, /*slots_per_layer=*/32);
    if (!worker.available()) {
        std::printf("没有可用的 CUDA 设备，设备路径没法跑\n");
        return 1;
    }
    CpuExpertWorker cpu;
    std::vector<ComputeGroup> groups{cuda_group(worker), cpu_group(cpu)};
    ExpertSource src(g, "blk." + std::to_string(layer) + ".ffn_", ml.spec,
                     qinfer::storage::PolicyKind::kLru, /*slots=*/64, /*ways=*/8, /*budget=*/0);
    LayerDispatchScratch ds;
    RerouteStats rs{};
    MoeOffload off{&groups, &src, &ds, &rs};
    MoeScratch s_dev;
    std::vector<float> out_dev(hidden, 0.0f);
    const auto d0 = std::chrono::steady_clock::now();
    if (!run_moe_layer(g, ml, x.data(), out_dev.data(), s_dev, err, nullptr, nullptr, &off)) {
        std::printf("设备路径失败：%s\n", err.c_str());
        return 1;
    }
    const auto d1 = std::chrono::steady_clock::now();
    std::printf("  单层 MoE 用时：主机 %.2f ms、设备 %.2f ms（设备侧含首次分配与上传）\n",
                host_moe_ms, std::chrono::duration<double, std::milli>(d1 - d0).count());

    int same = 0;
    double mag = 0.0;
    for (std::size_t i = 0; i < hidden; ++i) {
        if (std::bit_cast<std::uint32_t>(out_host[i]) == std::bit_cast<std::uint32_t>(out_dev[i])) {
            ++same;
        }
        mag += std::fabs(static_cast<double>(out_dev[i]));
    }
    const bool bitwise = same == static_cast<int>(hidden);
    std::printf("层 %d：top-%d（%s），两条路逐位一致：%s（位相同 %d/%zu，|设备输出| 合计 %.6g）\n",
                layer, ml.top_k, ml.shared.present ? "含共享专家" : "无共享专家",
                bitwise ? "是" : "否", same, hidden, mag);
    std::printf("  选组 routed=%llu rerouted=%llu；工人 jobs=%llu experts=%llu failed=%llu；常驻 %.3f MiB\n",
                (unsigned long long) rs.routed, (unsigned long long) rs.rerouted,
                (unsigned long long) worker.stats().jobs,
                (unsigned long long) worker.stats().experts,
                (unsigned long long) worker.stats().failed,
                static_cast<double>(worker.resident_bytes()) / (1024.0 * 1024.0));
    if (!bitwise) return 1;
    if (rs.routed != 1 || rs.rerouted != 0) {
        std::printf("  选组不对：这一批该由选中的组做完\n");
        return 1;
    }
    return 0;
}

// 整模型一步：不交出去（逐层主机路径）与交出去（48 层逐层交给 CUDA 组）两条路的 logits 该逐位
// 一致。这条是「设备侧那批真的进了整模型一步」的判据，且要求没有一次重派（否则有层没被接住）。
int manual_step_entry(const char* model, int layers, bool fleet, int tokens, int threads) {
    qinfer::artifact::GgufFile g;
    std::string err;
    if (!g.open(model, err)) {
        std::printf("打不开模型：%s（%s）\n", model, err.c_str());
        return 1;
    }
    // 真模型几何，与 test_step_loop 的 --model 入口同一套。层数可截短，用来二分找第一处分歧。
    qinfer::dense::StepConfig cfg;
    cfg.n_embd = 2560;
    cfg.hc = 4;
    cfg.hc_lr = 320;
    cfg.n_layers = layers;
    cfg.top_k = 10;
    cfg.gdn_head_dim = 128;
    cfg.qsa_head_dim = 256;
    cfg.n_rot = 64;
    cfg.idx_dim = 128;
    cfg.idx_top_k = 2048;
    cfg.idx_block = 4;
    qinfer::dense::StepModel m;
    if (!qinfer::dense::build_step_model(g, cfg, m, err)) {
        std::printf("建整模型失败：%s\n", err.c_str());
        return 1;
    }
    std::vector<float> x(2560);
    for (std::size_t i = 0; i < x.size(); ++i) {
        x[i] = 0.05f * static_cast<float>(static_cast<int>(i % 17) - 8);
    }

    // 先跑一遍当热身：把专家字节带进页缓存。不然第一次跑会把「读盘」算进计时的分母里。
    {
        qinfer::dense::StepScratch warm;
        if (!qinfer::dense::prepare_step_scratch(m, 64, 0, warm, err) ||
            !qinfer::dense::run_step_model(g, m, x.data(), warm, err)) {
            std::printf("热身那一遍跑不动：%s\n", err.c_str());
            return 1;
        }
    }
    qinfer::dense::StepScratch s_ref;
    if (!qinfer::dense::prepare_step_scratch(m, 64, 0, s_ref, err)) {
        std::printf("预备 scratch 失败：%s\n", err.c_str());
        return 1;
    }
    // 连跑若干步、逐步计时：设备侧第 0 步要把权重搬上去，只有后面的步才反映稳态。
    std::vector<double> host_ms;
    std::vector<std::vector<float>> host_logits;
    for (int t = 0; t < tokens; ++t) {
        const auto a = std::chrono::steady_clock::now();
        if (!qinfer::dense::run_step_model(g, m, x.data(), s_ref, err)) {
            std::printf("主机路径跑不动（第 %d 步）：%s\n", t, err.c_str());
            return 1;
        }
        const auto b = std::chrono::steady_clock::now();
        host_ms.push_back(std::chrono::duration<double, std::milli>(b - a).count());
        host_logits.push_back(s_ref.head.logits);
    }
    {
        const qinfer::dense::LayerPhases ph = qinfer::dense::sum_phases(s_ref.phases);
        std::printf("  主机单线程相位（%d 层、最后一步）：load %.0f / gr_read %.0f / attn %.0f / "
                    "moe %.0f / gr_write %.0f ms\n",
                    cfg.n_layers, ph.load_ms, ph.gr_read_ms, ph.attn_ms, ph.moe_ms, ph.gr_write_ms);
    }

    // 若给了 --threads：主机路径再跑一遍（换一份 scratch、开线程池），比用时与逐位结果。
    if (threads > 1) {
        qinfer::dense::StepScratch s_mt;
        if (!qinfer::dense::prepare_step_scratch(m, 64, 0, s_mt, err)) {
            std::printf("预备 scratch（多线程）失败：%s\n", err.c_str());
            return 1;
        }
        std::vector<double> mt_ms;
        std::vector<std::vector<float>> mt_logits;
        qinfer::scheduling::set_pool_threads(threads);
        for (int t = 0; t < tokens; ++t) {
            const auto a = std::chrono::steady_clock::now();
            if (!qinfer::dense::run_step_model(g, m, x.data(), s_mt, err)) {
                qinfer::scheduling::set_pool_threads(0);
                std::printf("主机 %d 线程跑不动（第 %d 步）：%s\n", threads, t, err.c_str());
                return 1;
            }
            const auto b = std::chrono::steady_clock::now();
            mt_ms.push_back(std::chrono::duration<double, std::milli>(b - a).count());
            mt_logits.push_back(s_mt.head.logits);
        }
        qinfer::scheduling::set_pool_threads(0);
        {
            const qinfer::dense::LayerPhases ph = qinfer::dense::sum_phases(s_mt.phases);
            std::printf("  主机 %d 线程相位（%d 层、最后一步）：load %.0f / gr_read %.0f / attn %.0f / "
                        "moe %.0f / gr_write %.0f ms\n",
                        threads, cfg.n_layers, ph.load_ms, ph.gr_read_ms, ph.attn_ms, ph.moe_ms,
                        ph.gr_write_ms);
        }
        std::printf("  主机 %d 线程（同一份 x、同一份状态推进）\n", threads);
        for (int t = 0; t < tokens; ++t) {
            int same = 0;
            const std::vector<float>& A = host_logits[static_cast<std::size_t>(t)];
            const std::vector<float>& B = mt_logits[static_cast<std::size_t>(t)];
            for (std::size_t i = 0; i < A.size() && i < B.size(); ++i) {
                if (std::bit_cast<std::uint32_t>(A[i]) == std::bit_cast<std::uint32_t>(B[i])) {
                    ++same;
                }
            }
            std::printf("   步 %d：单线程 %.0f ms、%d 线程 %.0f ms（%.2f×），logits 位相同 %d/%zu\n",
                        t, host_ms[static_cast<std::size_t>(t)], threads,
                        mt_ms[static_cast<std::size_t>(t)],
                        mt_ms[static_cast<std::size_t>(t)] / host_ms[static_cast<std::size_t>(t)],
                        same, A.size());
        }
    }

    CpuExpertWorker cpu;
    std::vector<ComputeGroup> one_group;
    // 槽位要按「这几步会碰到的不同专家数」给足：一步 10 个，3 步最多 30 个。给不够不会错（按 §8.5
    // 退回 CPU），但那样量出来的就不是设备路径了。真正的答案是专家缓存那套淘汰与配额（interfaces §5）。
    constexpr int kSlots = 32;
    CudaExpertWorker single_worker(0, kSlots);
    CudaGroupFleet dev_fleet;
    if (fleet) {
        if (!dev_fleet.init({0, 1}, kSlots, cpu, err)) {
            std::printf("按卡分工起不来：%s\n", err.c_str());
            return 1;
        }
    } else {
        if (!single_worker.available()) {
            std::printf("没有可用的 CUDA 设备\n");
            return 1;
        }
        one_group = {cuda_group(single_worker), cpu_group(cpu)};
    }
    std::vector<ExpertSource> sources;
    std::vector<MoeOffload> offloads;
    sources.reserve(static_cast<std::size_t>(cfg.n_layers));
    offloads.reserve(static_cast<std::size_t>(cfg.n_layers));
    for (int l = 0; l < cfg.n_layers; ++l) {
        sources.emplace_back(g, "blk." + std::to_string(l) + ".ffn_", m.blocks[l].moe.spec,
                             qinfer::storage::PolicyKind::kFifo, /*slots=*/16, /*ways=*/4,
                             /*budget=*/0);
    }
    LayerDispatchScratch ds;
    RerouteStats rs{};
    for (int l = 0; l < cfg.n_layers; ++l) {
        // 单卡：所有层用同一份候选表；按卡分工：按层切分取本卡那份。
        const std::vector<ComputeGroup>* cand =
            fleet ? &dev_fleet.candidates_for(dev_fleet.owner_of(cfg.n_layers, l)) : &one_group;
        offloads.push_back(MoeOffload{cand, &sources[static_cast<std::size_t>(l)], &ds, &rs});
    }
    cfg.offload_layers = &offloads;

    qinfer::dense::StepModel m2;
    if (!qinfer::dense::build_step_model(g, cfg, m2, err)) {
        std::printf("建整模型（含交出表）失败：%s\n", err.c_str());
        return 1;
    }
    qinfer::dense::StepScratch s_dev;
    if (!qinfer::dense::prepare_step_scratch(m2, 64, 0, s_dev, err)) {
        std::printf("预备 scratch（设备路径）失败：%s\n", err.c_str());
        return 1;
    }
    std::vector<double> dev_ms;
    std::vector<std::vector<float>> dev_logits;
    // 与主机多线程那条同口径：稠密部分也开池子，这样两边差的只是「谁算这批专家」。
    if (threads > 1) qinfer::scheduling::set_pool_threads(threads);
    for (int t = 0; t < tokens; ++t) {
        const auto a = std::chrono::steady_clock::now();
        if (!qinfer::dense::run_step_model(g, m2, x.data(), s_dev, err)) {
            std::printf("设备路径跑不动（第 %d 步）：%s\n", t, err.c_str());
            return 1;
        }
        const auto b = std::chrono::steady_clock::now();
        dev_ms.push_back(std::chrono::duration<double, std::milli>(b - a).count());
        dev_logits.push_back(s_dev.head.logits);
    }
    qinfer::scheduling::set_pool_threads(0);
    std::printf("  逐步用时（同一份 x，共 %d 步；第 0 步含设备侧首次分配与逐层上传%s）\n", tokens,
                threads > 1 ? "；与主机多线程同口径，稠密都开池子" : "");
    for (int t = 0; t < tokens; ++t) {
        int same = 0;
        for (std::size_t i = 0; i < host_logits[static_cast<std::size_t>(t)].size() &&
                                i < dev_logits[static_cast<std::size_t>(t)].size();
             ++i) {
            if (std::bit_cast<std::uint32_t>(host_logits[static_cast<std::size_t>(t)][i]) ==
                std::bit_cast<std::uint32_t>(dev_logits[static_cast<std::size_t>(t)][i])) {
                ++same;
            }
        }
        std::printf("   步 %d：主机 %.0f ms、设备 %.0f ms（%.2f×），logits 位相同 %d/%zu\n", t,
                    host_ms[static_cast<std::size_t>(t)], dev_ms[static_cast<std::size_t>(t)],
                    dev_ms[static_cast<std::size_t>(t)] / host_ms[static_cast<std::size_t>(t)],
                    same, host_logits[static_cast<std::size_t>(t)].size());
    }

    // 判据：设备路径按 ADR-009/§8.6 是**分布级**的（每层的 h 量化在舍入边界上可能与主机差一个
    // 量化步，48 层把这点差放大成处处不同），故这里量出相对差并要求它小，同时把位相同的个数报出来
    // ——不假装逐位一致，也不因为不逐位就判它错。
    const std::vector<float>& a = s_ref.head.logits;
    const std::vector<float>& b = s_dev.head.logits;
    int same = 0;
    double mag_ref = 0.0, mag_dev = 0.0, l1 = 0.0, max_abs = 0.0;
    std::size_t first_diff = a.size();
    for (std::size_t i = 0; i < a.size() && i < b.size(); ++i) {
        const double d = std::fabs(static_cast<double>(b[i]) - static_cast<double>(a[i]));
        if (std::bit_cast<std::uint32_t>(a[i]) == std::bit_cast<std::uint32_t>(b[i])) {
            ++same;
        } else if (first_diff == a.size()) {
            first_diff = i;
        }
        if (d > max_abs) max_abs = d;
        l1 += d;
        mag_ref += std::fabs(static_cast<double>(a[i]));
        mag_dev += std::fabs(static_cast<double>(b[i]));
    }
    const double rel = mag_ref > 0.0 ? l1 / mag_ref : l1;
    std::printf("整模型一步（48 层逐层交出）：|logits| 主机 %.6g / 设备 %.6g；L1 相对差 %.3g；"
                "最大绝对差 %.3g；位相同 %d/%zu（首个不同在 %zu）\n",
                mag_ref, mag_dev, rel, max_abs, same, a.size(), first_diff);
    std::printf("  选组 routed=%llu rerouted=%llu\n", (unsigned long long) rs.routed,
                (unsigned long long) rs.rerouted);
    if (fleet) {
        int busy = 0;
        for (int i = 0; i < dev_fleet.card_count(); ++i) {
            const ExpertWorker::Stats& st = dev_fleet.worker(i).stats();
            std::printf("  卡 %d：jobs=%llu experts=%llu failed=%llu\n", dev_fleet.device_of(i),
                        (unsigned long long) st.jobs, (unsigned long long) st.experts,
                        (unsigned long long) st.failed);
            if (st.jobs > 0) ++busy;
        }
        if (busy != dev_fleet.card_count()) {
            std::printf("  有卡没接到活：层切分没生效\n");
            return 1;
        }
    } else {
        std::printf("  单卡：jobs=%llu experts=%llu failed=%llu；常驻 %.1f MiB\n",
                    (unsigned long long) single_worker.stats().jobs,
                    (unsigned long long) single_worker.stats().experts,
                    (unsigned long long) single_worker.stats().failed,
                    static_cast<double>(single_worker.resident_bytes()) / (1024.0 * 1024.0));
    }
    if (a.size() != b.size()) return 1;
    // 分布级判据（ADR-009）：不是逐位，但差异要小到能说明两边算的是同一件事。
    if (!(rel < 1e-4)) {
        std::printf("  两条路的差异超出分布级容差\n");
        return 1;
    }
    if (rs.rerouted != 0) {
        std::printf("  有重派：说明有层没被设备侧接住\n");
        return 1;
    }
    // 每步每层一批：批数应该是 层数 × 步数。
    const std::uint64_t want_routed =
        static_cast<std::uint64_t>(cfg.n_layers) * static_cast<std::uint64_t>(tokens);
    if (rs.routed != want_routed) {
        std::printf("  交出的批数不对：%llu（应为 %llu）\n", (unsigned long long) rs.routed,
                    (unsigned long long) want_routed);
        return 1;
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc >= 3 && std::string(argv[1]) == "--model") {
        int layer = 0;
        int layers = 48;
        int tokens = 3;
        int threads = 0;
        bool step = false;
        bool fleet = false;
        for (int i = 2; i < argc; ++i) {
            const std::string a = argv[i];
            if (a == "--layer" && i + 1 < argc) {
                layer = std::atoi(argv[++i]);
            } else if (a == "--layers" && i + 1 < argc) {
                layers = std::atoi(argv[++i]);
            } else if (a == "--tokens" && i + 1 < argc) {
                tokens = std::atoi(argv[++i]);
            } else if (a == "--threads" && i + 1 < argc) {
                threads = std::atoi(argv[++i]);
            } else if (a == "--step") {
                step = true;
            } else if (a == "--fleet") {
                fleet = true;
            } else if (a != argv[2]) {
                // 认不出来的参数直接失败：最后一个参数曾经被循环上界跳过，于是「整模型一步」悄悄跑了
                // 单层入口、对照看着通过——这种静默换入口的错比跑挂更坏。
                std::printf("不认识的参数：%s\n", a.c_str());
                return 2;
            }
        }
        if (step) return manual_step_entry(argv[2], layers, fleet, tokens, threads);
        return manual_model_entry(argv[2], layer);
    }
    for (const GuFormat& gu : kGuFormats) test_device_matches_host(gu);
    test_same_slot_twice_gives_that_slot();
    test_unsupported_layers_fail_loudly();
    test_bad_inputs_fail_reply();
    test_worker_computes_a_job_and_stays_resident();
    test_select_group_prefers_the_cuda_group();
    test_unsupported_role_is_refused_by_the_worker();
    test_owner_split();
    test_fleet_and_each_card_runs();
    std::puts("expert_ffn_cuda: 设备侧一层前馈（权重常驻）、工人与 CUDA 组都成立");
    return 0;
}
