// 计划层到仲裁器的缝：每步预算的推导、三类需求到请求的转换、以及每步观测记录。
// 依据：interfaces.md 第 3、4、6、7 节与 engine.md §3.2。
//
// 三件事：
//   1. 每步 PCIe 字节预算由机器画像推导（画像 formulas 里的那条公式）；
//   2. 计划层每步的四类需求（interfaces §4）里要过仲裁器的那三类转成 Request；
//   3. 仲裁结果折成每步的观测记录（interfaces §7）——性能结论必须能追溯到它。
//
// 三者分开是因为它们的口径不同：预算是**设计输入 + 实测**（io_share 是设计输入，不是测出来的）；
// 需求是**计划层的输出**（约定为需求而不是命令）；观测是**唯一可引用的取数口径**。
#pragma once

#include <cstdint>
#include <vector>

#include "scheduling/budget_arbiter.hpp"

namespace qinfer::scheduling {

// 画像里参与预算推导的三项（interfaces §6 的「测得的输入」与「设计输入」）。
// io_share 明说是设计输入、不是测得：一个 decode 步里允许被 IO 占用的时间份额，本设计取 0.5。
struct ProfileInputs {
    double pcie_h2d_gbps = 0.0;  // 画像字段 pcie_h2d_gbps
    double step_ms = 0.0;        // 画像字段 step_p50_ms
    double io_share = 0.5;
};

// 每步预算的字节数。照画像里记的那条公式算：pcie_h2d × step_ms/1000 × io_share，再把结果里的「GB」按
// 1024 MiB 换算（画像的 formulas 字段明写 ×1024）。本实现照此，以保证与档案里环境2 的 186.8 MiB 一致。
// 口径提醒：若那个 28.64 GB/s 按十进制 GB 换算，同一条记录会得 173.9 MiB，与档案差 7.4%——这是画像
// 记的口径（登记在 S-52），不是本模块的选择；要改就得连档案一起改，不能只改这里。
std::uint64_t step_budget_bytes(const ProfileInputs& in);

// 一个 token 的表行需求（interfaces §4 第 1 类，确定性：由上一 token 唯一决定）。
// rows_per_token 是本模型的头数（16）；row_bytes 是 IQ4_NL 一行（90）。
// positions_per_token 是每生成一个 token 的位置数——G-13 实测 1.047 至 1.250（中位 1.133），
// 画像里那条 demand_table_mib 反推出来约 1.13（见测试）。
struct TableDemand {
    std::uint32_t rows_per_token = 0;
    std::uint64_t row_bytes = 0;
    double positions_per_token = 1.0;
};
std::uint64_t table_demand_bytes(const TableDemand& d);

// 预测性的专家候选（interfaces §4 第 2 类，概率性：来自共现与会话亲和）。
// bytes 逐条给而不是给一个公共值：逐层的量化档不同，专家字节也就不同。
struct ExpertCandidate {
    std::uint32_t id = 0;
    double probability = 0.0;
    std::uint64_t bytes = 0;
};

// 计划层一步向仲裁器提的需求。interfaces §4 的第 4 类（精度档）是决策、不进仲裁器，故不在这里。
struct StepDemand {
    std::int64_t step = 0;
    std::uint64_t kv_bytes = 0;        // 本步必需的 KV 换入（第 1 类）
    TableDemand table;                 // 本步必需的表行（第 2 类）
    std::uint32_t table_tokens = 1;    // 这一步要读几个 token 的表行（投机窗口下可多于 1）
    std::vector<ExpertCandidate> experts;  // 预测性（第 3 类）
};

// 转成仲裁器认识的请求：KV（必需）→ 表行（必需）→ 每个候选一个预测性请求。
// 排序键用概率 ÷ 字节（与仲裁器的键一致），故候选的先后不影响结果；各自带 id 作为 tag 便于回填。
std::vector<Request> build_requests(const StepDemand& d);

// 每步的观测（interfaces §7 的预算相关子集）。
struct StepObservation {
    std::int64_t step = 0;
    std::uint64_t budget_bytes = 0;
    std::uint64_t admitted_bytes = 0;
    std::uint64_t overspend_bytes = 0;
    std::uint32_t late_requests = 0;  // 被放弃的预测性请求数（迟到数）
    std::uint64_t late_bytes = 0;
    std::uint32_t kv_admitted = 0;
    std::uint32_t table_rows_admitted = 0;
    std::uint32_t experts_admitted = 0;
    double budget_used = 0.0;  // admitted ÷ budget；预算为 0 时记 0 而不是无穷
};
StepObservation observe(const Result& r, std::uint64_t budget_bytes, std::int64_t step);

}  // namespace qinfer::scheduling