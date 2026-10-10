// PCIe 字节预算仲裁器。规则出自 interfaces.md 第 3 节与 engine.md §3.2：
//
//   优先级自高到低：本步必需的 KV、本步必需的表行、预测性专家（按收益 ÷ 字节排序）。
//   必需请求绝不因预算不足被丢弃——它在消费点阻塞，仲裁器只把超出预算的部分记成 overspend，
//   绝不为它插入额外的同步点。预测性请求放不下就弃掉并计入迟到。
#pragma once

#include <cstdint>
#include <vector>

namespace qinfer::scheduling {

enum class FlowClass : std::uint8_t { kKv, kTableRow, kExpert };
const char* to_string(FlowClass c);

struct Request {
    FlowClass cls = FlowClass::kExpert;
    std::uint64_t bytes = 0;
    std::int64_t consume_at_step = 0;   // 消费点，以 decode 步为单位
    bool must = false;                  // true = 必需（KV / 表行）；false = 预测性
    double benefit_per_byte = 0.0;      // 预测性请求的排序键
    std::uint32_t tag = 0;              // 调用方标识，回填到 Decision
};

enum class Verdict : std::uint8_t { kAdmitted, kDropped };

struct Decision {
    std::uint32_t tag = 0;
    FlowClass cls = FlowClass::kExpert;   // 回填类别，便于按类别汇总观测（step_budget::observe）
    Verdict verdict = Verdict::kDropped;
    std::uint64_t bytes = 0;
};

struct Result {
    std::vector<Decision> decisions;         // 与输入同序
    std::vector<std::uint32_t> admit_order;  // 按占用预算的先后，用于观察优先级
    std::uint64_t admitted_bytes = 0;
    std::uint64_t overspend_bytes = 0;       // 必需项超出预算的部分
    std::uint32_t dropped = 0;               // 被放弃的预测性请求数（迟到数）
    std::uint64_t dropped_bytes = 0;
};

Result arbitrate(std::uint64_t budget_bytes, const std::vector<Request>& requests);

}  // namespace qinfer::scheduling
