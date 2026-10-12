// 请求记录环（规格 §6 的 /api/requests 与 /metrics 数据源；字段对齐 Strata 的 record 字典）：
// 容量 100、大字段 256 KiB 截断（带截断旗标）、tail 插入丢最老。任何东西都不允许拖垮服务——
// begin/finish 里没有会抛的路径（id 用进程内唯一即可，无需全局唯一）。
#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <random>
#include <string>
#include <vector>

namespace qinfer::serve {

// 一条请求记录。大字段（input/output/reasoning）截到 256 KiB 并置对应旗标。
struct RequestRecord {
    std::string id;               // 12 位十六进制（环内唯一）
    std::string path;             // 请求路径（/v1/chat/completions 等）
    std::string model;
    double created = 0.0;         // 请求开始时刻（unix 秒）
    std::string state = "queued"; // queued → completed / error
    bool stream = false;
    std::string response_format;  // json_object / 空（未指定）
    std::string input;  bool input_truncated = false;    // 原始请求体（≤256 KiB）
    std::string output; bool output_truncated = false;   // 生成全文
    std::string reasoning; bool reasoning_truncated = false;  // v0 恒空
    double queue_s = 0.0;         // 入队到租约
    double first_token_s = 0.0;   // 租约到首 token（0 = 未达）
    double wallclock_s = 0.0;     // 入队到结束；未结束时 list/detail 补实时
    double decode_tok_s = 0.0;    // 解码速度 tok/s
    std::uint64_t prompt_tokens = 0, output_tokens = 0;
    double clock_ = 0.0;          // 记录创建时刻（steady 秒）——未结束时补 wallclock_s 用
    bool finished = false;
};

// begin 的入参（服务端从 Job 与请求体取）。
struct RequestBegin {
    std::string path, model, input;
    bool stream = false;
    std::string response_format;
    double created = 0.0;   // unix 秒
};

// finish 的入参（服务端从 Job 的计时与结果取）。
struct RequestFinish {
    std::string text;                  // → output
    std::string error;                 // 非空 → state=error（如 "disconnected"）
    double queue_s = 0.0, first_token_s = 0.0, wallclock_s = 0.0, decode_tok_s = 0.0;
    std::uint64_t prompt_tokens = 0, output_tokens = 0;
};

class Records {
public:
    static constexpr std::size_t kCap = 100;           // Strata 的 retention=100
    static constexpr std::size_t kTruncBytes = 262144; // 256 KiB

    Records();                           // rng_ 用 random_device 播种（见 records.cpp）

    // 建记录（state=queued，input 截断），返回 id；满 kCap 丢最老。
    std::string begin(const RequestBegin& b);
    // 收尾：填计时/输出；error 非空 → state=error，否则 completed。
    void finish(const std::string& id, const RequestFinish& f);
    // 单条拷贝（id 不存在 → false）。未结束的 wallclock_s 补实时。
    bool detail(const std::string& id, RequestRecord& out) const;
    // 最近记录，最新在前（前端列表）。未结束的 wallclock_s 补实时。
    std::vector<RequestRecord> list() const;
    std::size_t size() const;

private:
    void truncate_inplace(std::string& s, bool& flag) const;   // 超长 → 截断 + 旗标
    std::string next_id();                                     // 12 位 hex

    mutable std::mutex mu_;
    std::vector<RequestRecord> ring_;   // tail 插入；超 kCap 丢最老
    std::mt19937_64 rng_;               // id 生成（random_device 播种；环内唯一足够）
};

}  // namespace qinfer::serve