// 每 Job 一个的有界阻塞事件缓冲 + 槽位调度器（规格 §4）。
//   * EventBuffer：worker 写、HTTP 线程读；满则 push 阻塞（背压），绝不丢事件。
//     结束协议：worker 推完 kDone（或 kError）后调 stop()；读侧 pop 取完剩余返回假。
//     cancel()：客户端断开时调用，唤醒一切阻塞并令后续 push/pop 失败（worker 随即终止生成）。
//   * Scheduler：有界队列（容量 = 槽位数的 2 倍）+ 槽位租约。enqueue 满 → 返回假（HTTP 层 503）；
//     take() 按队序租约并记 leased_at；done() 释放。计时观测（queue_s / first_token_s / wallclock_s
//     / tok_s）全部落在 Job 字段，由 worker 与调度器填，统计不经过共享可变全局。
#pragma once

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "serve/facade.hpp"

namespace qinfer::serve {

enum class EventKind { kToken, kDone, kError };

struct JobEvent {
    EventKind kind = EventKind::kToken;
    std::string text;     // kToken：某 token 的解码字节片段
    std::string finish;   // kDone：finish_reason（stop / length）
    std::string err;      // kError：错误消息
};

class EventBuffer {
public:
    explicit EventBuffer(std::size_t capacity) : cap_(capacity > 0 ? capacity : 1) {}

    bool push(const JobEvent& e);           // 满则阻塞；已停止/取消 → false
    bool pop(JobEvent& out);                // 空则阻塞；已停止且取空 → false
    void stop();                            // 正常收尾：剩余事件可取完
    void stop_with_error(std::string err);  // 先入队 kError 再 stop
    void cancel();                          // 客户端断开：唤醒全部、后续失败
    bool cancelled() const;
    std::string error() const;              // stop_with_error 的消息；未出错为空

private:
    const std::size_t cap_;
    mutable std::mutex mu_;   // cancelled()/error() 是 const，锁内读数需 mutable（同 Scheduler::mu_）
    std::condition_variable cv_empty_, cv_full_;
    std::deque<JobEvent> q_;
    bool stopped_ = false;
    bool cancelled_ = false;
    std::string err_;
};

// 一个请求的生命。HTTP 线程填前段（path/model/prompt/opts/stream），worker 填中段（prompt_ids、
// n_gen/text/finish/tok_s/error），调度器填取件时间（leased_at），HTTP 线程读全量写记录。
struct Job {
    std::uint64_t id = 0;
    std::string path;                    // 请求路径（记录与 /metrics 的 phase 用）
    std::string model;                   // 模型名（记录用）
    std::string prompt;                  // 渲染好的完整 prompt 文本
    GenOptions opts;
    std::vector<std::int32_t> prompt_ids;    // worker 分词后回填（供统计）
    std::uint64_t n_prompt = 0;              // 预填 token 数（= prompt_ids.size()）
    std::uint64_t n_gen = 0;
    std::string text;                        // 生成全文（stop 截断后）
    std::string finish;                      // stop / length / error
    std::string error;                       // 非空 = 失败原因
    double tok_s = 0.0;                      // 生成速率 = n_gen / (done_at - first_at)
    std::chrono::steady_clock::time_point enqueued_at, leased_at, first_at, done_at;
    std::shared_ptr<EventBuffer> events = std::make_shared<EventBuffer>(4096);
    bool stream = false;
    std::string raw_body;    // 原始请求体（HTTP 线程在 launch 后填；入记录的 input）
    std::string record_id;   // 记录环键（空 = 未建；record_request 幂等用）
};

struct SlotView {
    int index = 0;
    bool busy = false;
    std::uint64_t job_id = 0;
    double queue_s = 0.0;      // 当前 job 已排队秒数
    double elapsed_s = 0.0;    // 当前 job 已租约秒数
    double tok_s = 0.0;
};

class Scheduler {
public:
    Scheduler(std::size_t n_slots, std::size_t queue_capacity);
    ~Scheduler();

    bool enqueue(std::shared_ptr<Job> job);   // 队列满 → false（HTTP 层回 503）
    std::shared_ptr<Job> take();              // 阻塞；stop() 后返回 nullptr
    void done(std::shared_ptr<Job> job);      // 释放槽位
    void stop();

    std::size_t slots() const { return slots_.size(); }
    std::size_t queued() const;               // 排队等待数（不含槽位内）
    bool any_busy() const;
    std::vector<SlotView> slots_view() const;
    bool is_stopped() const;

private:
    const std::size_t qcap_;
    mutable std::mutex mu_;
    std::condition_variable cv_;
    std::deque<std::shared_ptr<Job>> q_;
    std::vector<std::shared_ptr<Job>> slots_;   // 槽位 → 当前 job（空 = 空闲）
    bool stop_ = false;
};

}  // namespace qinfer::serve