// HTTP 接线层：httplib 路由、鉴权、worker 池。请求在 HTTP 线程内完成解析与校验，交入调度器的有界
// 队列；槽位 worker 驱动门面生成，增量经 Job 的事件缓冲回写 HTTP 线程（非流式收齐、SSE 逐帧）。
//
// v0 并发正确性的一条关键约束：artifact::GgufFile 的读取用一份 mutable ifstream（seekg+read，
// gguf_table.cpp 的 read_at），不是线程安全的；PLE 的 GgufTable 同理。故这里用一把进程级 engine_mu_
// 串行化 Engine 的全部可变访问（generate 与 tokenizer().encode——encode 会写各 worker 共享的
// piece_cache_，见 tokenizer.cpp:261/275，核读结论见任务报告）——队列/槽位/事件缓冲结构已为 M2
// 就位，锁是吞吐优化的第一刀（换按线程文件句柄或 pread）。锁放服务层不放门面：Engine::generate
// 自身无锁，服务层是唯一并行使用引擎的地方。
#pragma once

#include <atomic>
#include <ctime>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "serve/facade.hpp"
#include "serve/prompt_cache.hpp"
#include "serve/prometheus.hpp"
#include "serve/records.hpp"
#include "serve/scheduler.hpp"
#include "serve/telemetry.hpp"
#include "serve/toolcall.hpp"

namespace httplib {
class Server;
struct Request;
struct Response;
}

namespace qinfer::serve {

// 503 的响应体是全服务唯一的字面量（规格 §4：`{error: "queue full"}` 的扁平形状）。
inline constexpr const char* kQueueFullBody = "{\"error\":\"queue full\"}";

struct ServeConfig {
    std::string model_path;
    std::string host = "127.0.0.1";
    int port = 8080;
    std::string api_key;                 // 空 = 不鉴权（本机默认）
    std::size_t slots = 8;
    std::size_t queue_capacity = 0;      // 0 = 2 × slots（规格 §4 第 1 条）
    int threads = 0;                     // httplib HTTP 线程池；0 = 2 × slots + 2
    std::size_t conversation_cache = 64; // token 前缀缓存条数；0 = 关（/metrics 的 conversation_cache 实读）
    std::uint64_t reject_after = 0;      // 0 = 关；>0 = 第 N 个请求起一律 503（集成测试用，注释见 main）
    // 门面几何直通（默认 = 真模型推导值；集成测试覆写成夹具值）。
    FacadeConfig facade;
};

class Serve {
public:
    Serve(ServeConfig cfg, std::shared_ptr<Engine> eng);
    ~Serve();

    bool listen(std::string& err);   // 注册路由并绑定，阻塞在 HTTP 线程池内；失败写 err
    void stop();                     // 线程安全，可从信号处理器调用

private:
    void register_routes();
    void worker_main();
    void handle_chat(const httplib::Request& req, httplib::Response& res, bool stream);
    void handle_completions(const httplib::Request& req, httplib::Response& res);
    void handle_messages(const httplib::Request& req, httplib::Response& res);
    void handle_count_tokens(const httplib::Request& req, httplib::Response& res);
    void serve_sse(const std::shared_ptr<Job>& job, bool include_usage,
                   const std::vector<Json>& tools, httplib::Response& res);
    void serve_anthropic_sse(const std::shared_ptr<Job>& job, const std::vector<Json>& tools,
                             httplib::Response& res);
    std::shared_ptr<Job> launch(const std::string& path, std::string prompt,
                                const GenOptions& opts, bool stream);
    void record_request(const std::shared_ptr<Job>& job, int http_status, bool disconnected);   // T12：填 last_timings 计时；T13：gen 累计；T14：Records 环
    TelemetryExtra telemetry_extra() const;   // T13：遥测每采样读一份的引擎侧指标（随 record_request 的累计）

    ServeConfig cfg_;
    std::shared_ptr<Engine> eng_;
    Scheduler sched_;
    PromptCache prompt_cache_;   // token 前缀缓存：命中免进 engine_mu_（spec §6 的 conversation_cache 数据源）
    std::mutex engine_mu_;             // 见文件头注释：v0 串行化引擎访问（encode + generate）
    std::vector<std::thread> workers_;
    std::atomic<std::uint64_t> req_id_{0};
    std::unique_ptr<httplib::Server> svr_;
    std::string web_index_, web_monitor_;   // 启动时读入的静态页
    // 最近一次完成的请求的计时（/v1/status 的 last_timings 与 /metrics 的 live 共用）。
    struct LastTimings {
        bool have = false;
        double queue_s = 0.0, first_token_s = 0.0, wallclock_s = 0.0, decode_tok_s = 0.0;
        std::uint64_t prompt_tokens = 0, output_tokens = 0;
        std::string finish;
    };
    mutable std::mutex last_mu_;     // telemetry_extra() 是 const，锁内读数需 mutable（同 EventBuffer::mu_）
    LastTimings last_;
    std::time_t started_;              // 服务启动时刻（/v1/status 的 uptime_s）
    // 遥测：T13 起在构造函数里 start()（launch 是每请求入口，无请求时也应采样）；/v1/status 的
    // machine 与 T14 的 /metrics 都读它。
    std::unique_ptr<Telemetry> tel_;
    // tok_s_mean 的累计口径：最近一次请求的解码速度是 v0 的 tok_s，全程均值另记这两项。
    std::uint64_t gen_done_ = 0;   // 累计生成的 token 数（成功请求才累加）
    std::uint64_t gen_us_ = 0;     // 累计解码耗时（µs）
    std::atomic<std::uint64_t> reused_{0};   // 缓存命中的 prompt token 累计（totals.reused；worker 线程写）
    // 请求记录环（/metrics 的 requests/totals 与 /api/requests 的数据源，T14 起 record_request 填）。
    Records records_;
    // 延迟三直方图（/metrics 的 Prometheus text 端口的 ttft/itl/e2e 家族；record_request 落完成时刻时
    // observe，自带锁，HTTP 线程 snapshot——与 last_ 的临界区无关）。
    Latencies lat_;
    // /metrics 的 totals 累计（与 last_ 同临界区更新——record_request 是 HTTP 线程调用的唯一写入者；
    // /metrics 读时也持 last_mu_，无竞态）。
    struct Totals {
        std::uint64_t prompt_tokens = 0, completion_tokens = 0, requests = 0;
        double prompt_ms = 0.0, decode_ms = 0.0;
    };
    Totals totals_;
};

// main.cpp 用：命令行解析（自写的键值扫描，零依赖）。
ServeConfig parse_flags(int argc, char** argv, std::string& err);

}  // namespace qinfer::serve