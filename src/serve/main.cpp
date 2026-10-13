// 服务入口：命令行解析（自写键值扫描，零依赖）→ 装载门面 → 起 Serve。
#include "httplib.h"
#include "scheduling/thread_pool.hpp"
#include "serve/facade.hpp"
#include "serve/server.hpp"
#ifdef QINFER_CUDA_BUILD
#include "dense/dense_exec_cuda.hpp"
#endif

#include <algorithm>
#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <thread>
#include <vector>

using namespace qinfer::serve;

namespace {

Serve* g_serve = nullptr;
volatile std::sig_atomic_t g_signaled = 0;

void on_signal(int) {
    if (g_serve) g_serve->stop();
    g_signaled = 1;
}

// 数值参数（裁定⑥）：必须整串是合法数字（endptr 走到串尾、无 ERANGE 溢出），垃圾 → err，main 退出 2。
std::int64_t arg_int(const std::map<std::string, std::string>& a, const char* key, std::int64_t dflt,
                     std::string& err) {
    const auto it = a.find(key);
    if (it == a.end()) return dflt;
    const std::string v = it->second;
    errno = 0;
    char* end = nullptr;
    const long long r = std::strtoll(v.c_str(), &end, 10);
    if (end == v.c_str() || *end != '\0' || errno == ERANGE) {
        err = "参数 " + std::string(key) + " 需要整数，得「" + v + "」";
        return dflt;
    }
    return static_cast<std::int64_t>(r);
}

// --gpu 解析：auto（全部可见卡）或逗号分隔的非负设备索引（nvidia-smi 序号）。非法 → err。
bool parse_gpu(const std::map<std::string, std::string>& a, bool& gpu_auto,
               std::vector<int>& devices, std::string& err) {
    const auto it = a.find("--gpu");
    if (it == a.end()) return true;
    const std::string v = it->second;
    if (v == "auto") {
        gpu_auto = true;
        return true;
    }
    std::size_t start = 0;
    while (start <= v.size()) {
        const std::size_t comma = v.find(',', start);
        const std::string tok = v.substr(start, comma == std::string::npos ? std::string::npos
                                                                          : comma - start);
        if (tok.empty()) {
            err = "参数 --gpu 需要 auto 或逗号分隔的设备索引，得「" + v + "」（空段）";
            return false;
        }
        errno = 0;
        char* end = nullptr;
        const long long r = std::strtoll(tok.c_str(), &end, 10);
        if (end == tok.c_str() || *end != '\0' || errno == ERANGE || r < 0) {
            err = "参数 --gpu 需要非负设备索引，段「" + tok + "」非法（得 " + v + "）";
            return false;
        }
        devices.push_back(static_cast<int>(r));
        if (comma == std::string::npos) break;
        start = comma + 1;
    }
    return true;
}

double arg_double(const std::map<std::string, std::string>& a, const char* key, double dflt,
                  std::string& err) {
    const auto it = a.find(key);
    if (it == a.end()) return dflt;
    const std::string v = it->second;
    errno = 0;
    char* end = nullptr;
    const double r = std::strtod(v.c_str(), &end);
    if (end == v.c_str() || *end != '\0' || errno == ERANGE) {
        err = "参数 " + std::string(key) + " 需要数字，得「" + v + "」";
        return dflt;
    }
    return r;
}

}  // namespace

// server.hpp 声明的是 qinfer::serve::parse_flags，定义要在同名命名空间里补上（main 保持全局作用域）。
namespace qinfer::serve {

ServeConfig parse_flags(int argc, char** argv, std::string& err) {
    std::map<std::string, std::string> a;
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        if (s.rfind("--", 0) != 0) { err = "位置参数不需要：" + s; return {}; }
        const auto eq = s.find('=');
        const std::string k = eq == std::string::npos ? s : s.substr(0, eq);
        const std::string v = eq == std::string::npos ? (i + 1 < argc ? argv[++i] : std::string{}) : s.substr(eq + 1);
        if (v.empty()) { err = "参数 " + k + " 需要值"; return {}; }
        a[k] = v;
    }
    ServeConfig c;
    const auto it = a.find("--model");
    if (it == a.end()) { err = "缺 --model（主案例 GGUF 的路径）"; return {}; }
    c.model_path = it->second;
    if (a.count("--host")) c.host = a["--host"];
    c.port = static_cast<int>(arg_int(a, "--port", 8080, err));
    if (a.count("--api-key")) c.api_key = a["--api-key"];
    c.slots = static_cast<std::size_t>(arg_int(a, "--slots", 8, err));
    c.queue_capacity = static_cast<std::size_t>(arg_int(a, "--queue", 0, err));
    c.threads = static_cast<int>(arg_int(a, "--threads", 0, err));
    // 推理线程池：默认取物理并发数的一半、封顶 8（不满载——用户要求；HTTP 池与之无关）。
    {
        unsigned hw = std::thread::hardware_concurrency();
        const int dflt = hw > 0 ? static_cast<int>(std::min<unsigned>(hw / 2, 8)) : 2;
        c.compute_threads = static_cast<int>(arg_int(a, "--compute-threads", dflt, err));
    }
    c.conversation_cache = static_cast<std::size_t>(arg_int(a, "--conversation-cache", 64, err));
    if (!parse_gpu(a, c.gpu_auto, c.gpu_devices, err)) return {};
    c.reject_after = static_cast<std::uint64_t>(arg_int(a, "--reject-after", 0, err));
    FacadeConfig& f = c.facade;
    f.model_path = c.model_path;
    f.n_embd = static_cast<std::uint64_t>(arg_int(a, "--n-embd", 2560, err));
    f.hc = static_cast<std::uint64_t>(arg_int(a, "--hc", 4, err));
    f.hc_lr = static_cast<std::uint64_t>(arg_int(a, "--hc-lr", 320, err));
    f.n_layers = static_cast<int>(arg_int(a, "--layers", 48, err));
    f.top_k = static_cast<int>(arg_int(a, "--top-k", 10, err));
    f.gdn_head_dim = static_cast<std::uint64_t>(arg_int(a, "--gdn-head-dim", 128, err));
    f.qsa_head_dim = static_cast<std::uint64_t>(arg_int(a, "--qsa-head-dim", 256, err));
    f.n_rot = static_cast<std::uint64_t>(arg_int(a, "--n-rot", 64, err));
    f.idx_dim = static_cast<std::uint64_t>(arg_int(a, "--idx-dim", 128, err));
    f.idx_top_k = static_cast<std::uint64_t>(arg_int(a, "--idx-top-k", 2048, err));
    f.idx_block = static_cast<std::uint64_t>(arg_int(a, "--idx-block", 4, err));
    f.max_context = static_cast<std::uint64_t>(arg_int(a, "--max-context", 262144, err));
    f.cache_slots = static_cast<std::size_t>(arg_int(a, "--cache-slots", 0, err));
    f.cache_byte_budget = static_cast<std::uint64_t>(arg_int(a, "--cache-byte-budget", 0, err));
    f.ple_budget_bytes = static_cast<std::uint64_t>(arg_int(a, "--ple-budget-bytes", 0, err));
    if (a.count("--ple-table")) f.ple_table = a["--ple-table"];
    f.kv_slots = static_cast<std::uint64_t>(arg_int(a, "--kv-slots", 0, err));
#ifdef QINFER_CUDA_BUILD
    // --gpu 激活 dense 设备执行（阶段 A dense 接线）：把 CUDA 整层入口填进门面，step_loop 每层走 *_cuda。
    // 没给 --gpu 时保持空 → 全主机路径（与主机构建行为一致）。
    if (c.gpu_auto || !c.gpu_devices.empty()) {
        c.facade.dense_exec = dense::dense_exec_cuda();
    }
#endif
    return c;
}

}  // namespace qinfer::serve

int main(int argc, char** argv) {
    std::string err;
    const ServeConfig cfg = parse_flags(argc, argv, err);
    if (!err.empty()) {
        std::fprintf(stderr, "%s\n", err.c_str());
        std::fprintf(stderr, "用法：qinfer-server --model <gguf> [--host H] [--port P] [--api-key K] "
                             "[--slots N] [--queue N] [--threads N] [--compute-threads N] [--gpu auto|0,1] [--reject-after N] "
                             "[--n-embd N] [--hc N] [--hc-lr N] [--layers N] [--top-k N] "
                             "[--gdn-head-dim N] [--qsa-head-dim N] [--n-rot N] [--idx-dim N] "
                             "[--idx-top-k N] [--idx-block N] [--max-context N] [--cache-slots N] "
                             "[--cache-byte-budget N] [--ple-table P] [--ple-budget-bytes N] [--kv-slots N] "
                             "[--conversation-cache N]\n        --threads N 是 HTTP 线程池；"
                             "--compute-threads N 是推理线程池（默认物理核数一半、封顶 8，1 或负 = 单线程）；"
                             "--gpu 选择 CUDA 设备（auto = 全部，0,1 = 指定；需带 CUDA 的构建）\n");
        return 2;
    }
    // --reject-after：仅供集成测试制造确定性 503（第 N 个请求起一律拒绝入队），生产不用。
    // 在 handler 里生效：launch() 前检查 req_id_ 与阈值。
#ifndef QINFER_CUDA_BUILD
    if (cfg.gpu_auto || !cfg.gpu_devices.empty()) {
        std::fprintf(stderr, "--gpu 需要带 CUDA 的构建（cmake -B build/cuda -DQINFER_ENABLE_CUDA=ON）；"
                             "本构建 CUDA 未启用\n");
        return 1;
    }
#endif
    std::shared_ptr<Engine> eng = std::make_shared<Engine>();
    if (!Engine::open(cfg.facade, *eng, err)) {
        std::fprintf(stderr, "装载失败：%s\n", err.c_str());
        return 1;
    }
    std::printf("模型 %s（词表 %llu、n_embd %llu、%d 层）就绪\n", eng->model_name().c_str(),
                (unsigned long long) eng->tokenizer().vocab_size(),
                (unsigned long long) cfg.facade.n_embd, cfg.facade.n_layers);
    // 推理线程池：显式 --compute-threads N 生效；默认 = 物理核数一半封顶 8（不满载，用户要求）。
    // 1 或负 = 关池子（单线程，与 engine §6 的 set_pool_threads 语义一致）。HTTP 池由 cfg.threads
    // 单独管理（2*slots+2），两者互不影响。
    qinfer::scheduling::set_pool_threads(cfg.compute_threads);
    std::printf("推理线程池：%d 线程（--compute-threads）\n", cfg.compute_threads);
    Serve serve(cfg, eng);
    g_serve = &serve;
    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);
    if (!serve.listen(err)) {
        std::fprintf(stderr, "%s\n", err.c_str());
        return 1;
    }
    std::printf("服务退出（信号 %d）\n", static_cast<int>(g_signaled));
    return 0;
}