// 一个只做一件事的线程池：把若干**独立的行**并行算完。
//
// 为什么按行切：所有量化内核的一行是一个独立的点积，行与行之间没有任何顺序依赖，故按行并行
// 不改变任何一个输出元素的浮点累加顺序——同配置下仍逐位可复现（engine §16 那条口径要的就是这个）。
// 门控残差那两个 F32 投影同理（一行也是一个独立的点积，见 dense/gated_residual）：池子只认「行」，
// 不认那是量化档还是 F32。
// 它**不切归约维**：那会改掉求和顺序，正是「同配置逐位可复现」要避免的。
//
// 默认关（线程数 0 或 1 → `pool()` 返回空），所以既有测试与 CI 仍在单线程下跑；并行版与单线程版
// 逐位相同这一点由测试钉住（tests/test_thread_pool.cpp）。
#pragma once

#include <cstdint>
#include <functional>
#include <memory>

namespace qinfer::scheduling {

class ThreadPool {
public:
    explicit ThreadPool(int n_threads);  // < 1 视作 1
    ~ThreadPool();
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    int threads() const { return n_threads_; }

    // 把 [0, n) 切成至多 threads() 块（每块不少于 grain 行）跑完再返回；每块调 fn(begin, end)。
    void run_rows(std::uint64_t n, std::uint64_t grain,
                  const std::function<void(std::uint64_t, std::uint64_t)>& fn);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    int n_threads_ = 1;
};

// 进程级的那一个池：`set_pool_threads(n)` 配置（n <= 1 → 关，`pool()` 返回空），`pool()` 取它。
// 为什么做成进程级而不是穿参数：解码循环只有一个，而所有量化内核与注意力投影都要用到它；
// 穿参数会把一串内核的签名都改一遍，换来的只是「显式」。
void set_pool_threads(int n);
ThreadPool* pool();

}  // namespace qinfer::scheduling
