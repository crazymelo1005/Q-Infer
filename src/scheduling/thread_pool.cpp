#include "scheduling/thread_pool.hpp"

#include <algorithm>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>

namespace qinfer::scheduling {

// 一次只跑一个作业的池：没有任务队列，因为调用点很密、每次的活很小，队列的延迟比它省下的更多。
// 作业放在固定字段里，工作线程靠 epoch 变化醒过来，然后自己不停认领下一块，直到块被认领完。
struct ThreadPool::Impl {
    std::mutex m;
    std::condition_variable cv_job;
    std::condition_variable cv_idle;

    const std::function<void(std::uint64_t, std::uint64_t)>* fn = nullptr;
    std::uint64_t n = 0;
    std::uint64_t chunk = 1;
    std::uint64_t next = 0;   // 下一个可认领的块起点
    std::uint64_t epoch = 0;  // 每来一个作业加一
    std::uint64_t done = 0;   // 这一轮已经收工的参与方数（调用线程 + 醒过来的工作线程）
    std::size_t participants = 1;
    bool stop = false;
    std::vector<std::thread> workers;

    // 一个作业：醒过来 → 不停认领块直到认领完 → 记一次收工。
    // 判「作业做完」用的是**参与方个数**而不是「还有谁在忙」：工作线程可能醒得晚，用忙闲计数会在
    // 它还没醒的时候就以为做完了（那样调用方会先返回，fn 就成了悬垂指针）。按个数等就没有这个缝。
    void worker_loop() {
        std::uint64_t seen = 0;
        for (;;) {
            const std::function<void(std::uint64_t, std::uint64_t)>* job = nullptr;
            std::uint64_t total = 0, step = 1;
            {
                std::unique_lock<std::mutex> lk(m);
                cv_job.wait(lk, [&] { return stop || epoch != seen; });
                if (stop) return;
                seen = epoch;
                job = fn;
                total = n;
                step = chunk;
            }
            for (;;) {
                std::uint64_t b = 0;
                {
                    std::lock_guard<std::mutex> lk(m);
                    if (next >= total) break;
                    b = next;
                    next += step;
                }
                (*job)(b, std::min(b + step, total));
            }
            {
                std::lock_guard<std::mutex> lk(m);
                if (++done == participants) cv_idle.notify_all();
            }
        }
    }
};

ThreadPool::ThreadPool(int n_threads) : impl_(std::make_unique<Impl>()) {
    n_threads_ = std::max(1, n_threads);
    impl_->workers.reserve(static_cast<std::size_t>(n_threads_ - 1));
    for (int i = 1; i < n_threads_; ++i) {
        impl_->workers.emplace_back([this] { impl_->worker_loop(); });
    }
}

ThreadPool::~ThreadPool() {
    {
        std::lock_guard<std::mutex> lk(impl_->m);
        impl_->stop = true;
    }
    impl_->cv_job.notify_all();
    for (std::thread& t : impl_->workers) {
        if (t.joinable()) t.join();
    }
}

void ThreadPool::run_rows(std::uint64_t n, std::uint64_t grain,
                          const std::function<void(std::uint64_t, std::uint64_t)>& fn) {
    if (n == 0) return;
    const std::uint64_t threads = static_cast<std::uint64_t>(n_threads_);
    if (n_threads_ <= 1 || static_cast<std::uint64_t>(n) <= std::max<std::uint64_t>(1, grain)) {
        fn(0, n);
        return;
    }
    // 块大小取「按线程数均分」与 grain 的较大者：行太少时别切得太碎（切块的固定开销会吃掉收益）。
    const std::uint64_t by_threads = (n + threads - 1) / threads;
    const std::uint64_t step = std::max<std::uint64_t>(by_threads, std::max<std::uint64_t>(1, grain));
    {
        std::lock_guard<std::mutex> lk(impl_->m);
        impl_->fn = &fn;
        impl_->n = n;
        impl_->chunk = step;
        impl_->next = 0;
        impl_->done = 0;
        impl_->participants = impl_->workers.size() + 1;  // 工作线程 + 调用线程自己
        ++impl_->epoch;
    }
    impl_->cv_job.notify_all();
    // 调用线程自己也认领块——20 个核时它不该站着看。
    for (;;) {
        std::uint64_t b = 0;
        {
            std::lock_guard<std::mutex> lk(impl_->m);
            if (impl_->next >= n) break;
            b = impl_->next;
            impl_->next += step;
        }
        fn(b, std::min(b + step, n));
    }
    {
        std::unique_lock<std::mutex> lk(impl_->m);
        if (++impl_->done == impl_->participants) impl_->cv_idle.notify_all();
        // 等所有参与方收工：工作线程可能醒得晚，只等「手上这块认领完」会提早返回。
        impl_->cv_idle.wait(lk, [&] { return impl_->done >= impl_->participants; });
        impl_->fn = nullptr;
    }
    // 注意：这里不能再被调用——工作线程此刻都在等下一个作业，嵌套调用会死等。
}

namespace {
std::unique_ptr<ThreadPool>& pool_slot() {
    static std::unique_ptr<ThreadPool> p;
    return p;
}
int g_threads = 0;
}  // namespace

void set_pool_threads(int n) {
    g_threads = n;
    if (n <= 1) {
        pool_slot().reset();
        return;
    }
    pool_slot() = std::make_unique<ThreadPool>(n);
}

ThreadPool* pool() { return pool_slot().get(); }

}  // namespace qinfer::scheduling
