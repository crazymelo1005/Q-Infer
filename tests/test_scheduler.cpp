// 每 Job 一个的有界阻塞事件缓冲 + 槽位调度器。环形满 = 背压（push 阻塞，绝不丢事件）；
// 槽位租约按队列序；队列满 enqueue 返回假（HTTP 层回 503）。
#include "check.hpp"
#include "serve/scheduler.hpp"
#include "serve/server.hpp"   // kQueueFullBody：队列满的 503 体（server.cpp 直接用同一常量）

#include <atomic>
#include <chrono>
#include <cstdio>
#include <thread>

using namespace qinfer::serve;

int main() {
    // ---- EventBuffer：先入先出 + 停止协议 ----
    {
        EventBuffer b(2);
        CHECK(b.push({EventKind::kToken, "a", {}, {}}));
        CHECK(b.push({EventKind::kToken, "b", {}, {}}));
        JobEvent e;
        CHECK(b.pop(e)); CHECK(e.text == "a");
        CHECK(b.pop(e)); CHECK(e.text == "b");
        // 空且未停止：pop 阻塞；另起线程 push 后唤醒。
        std::atomic<bool> got{false};
        std::thread t([&] { JobEvent x;
                            if (b.pop(x)) got = x.text == "c"; });
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
        b.push({EventKind::kToken, "c", {}, {}});
        t.join();
        CHECK(got);
        // stop：取完剩余事件后 pop 返回假。
        b.stop();
        JobEvent after;
        CHECK(!b.pop(after));
    }
    // 满环背压：cap=2 塞 3 个，第三个 push 必须等 pop 腾位。
    {
        EventBuffer b(2);
        b.push({EventKind::kToken, "1", {}, {}});
        b.push({EventKind::kToken, "2", {}, {}});
        std::thread t([&] { b.push({EventKind::kToken, "3", {}, {}}); });
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
        JobEvent e;
        CHECK(b.pop(e)); CHECK(e.text == "1");       // 腾位后 "3" 入队
        t.join();
        CHECK(b.pop(e)); CHECK(e.text == "2");
        CHECK(b.pop(e)); CHECK(e.text == "3");
        b.stop();
    }
    // cancel：唤醒阻塞的 pop 并令 push 失败；error() 给消息。
    {
        EventBuffer b(4);
        b.push({EventKind::kToken, "x", {}, {}});
        b.cancel();
        JobEvent e;
        CHECK(!b.pop(e));                            // 已取消：残余不被消费
        CHECK(!b.push({EventKind::kToken, "y", {}, {}}));
        CHECK(b.cancelled());
        EventBuffer b2(2);
        b2.stop_with_error("引擎炸了");
        JobEvent e2;
        CHECK(b2.pop(e2)); CHECK(e2.kind == EventKind::kError);
        CHECK(!e2.err.empty());
        CHECK(!b2.pop(e2));
        CHECK(b2.error() == "引擎炸了");
    }
    // ---- Scheduler：租约、队列序、释放、队列满、stop ----
    {
        Scheduler s(1, 2);                           // 1 槽、队列容量 2（=槽位数×2，规格 §4.1）
        auto mk = [](std::uint64_t id) { return std::make_shared<Job>(Job{id}); };
        CHECK(s.enqueue(mk(1)));
        CHECK(s.enqueue(mk(2)));
        CHECK(!s.enqueue(mk(3)));                    // 队列满（qcap==2，2 排队）→ 回 503
        CHECK(s.queued() == 2u);
        auto j1 = s.take();
        CHECK(j1 && j1->id == 1);                    // 队首先租约
        CHECK(s.queued() == 1u);
        s.done(j1);                                  // 生成结束释放槽位，下一个请求才能租约
        auto j2 = s.take();
        CHECK(j2 && j2->id == 2);
        CHECK(j2->leased_at > j1->leased_at);        // 租约计时在跑
        // 槽位被占（job2）：第三个 take 阻塞，done 后才返回。
        std::atomic<bool> leased{false};
        std::thread t([&] { auto j = s.take(); if (j) { leased = j->id == 3; s.done(j); } });
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
        s.enqueue(mk(3));                            // 入队后槽位仍被占：take 依旧阻塞（槽条件）
        s.done(j2);                                  // done 释放槽位 → 线程取到 job3
        t.join();
        CHECK(leased);
        // 队列空时 take 挂起；入队后按队序取。
        s.enqueue(mk(4));
        auto j4 = s.take();
        CHECK(j4 && j4->id == 4);
        s.done(j4);
        CHECK(!s.any_busy());
        auto view = s.slots_view();
        CHECK(view.size() == 1u && !view[0].busy);
        s.stop();
        auto jn = s.take();
        CHECK(!jn);                                  // stop 后 take 返回空
    }
    // ---- 队列满 → 503 的字面量：全服务唯一的常量（server.cpp 直接用），两端钉死它 ----
    {
        CHECK(std::string(kQueueFullBody) == "{\"error\":\"queue full\"}");
        Scheduler s(1, 2);                           // 1 槽、队列容量 2（=槽位数×2，规格 §4.1）
        CHECK(s.enqueue(std::make_shared<Job>(Job{1})));
        CHECK(s.enqueue(std::make_shared<Job>(Job{2})));
        CHECK(!s.enqueue(std::make_shared<Job>(Job{3})));   // 队列满，入队拒绝
        s.stop();
    }
    std::puts("scheduler: 通过");
    return 0;
}