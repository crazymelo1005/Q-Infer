#include "serve/scheduler.hpp"

#include <algorithm>

namespace qinfer::serve {

// ---- EventBuffer ----
bool EventBuffer::push(const JobEvent& e) {
    std::unique_lock<std::mutex> lk(mu_);
    cv_full_.wait(lk, [&] { return stopped_ || cancelled_ || q_.size() < cap_; });
    if (stopped_ || cancelled_) return false;
    q_.push_back(e);
    cv_empty_.notify_one();
    return true;
}

bool EventBuffer::pop(JobEvent& out) {
    std::unique_lock<std::mutex> lk(mu_);
    cv_empty_.wait(lk, [&] { return cancelled_ || !q_.empty() || stopped_; });
    if (cancelled_ || q_.empty()) return false;
    out = std::move(q_.front());
    q_.pop_front();
    cv_full_.notify_one();
    return true;
}

void EventBuffer::stop() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        stopped_ = true;
    }
    cv_empty_.notify_all();
    cv_full_.notify_all();
}

void EventBuffer::stop_with_error(std::string err) {
    if (!err.empty()) push(JobEvent{EventKind::kError, {}, {}, err});
    {
        std::lock_guard<std::mutex> lk(mu_);
        err_ = std::move(err);
        stopped_ = true;
    }
    cv_empty_.notify_all();
    cv_full_.notify_all();
}

void EventBuffer::cancel() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        cancelled_ = true;
    }
    cv_empty_.notify_all();
    cv_full_.notify_all();
}

bool EventBuffer::cancelled() const {
    std::lock_guard<std::mutex> lk(mu_);
    return cancelled_;
}

std::string EventBuffer::error() const {
    std::lock_guard<std::mutex> lk(mu_);
    return err_;
}

// ---- Scheduler ----
Scheduler::Scheduler(std::size_t n_slots, std::size_t queue_capacity)
    : qcap_(queue_capacity < 1 ? 1 : queue_capacity), slots_(n_slots < 1 ? 1 : n_slots) {}

Scheduler::~Scheduler() { stop(); }

bool Scheduler::enqueue(std::shared_ptr<Job> job) {
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (stop_ || q_.size() >= qcap_) return false;
        job->enqueued_at = std::chrono::steady_clock::now();
        q_.push_back(std::move(job));
    }
    cv_.notify_one();
    return true;
}

std::shared_ptr<Job> Scheduler::take() {
    std::unique_lock<std::mutex> lk(mu_);
    cv_.wait(lk, [&] { return stop_ || (!q_.empty() && std::any_of(slots_.begin(), slots_.end(),
                                                                    [](const auto& j) { return !j; })); });
    if (stop_) return nullptr;
    auto it = std::find_if(slots_.begin(), slots_.end(), [](const auto& j) { return !j; });
    std::shared_ptr<Job> job = std::move(q_.front());
    q_.pop_front();
    job->leased_at = std::chrono::steady_clock::now();
    *it = job;
    return job;
}

void Scheduler::done(std::shared_ptr<Job> job) {
    {
        std::lock_guard<std::mutex> lk(mu_);
        for (auto& s : slots_)
            if (s == job) s.reset();
    }
    cv_.notify_all();
}

void Scheduler::stop() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        stop_ = true;
    }
    cv_.notify_all();
}

std::size_t Scheduler::queued() const {
    std::lock_guard<std::mutex> lk(mu_);
    return q_.size();
}

bool Scheduler::any_busy() const {
    std::lock_guard<std::mutex> lk(mu_);
    for (const auto& s : slots_)
        if (s) return true;
    return false;
}

std::vector<SlotView> Scheduler::slots_view() const {
    std::lock_guard<std::mutex> lk(mu_);
    const auto now = std::chrono::steady_clock::now();
    std::vector<SlotView> out;
    out.reserve(slots_.size());
    for (std::size_t i = 0; i < slots_.size(); ++i) {
        SlotView v;
        v.index = static_cast<int>(i);
        if (slots_[i]) {
            v.busy = true;
            v.job_id = slots_[i]->id;
            v.queue_s = std::chrono::duration<double>(slots_[i]->leased_at - slots_[i]->enqueued_at).count();
            v.elapsed_s = std::chrono::duration<double>(now - slots_[i]->leased_at).count();
            v.tok_s = slots_[i]->tok_s;
        }
        out.push_back(v);
    }
    return out;
}

bool Scheduler::is_stopped() const {
    std::lock_guard<std::mutex> lk(mu_);
    return stop_;
}

}  // namespace qinfer::serve