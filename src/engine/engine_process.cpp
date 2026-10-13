#include "engine/engine_process.hpp"

#include <utility>

#if defined(__unix__)
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstring>
#include <fcntl.h>
#include <poll.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace qinfer::engine {

EngineProcess::~EngineProcess() { kill(); }

EngineProcess::EngineProcess(EngineProcess&& o) noexcept
    : in_fd_(o.in_fd_), out_fd_(o.out_fd_), pid_(o.pid_),
      out_buf_(std::move(o.out_buf_)), eof_(o.eof_) {
    o.in_fd_ = -1;
    o.out_fd_ = -1;
    o.pid_ = -1;
}

EngineProcess& EngineProcess::operator=(EngineProcess&& o) noexcept {
    if (this != &o) {
        kill();
        in_fd_ = o.in_fd_;
        out_fd_ = o.out_fd_;
        pid_ = o.pid_;
        out_buf_ = std::move(o.out_buf_);
        eof_ = o.eof_;
        o.in_fd_ = -1;
        o.out_fd_ = -1;
        o.pid_ = -1;
    }
    return *this;
}

long long EngineProcess::pid() const { return pid_; }

#if defined(__unix__)

namespace {
void close_fd(int& fd) {
    if (fd >= 0) {
        ::close(fd);
        fd = -1;
    }
}
}  // namespace

bool EngineProcess::start(const ProcessConfig& cfg, EngineProcess& out, std::string& err) {
    err.clear();

    int in_pipe[2] = {-1, -1};    // [0] child stdin read, [1] parent write
    int out_pipe[2] = {-1, -1};   // [0] parent read, [1] child stdout write
    int err_pipe[2] = {-1, -1};   // exec-error channel: child writes errno, closes on exec
    if (::pipe2(in_pipe, O_CLOEXEC) != 0 || ::pipe2(out_pipe, O_CLOEXEC) != 0 ||
        ::pipe2(err_pipe, O_CLOEXEC) != 0) {
        err = std::string("pipe2: ") + std::strerror(errno);
        close_fd(in_pipe[0]); close_fd(in_pipe[1]);
        close_fd(out_pipe[0]); close_fd(out_pipe[1]);
        close_fd(err_pipe[0]); close_fd(err_pipe[1]);
        return false;
    }

    // Open stderr's target in the parent; O_CLOEXEC so the original fd closes on a successful
    // exec (the dup2 onto fd 2 clears CLOEXEC on the copy, which stays open).
    int err_fd = ::open(cfg.stderr_path.empty() ? "/dev/null" : cfg.stderr_path.c_str(),
                        O_CREAT | O_APPEND | O_WRONLY | O_CLOEXEC, 0644);
    if (err_fd < 0) err_fd = ::open("/dev/null", O_WRONLY | O_CLOEXEC);

    // Build argv in the parent: nothing may allocate in the child after fork.
    std::vector<std::string> arg_strings;
    arg_strings.reserve(cfg.args.size() + 1);
    arg_strings.emplace_back(cfg.exe);
    for (const std::string& a : cfg.args) arg_strings.push_back(a);
    std::vector<char*> argv;
    argv.reserve(arg_strings.size() + 1);
    for (std::string& a : arg_strings) argv.push_back(a.data());
    argv.push_back(nullptr);

    const pid_t pid = ::fork();
    if (pid < 0) {
        err = std::string("fork: ") + std::strerror(errno);
        close_fd(in_pipe[0]); close_fd(in_pipe[1]);
        close_fd(out_pipe[0]); close_fd(out_pipe[1]);
        close_fd(err_pipe[0]); close_fd(err_pipe[1]);
        close_fd(err_fd);
        return false;
    }

    if (pid == 0) {
        // Child: only dup2/execvp/_exit (plus the errno write) — no allocation.
        ::dup2(in_pipe[0], STDIN_FILENO);
        ::dup2(out_pipe[1], STDOUT_FILENO);
        if (err_fd >= 0) ::dup2(err_fd, STDERR_FILENO);
        ::execvp(cfg.exe.c_str(), argv.data());
        const int exec_errno = errno;
        const ssize_t ignored = ::write(err_pipe[1], &exec_errno, sizeof(exec_errno));
        (void)ignored;
        ::_exit(127);
    }

    // Parent: keep our ends, drop the child's ends.
    close_fd(in_pipe[0]);
    close_fd(out_pipe[1]);
    close_fd(err_pipe[1]);
    close_fd(err_fd);

    int exec_errno = 0;
    ssize_t got;
    do {
        got = ::read(err_pipe[0], &exec_errno, sizeof(exec_errno));
    } while (got < 0 && errno == EINTR);
    close_fd(err_pipe[0]);

    if (got > 0) {
        // exec failed: reap the child and report the reason.
        int status = 0;
        while (::waitpid(pid, &status, 0) < 0 && errno == EINTR) {}
        err = "exec " + cfg.exe + ": " + std::strerror(exec_errno);
        close_fd(in_pipe[1]);
        close_fd(out_pipe[0]);
        return false;
    }

    out.in_fd_ = in_pipe[1];
    out.out_fd_ = out_pipe[0];
    out.pid_ = pid;
    out.out_buf_.clear();
    out.eof_ = false;
    return true;
}

bool EngineProcess::write_line(const std::string& line, std::string& err) {
    err.clear();
    const std::string data = line + "\n";
    std::size_t off = 0;
    while (off < data.size()) {
        const ssize_t n = ::write(in_fd_, data.data() + off, data.size() - off);
        if (n > 0) {
            off += static_cast<std::size_t>(n);
            continue;
        }
        if (n < 0 && errno == EINTR) continue;
        err = std::string("write: ") + std::strerror(errno);
        return false;
    }
    return true;
}

bool EngineProcess::read_line(std::string& line, std::string& err) {
    err.clear();
    for (;;) {
        const std::size_t nl = out_buf_.find('\n');
        if (nl != std::string::npos) {
            line = out_buf_.substr(0, nl);
            out_buf_.erase(0, nl + 1);
            if (!line.empty() && line.back() == '\r') line.pop_back();
            return true;
        }
        if (eof_) return false;
        char buf[4096];
        const ssize_t n = ::read(out_fd_, buf, sizeof(buf));
        if (n > 0) {
            out_buf_.append(buf, static_cast<std::size_t>(n));
            continue;
        }
        if (n == 0) {
            eof_ = true;
            return false;
        }
        if (errno == EINTR) continue;
        err = std::string("read: ") + std::strerror(errno);
        return false;
    }
}

bool EngineProcess::read_line_timeout(std::string& line, std::string& err, int ms) {
    err.clear();
    const std::chrono::steady_clock::time_point deadline =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(ms > 0 ? ms : 0);
    for (;;) {
        if (out_buf_.find('\n') != std::string::npos || eof_) return read_line(line, err);
        const auto now = std::chrono::steady_clock::now();
        if (now >= deadline) {
            err = "timed out waiting for a line";
            return false;
        }
        const int remaining = static_cast<int>(
            std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now).count());
        pollfd pfd;
        pfd.fd = out_fd_;
        pfd.events = POLLIN;
        pfd.revents = 0;
        const int r = ::poll(&pfd, 1, remaining > 0 ? remaining : 1);
        if (r < 0) {
            if (errno == EINTR) continue;
            err = std::string("poll: ") + std::strerror(errno);
            return false;
        }
        if (r == 0) {
            err = "timed out waiting for a line";
            return false;
        }
        return read_line(line, err);  // data (or EOF) is ready; the read returns promptly
    }
}

bool EngineProcess::alive() const {
    if (pid_ <= 0) return false;
    const pid_t pid = static_cast<pid_t>(pid_);
    int status = 0;
    pid_t r = ::waitpid(pid, &status, WNOHANG);
    if (r == 0 && eof_) {
        // stdout already closed: the child is exiting. Give it a moment so a liveness check
        // right after EOF does not read an about-to-be-reaped zombie as alive.
        for (int waited = 0; waited < 200 && r == 0; waited += 5) {
            ::usleep(5000);
            r = ::waitpid(pid, &status, WNOHANG);
        }
    }
    if (r == pid) {
        pid_ = -1;
        return false;
    }
    if (r < 0) {  // ECHILD or another error: no child to wait on.
        pid_ = -1;
        return false;
    }
    return true;
}

int EngineProcess::wait_timeout(int ms) {
    if (pid_ <= 0) return 0;
    const pid_t pid = static_cast<pid_t>(pid_);
    const int step_ms = 5;
    int waited = 0;
    for (;;) {
        int status = 0;
        const pid_t r = ::waitpid(pid, &status, WNOHANG);
        if (r == pid) {
            pid_ = -1;
            return WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status);
        }
        if (r < 0) {
            pid_ = -1;
            return -1;
        }
        if (ms >= 0 && waited >= ms) return -1;
        ::usleep(step_ms * 1000);
        waited += step_ms;
    }
}

void EngineProcess::kill() {
    close_fd(in_fd_);
    close_fd(out_fd_);
    if (pid_ > 0) {
        const pid_t pid = static_cast<pid_t>(pid_);
        ::kill(pid, SIGKILL);
        int status = 0;
        for (int waited = 0; waited < 2000; waited += 5) {
            const pid_t r = ::waitpid(pid, &status, WNOHANG);
            if (r == pid || r < 0) break;
            ::usleep(5000);
        }
        pid_ = -1;  // SIGKILL cannot be ignored; do not wait on this pid again
    }
}

#else  // !__unix__

bool EngineProcess::start(const ProcessConfig&, EngineProcess&, std::string& err) {
    err = "EngineProcess is POSIX-only in this build";
    return false;
}

bool EngineProcess::write_line(const std::string&, std::string& err) {
    err = "EngineProcess is POSIX-only in this build";
    return false;
}

bool EngineProcess::read_line(std::string&, std::string& err) {
    err = "EngineProcess is POSIX-only in this build";
    return false;
}

bool EngineProcess::read_line_timeout(std::string&, std::string& err, int) {
    err = "EngineProcess is POSIX-only in this build";
    return false;
}

bool EngineProcess::alive() const { return false; }
int EngineProcess::wait_timeout(int) { return -1; }
void EngineProcess::kill() {}

#endif  // __unix__

}  // namespace qinfer::engine
