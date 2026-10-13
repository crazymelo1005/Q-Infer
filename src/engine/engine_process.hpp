// Spawn a child process and drive it over two pipes with a line protocol.
//
// POSIX-only: on a non-Unix configure start() reports unavailability and never forks, so the
// header still compiles there (no behavior is claimed for those platforms).
#pragma once

#include <string>
#include <vector>

namespace qinfer::engine {

struct ProcessConfig {
    std::string exe;
    std::vector<std::string> args;      // already split; "--serve" is added by the caller
    std::string stderr_path;            // empty = /dev/null
};

class EngineProcess {
public:
    EngineProcess() = default;
    ~EngineProcess();                   // kill() + reap
    EngineProcess(EngineProcess&&) noexcept;
    EngineProcess& operator=(EngineProcess&&) noexcept;
    EngineProcess(const EngineProcess&) = delete;
    EngineProcess& operator=(const EngineProcess&) = delete;

    // Fork+exec; waits for the child's exec to succeed or fail so a bad exe is reported here
    // (exec-error pipe) rather than surfacing later as a silent EOF.
    static bool start(const ProcessConfig&, EngineProcess& out, std::string& err);

    bool write_line(const std::string& line, std::string& err);   // appends '\n', writes all
    bool read_line(std::string& line, std::string& err);          // blocking; false at EOF
    // Like read_line, but returns false with a "timed out" err if no line arrives within ms.
    // false with an empty err still means EOF. Used for the engine's bounded READY handshake.
    bool read_line_timeout(std::string& line, std::string& err, int ms);
    bool alive() const;                 // false once the child has exited (reaps a zombie)
    int wait_timeout(int ms);           // -1 still running / error, else exit code; reaps
    void kill();                        // SIGKILL then reap; idempotent

    // Child pid, or -1. long long keeps <sys/types.h> out of this header.
    long long pid() const;

private:
    int in_fd_ = -1;        // child's stdin (we write)
    int out_fd_ = -1;       // child's stdout (we read)
    // alive() is const but must reap: a reaped zombie otherwise reads as "alive". Hence mutable.
    mutable long long pid_ = -1;
    std::string out_buf_;   // stdout bytes read but not yet returned as a line
    bool eof_ = false;
};

}  // namespace qinfer::engine
