// EngineProcess spawns the engine and drives its stdin/stdout line protocol. No GPU and no
// Strata binary: /bin/sh stands in as the "engine", so this runs anywhere.
#include "engine/engine_process.hpp"

#include <string>

#include "check.hpp"

using namespace qinfer::engine;

int main() {
    // Echo server: writes back each line it reads.
    ProcessConfig c;
    c.exe = "/bin/sh";
    c.args = {"-c", "while read -r l; do echo \"GOT $l\"; done"};
    EngineProcess p;
    std::string err;
    CHECK(EngineProcess::start(c, p, err));
    CHECK(err.empty());

    CHECK(p.write_line("hello", err));
    std::string line;
    CHECK(p.read_line(line, err));
    CHECK(line == "GOT hello");
    CHECK(p.alive());
    p.kill();
    CHECK(!p.alive());

    // Reading past EOF reports failure, not a hang and not a crash.
    ProcessConfig q;
    q.exe = "/bin/sh";
    q.args = {"-c", "echo one; echo two; exit 0"};
    EngineProcess e;
    CHECK(EngineProcess::start(q, e, err));
    CHECK(e.read_line(line, err) && line == "one");
    CHECK(e.read_line(line, err) && line == "two");
    CHECK(!e.read_line(line, err));
    CHECK(!e.alive());

    // A nonexistent exe fails to start with a message, never aborts. exec runs in the child,
    // so start() must detect the failure synchronously (exec-error pipe).
    ProcessConfig bad;
    bad.exe = "/nonexistent/strata";
    EngineProcess z;
    err.clear();
    CHECK(!EngineProcess::start(bad, z, err));
    CHECK(!err.empty());
    return 0;
}
