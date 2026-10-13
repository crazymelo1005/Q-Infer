// EngineDriver handshake and solo generation, exercised against the mock engine (no GPU).
#include "engine/engine_driver.hpp"

#include <string>
#include <vector>

#include "check.hpp"

#ifndef MOCK_ENGINE_PATH
#error "MOCK_ENGINE_PATH must be defined by CMake"
#endif

using namespace qinfer::engine;

static DriverConfig mock_cfg(const char* scenario) {
    DriverConfig c;
    c.exe = MOCK_ENGINE_PATH;
    c.engine_args = {scenario};
    c.ready_timeout_ms = 5000;
    return c;
}

int main() {
    // 1. start() consumes INFO and stops at READY; max_context comes from READY.
    {
        EngineDriver d;
        std::string err;
        CHECK(EngineDriver::start(mock_cfg("basic"), d, err));
        CHECK(d.max_context() == 65536);
        CHECK(d.alive());
    }

    // 2. Solo generate streams ids and reports the DONE summary. A second generate() on the
    //    same driver must also work (no per-request state leaks).
    {
        EngineDriver d;
        std::string err;
        CHECK(EngineDriver::start(mock_cfg("basic"), d, err));

        GenRequest r;
        r.prompt = {1, 2, 3};
        r.max_new = 4;
        std::vector<int32_t> got;
        GenSummary s;
        CHECK(d.generate(r, [&](int32_t t) { got.push_back(t); return true; }, s, err));
        CHECK(got.size() == 4 && got[0] == 101 && got[3] == 104);
        CHECK(s.finish == "length" && s.generated == 4 && s.prompt_tokens == 3 && !s.cancelled);

        r.max_new = 2;
        std::vector<int32_t> got2;
        GenSummary s2;
        CHECK(d.generate(r, [&](int32_t t) { got2.push_back(t); return true; }, s2, err));
        CHECK(got2.size() == 2 && got2[0] == 101 && got2[1] == 102);
        CHECK(s2.finish == "length" && s2.generated == 2);
    }

    // 3. Returning false cancels: STOP is sent, DONE is still read, summary says cancel.
    {
        EngineDriver d;
        std::string err;
        CHECK(EngineDriver::start(mock_cfg("cancel"), d, err));
        GenRequest r;
        r.prompt = {1, 2, 3};
        r.max_new = 4;
        int seen = 0;
        GenSummary s;
        CHECK(d.generate(r, [&](int32_t) { return ++seen < 2; }, s, err));
        CHECK(s.cancelled && s.finish == "cancel");
    }

    // 4. An ERR with no DONE is an error, and the driver stays usable.
    {
        EngineDriver d;
        std::string err;
        CHECK(EngineDriver::start(mock_cfg("err"), d, err));
        GenRequest r;
        r.prompt = {1};
        r.max_new = 4;
        GenSummary s;
        CHECK(!d.generate(r, [](int32_t) { return true; }, s, err));
        CHECK(err.find("boom") != std::string::npos);
        CHECK(d.alive());
    }

    // 5. The engine dying mid-request surfaces as an error, never a hang.
    {
        EngineDriver d;
        std::string err;
        CHECK(EngineDriver::start(mock_cfg("crash"), d, err));
        GenRequest r;
        r.prompt = {1};
        r.max_new = 4;
        GenSummary s;
        CHECK(!d.generate(r, [](int32_t) { return true; }, s, err));
        CHECK(!d.alive());
    }
    return 0;
}
