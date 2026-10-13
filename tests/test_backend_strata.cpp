// StrataBackend over the mock engine: no GPU, no weights. A tiny vocabulary maps the mock's
// scripted ids 101..104 to "a".."d" so the id -> text path is observable.
#include "serve/backend_strata.hpp"

#include <string>
#include <vector>

#include "check.hpp"

#ifndef MOCK_ENGINE_PATH
#error "MOCK_ENGINE_PATH must be defined by CMake"
#endif

using namespace qinfer::serve;
using qinfer::text::Tokenizer;

static bool tiny_tokenizer(Tokenizer& t, std::string& err) {
    std::vector<std::string> toks;
    toks.reserve(110);
    for (int i = 0; i < 110; ++i) toks.push_back("x" + std::to_string(i));
    toks[101] = "a";
    toks[102] = "b";
    toks[103] = "c";
    toks[104] = "d";
    return t.init(std::move(toks), {}, {}, err);
}

int main() {
    StrataBackendConfig c;
    c.driver.exe = MOCK_ENGINE_PATH;
    c.driver.engine_args = {"basic"};
    c.driver.ready_timeout_ms = 5000;
    std::string err;
    CHECK(tiny_tokenizer(c.tokenizer, err));
    c.chat_template = "TPL";
    c.model_name = "mock";

    StrataBackend b;
    CHECK(StrataBackend::open(std::move(c), b, err));
    CHECK(b.config().max_context == 65536);   // READY overrode the FacadeConfig default
    CHECK(b.model_name() == "mock");
    CHECK(b.chat_template() == "TPL");
    CHECK(b.tokenizer().vocab_size() == 110);
    CHECK(b.kv_misses() == 0 && b.kv_evictions() == 0 && b.kv_bytes_moved() == 0);

    GenOptions o;
    o.max_tokens = 4;
    std::string streamed;
    GenResult r;
    CHECK(b.generate({1, 2, 3}, o, [&](const std::string& s) { streamed += s; return true; }, r, err));
    CHECK(r.n_prompt == 3);
    CHECK(r.n_gen == 4);
    CHECK(r.text == "abcd");
    CHECK(streamed == "abcd");
    CHECK(r.tokens.size() == 4 && r.tokens[0] == 101 && r.tokens[3] == 104);
    CHECK(r.finish == Finish::kLength);

    // A second call on the same backend works (no per-request state leaks).
    std::string streamed2;
    GenResult r2;
    CHECK(b.generate({1}, o, [&](const std::string& s) { streamed2 += s; return true; }, r2, err));
    CHECK(r2.text == "abcd" && streamed2 == "abcd");

    // Returning false from the callback is a client disconnect: generate fails.
    GenResult r3;
    const bool ok = b.generate({1}, o, [](const std::string&) { return false; }, r3, err);
    CHECK(!ok);
    CHECK(!err.empty());

    // --- incremental UTF-8: emit whole characters only ---
    CHECK(utf8_complete_prefix("") == 0);
    CHECK(utf8_complete_prefix("abc") == 3);
    CHECK(utf8_complete_prefix("\xE4\xB8\xAD") == 3);   // "中", three bytes
    CHECK(utf8_complete_prefix("\xE4\xB8") == 0);       // partial tail is held back
    CHECK(utf8_complete_prefix("\xE4\xB8\xADx") == 4);
    CHECK(utf8_complete_prefix("\xFF") == 1);           // invalid lead byte: emit one, never stall

    // --- stop string: cut at the hit, report kStopString, don't emit the suffix ---
    {
        StrataBackendConfig c2;
        c2.driver.exe = MOCK_ENGINE_PATH;
        c2.driver.engine_args = {"basic"};
        c2.driver.ready_timeout_ms = 5000;
        CHECK(tiny_tokenizer(c2.tokenizer, err));
        StrataBackend b2;
        CHECK(StrataBackend::open(std::move(c2), b2, err));

        GenOptions os;
        os.max_tokens = 4;
        os.stop = {"c"};  // the mock streams "a","b","c","d"
        std::string s2;
        GenResult r4;
        CHECK(b2.generate({1}, os, [&](const std::string& s) { s2 += s; return true; }, r4, err));
        CHECK(r4.finish == Finish::kStopString);
        CHECK(r4.text == "ab");
        CHECK(s2 == "ab");
        CHECK(r4.tokens.size() == 2);   // the cut token is not counted as produced
    }
    return 0;
}
