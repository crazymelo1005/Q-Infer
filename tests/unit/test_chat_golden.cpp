// Byte-for-byte parity of our chat renderer against the reference engine's chat_golden.json
// (Strata serve/chat_golden.json + chat_template.jinja, MIT; see NOTICE). The template is fed
// as data, so this needs no weights and runs in CI.
//
// Cases our renderer cannot express yet are listed in known_gaps and must be exactly those:
// anything else must match byte for byte, so a new mismatch fails the run.
#include "serve/json.hpp"
#include "serve/prompt.hpp"

#include <cstdio>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "check.hpp"

using namespace qinfer::serve;

static std::string slurp(const std::string& path, bool& ok) {
    std::ifstream in(path, std::ios::binary);
    ok = in.good();
    if (!ok) return {};
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

int main(int argc, char** argv) {
    const std::string base = argc > 1 ? argv[1] : "tests/data";

    bool ok = false;
    const std::string tpl = slurp(base + "/chat_template.jinja", ok);
    CHECK(ok);
    const std::string golden_text = slurp(base + "/chat_golden.json", ok);
    CHECK(ok);

    Json root;
    std::string err;
    CHECK(json_parse(golden_text, root, err));
    CHECK(root.is_arr());

    // The renderer carries every input the template reads (it grew `reasoning_content` and the
    // extra top-level variables to close these cases), so nothing may be skipped here anymore.
    const std::set<std::string> known_gaps = {};

    std::set<std::string> gaps;
    int matched = 0;
    for (const Json& c : root.arr) {
        const Json* nm = c.find("name");
        const Json* msgs = c.find("messages");
        const Json* rendered = c.find("rendered");
        CHECK(nm && msgs && rendered);
        const std::string name = nm->str;

        std::vector<Message> m;
        for (const Json& jm : msgs->arr) {
            Message mm;
            const Json* r = jm.find("role");
            const Json* ct = jm.find("content");
            const Json* rc = jm.find("reasoning_content");
            const Json* tc = jm.find("tool_calls");
            mm.role = r && r->is_str() ? r->str : "";
            mm.content = ct && ct->is_str() ? ct->str : "";
            if (rc && rc->is_str()) mm.reasoning_content = rc->str;
            if (tc && tc->is_arr()) mm.tool_calls = tc->arr;
            m.push_back(std::move(mm));
        }
        const Json* agp = c.find("add_generation_prompt");
        const bool gen = agp ? agp->b : true;
        const Json* tools = c.find("tools");
        // enable_thinking and friends are plain template variables, handed in as `vars`.
        const Json* et = c.find("enable_thinking");
        Json vars;
        if (et) {
            vars = Json::objv();
            vars.set("enable_thinking", *et);
        }

        std::string out;
        if (!render_chat(tpl, m, gen, out, err, tools, et ? &vars : nullptr)) {
            std::fprintf(stderr, "chat_golden: case '%s' failed to render: %s\n", name.c_str(),
                         err.c_str());
            gaps.insert(name);
            continue;
        }
        if (out != rendered->str) {
            std::size_t i = 0;
            while (i < out.size() && i < rendered->str.size() && out[i] == rendered->str[i]) ++i;
            std::fprintf(stderr,
                         "chat_golden: case '%s' differs at byte %zu (got %zu bytes, want %zu)\n"
                         "  got  ...%s\n  want ...%s\n",
                         name.c_str(), i, out.size(), rendered->str.size(),
                         out.substr(i, 32).c_str(), rendered->str.substr(i, 32).c_str());
            gaps.insert(name);
            continue;
        }
        ++matched;
    }

    std::printf("chat_golden: %d/%zu matched, gaps: %zu\n", matched, root.arr.size(), gaps.size());
    CHECK(gaps == known_gaps);
    return 0;
}
