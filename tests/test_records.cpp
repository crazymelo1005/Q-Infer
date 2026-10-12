// 请求记录环：begin 建记录（id 12 位 hex、state=queued、大输入截断 256 KiB 并置旗标）、
// finish 填计时与输出（error 空 → completed）、满 100 逐最老、detail 缺 id → false。
// id 恰为 12 位小写 hex 的期望由实现侧 next_id 掩 48 位保证（T14 Ruling 4）。
#include "serve/records.hpp"

#include <cmath>
#include <cstdio>
#include <string>

#include "check.hpp"

using namespace qinfer::serve;

static bool is_hex12(const std::string& s) {
    if (s.size() != 12) return false;
    for (char c : s) {
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) return false;
    }
    return true;
}

int main() {
    // 1. begin：id 12 位 hex、queued、300 KiB 输入截到 256 KiB 并置旗标。
    Records recs;
    RequestBegin b;
    b.path = "/v1/chat/completions";
    b.model = "svr-test";
    b.input = std::string(300000, 'x');
    b.stream = true;
    b.response_format = "json_object";
    b.created = 1000.0;
    const std::string id = recs.begin(b);
    CHECK(is_hex12(id));
    RequestRecord r0;
    CHECK(recs.detail(id, r0));
    CHECK(r0.state == "queued");
    CHECK(r0.input.size() == Records::kTruncBytes && r0.input_truncated);
    CHECK(r0.stream && r0.response_format == "json_object");
    CHECK(std::abs(r0.created - 1000.0) < 1e-9);

    // 2. finish：计时与输出落盘；error 空 → completed。
    RequestFinish f;
    f.text = "你好";
    f.queue_s = 0.5;
    f.first_token_s = 1.2;
    f.wallclock_s = 9.7;
    f.decode_tok_s = 33.0;
    f.prompt_tokens = 20;
    f.output_tokens = 30;
    recs.finish(id, f);
    RequestRecord out;
    CHECK(recs.detail(id, out));
    CHECK(out.state == "completed" && out.output == "你好");
    CHECK(std::abs(out.wallclock_s - 9.7) < 1e-9);
    CHECK(out.finished && !out.output_truncated && out.input_truncated);   // input 仍是被截断的 256 KiB，旗标保持（brief 的 ! 为笔误，已修正）

    // 3. 满环：连同最先那条共 106 条 → 保留最新 100；最老被逐；list 最新在前。
    for (int i = 0; i < 105; ++i) {
        RequestBegin x;
        x.path = "/x";
        x.model = "m";
        x.input = "in";
        x.created = 1000.0 + i;
        const std::string xid = recs.begin(x);
        RequestFinish fx;
        fx.text = std::to_string(i);
        recs.finish(xid, fx);
    }
    CHECK(recs.size() == Records::kCap);
    const auto list = recs.list();
    CHECK(list.size() == Records::kCap);
    CHECK(list.front().output == "104");   // 最新在前
    CHECK(list.back().output == "5");      // 106-100-1 = 第 5 条之后的全在
    RequestRecord gone;
    CHECK(!recs.detail(id, gone));         // 最早一条已被逐

    // 4. 详情缺 id → false；error 非空 → state=error。
    CHECK(!recs.detail("deadbeef0000", out));
    RequestBegin eb;
    eb.path = "/v1/messages";
    eb.model = "m";
    eb.input = "in";
    const std::string eid = recs.begin(eb);
    RequestFinish fe;
    fe.error = "boom";
    recs.finish(eid, fe);
    RequestRecord eout;
    CHECK(recs.detail(eid, eout) && eout.state == "error");

    puts("test_records: OK");
    return 0;
}