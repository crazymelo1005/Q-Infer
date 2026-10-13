#include "serve/server.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <sstream>

#include "httplib.h"
#include "serve/json.hpp"

namespace qinfer::serve {

namespace {

// ---- Request body parameter reading (all invalid parameters per spec §5 → 400 semantics) ----
bool num_param(const Json& body, const char* key, double dflt, double& out) {
    const Json* v = body.find(key);
    if (v == nullptr) { out = dflt; return true; }
    if (!v->is_num()) return false;
    out = v->num;
    return true;
}

bool str_param(const Json& body, const char* key, std::string& out, bool& present) {
    const Json* v = body.find(key);
    present = v != nullptr;
    if (v == nullptr) return true;
    if (!v->is_str()) return false;
    out = v->str;
    return true;
}

// stop: missing → empty; string → one; array → all strings; other → false (400).
bool stop_param(const Json& body, std::vector<std::string>& out) {
    const Json* v = body.find("stop");
    out.clear();
    if (v == nullptr) return true;
    if (v->is_str()) { out.push_back(v->str); return true; }
    if (v->is_arr()) {
        for (const Json& e : v->arr) {
            if (!e.is_str()) return false;
            out.push_back(e.str);
        }
        return true;
    }
    return false;
}

// ---- Error body (the two domain shapes of spec §5) ----
Json openai_err(const std::string& message, const std::string& type, const char* param) {
    Json inner = Json::objv();
    inner.set("message", Json::strv(message));
    inner.set("type", Json::strv(type));
    inner.set("param", param == nullptr ? Json::nullv() : Json::strv(param));
    Json e = Json::objv();
    e.set("error", std::move(inner));
    return e;
}

Json anthropic_err(const std::string& type, const std::string& message) {
    Json inner = Json::objv();
    inner.set("type", Json::strv(type));
    inner.set("message", Json::strv(message));
    Json e = Json::objv();
    e.set("type", Json::strv("error"));
    e.set("error", std::move(inner));
    return e;
}

void send_json(httplib::Response& res, int status, const Json& body) {
    res.status = status;
    res.set_content(json_dump(body), "application/json; charset=utf-8");
}

// ---- Record serialization (list view omits large fields, detail includes all; same as Strata request_records) ----
Json record_summary_json(const RequestRecord& r) {
    Json o = Json::objv();
    o.set("id", Json::strv(r.id));
    o.set("path", Json::strv(r.path));
    o.set("model", Json::strv(r.model));
    o.set("created", Json::numv(r.created));
    o.set("state", Json::strv(r.state));
    o.set("stream", Json::boolv(r.stream));
    o.set("response_format", r.response_format.empty()
                                 ? Json::nullv() : Json::strv(r.response_format));
    o.set("queue_s", Json::numv(r.queue_s));
    o.set("first_token_s", r.first_token_s > 0 ? Json::numv(r.first_token_s) : Json::nullv());
    o.set("wallclock_s", Json::numv(r.wallclock_s));
    o.set("decode_tok_s", Json::numv(r.decode_tok_s));
    o.set("prompt_tokens", Json::numv(static_cast<double>(r.prompt_tokens)));
    o.set("output_tokens", Json::numv(static_cast<double>(r.output_tokens)));
    return o;
}

Json record_detail_json(const RequestRecord& r) {
    Json o = record_summary_json(r);
    o.set("input", Json::strv(r.input));
    o.set("input_truncated", Json::boolv(r.input_truncated));
    o.set("output", Json::strv(r.output));
    o.set("output_truncated", Json::boolv(r.output_truncated));
    o.set("reasoning", Json::strv(r.reasoning));
    o.set("reasoning_truncated", Json::boolv(r.reasoning_truncated));
    return o;
}

// History series (from log. Each point: a value or null).
Json series_json(const std::vector<std::optional<double>>& s) {
    Json a = Json::arrv();
    for (const auto& v : s) a.push(v ? Json::numv(*v) : Json::nullv());
    return a;
}

// One decimal place (Strata's round(x, 1)), display only.
double round1(double v) { return std::round(v * 10.0) / 10.0; }

bool anthropic_path(const std::string& p) {
    return p == "/v1/messages" || p == "/v1/messages/count_tokens";
}

std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    std::ostringstream s;
    s << f.rdbuf();
    return s.str();
}

// vector<Json> → Json array (the shape of render_chat's tools argument; an empty array is semantically
// equivalent to nullptr, callers uniformly test tools.empty() — omitted by default, the template's
// tools branch is never entered).
Json tools_array(const std::vector<Json>& tools) {
    Json a = Json::arrv();
    for (const Json& t : tools) a.push(t);
    return a;
}

// Run the stop-and-wait protocol once (worker-side semantics in worker_main): read kToken to
// accumulate text, kDone/kError to finish. HTTP-thread only (blocking pop). Returns false = error
// (err carries the message).
bool drain_nonstream(const std::shared_ptr<Job>& job, std::string& text, std::string& err) {
    JobEvent e;
    for (;;) {
        if (!job->events->pop(e)) { err = "Event buffer terminated abnormally"; return false; }
        if (e.kind == EventKind::kToken) { text += e.text; continue; }
        if (e.kind == EventKind::kDone) return true;
        // kError
        err = e.err;
        return false;
    }
}

// Conversation assembly (Anthropic domain): system + messages (accepts user/assistant/tool; content
// accepts a string or block array — text blocks are concatenated, thinking blocks are skipped,
// tool_use blocks → calls, tool_result blocks → inline tool messages; image blocks etc. are all
// rejected) → the same Message list as the OpenAI domain goes to render_chat. The two compatibility
// domains producing the same token sequence for the same request (M1.3 hard requirement) rests on
// this single rendering path. Tool-call semantics cross-check Strata's anthropic_to_messages:
// tool_use's input is converted via tool_arguments to the template parameter shape (malformed JSON
// is rejected cleanly rather than derailing the whole conversation, #510), and tool_result is split
// into per-item {"role":"tool"} messages.
bool messages_to_conversation(const Json& body, std::vector<Message>& conv, std::string& err) {
    const Json* sys = body.find("system");
    if (sys != nullptr) {
        if (!sys->is_str()) { err = "system must be a string"; return false; }
        conv.emplace_back(Message{"system", sys->str});
    }
    const Json* msgs = body.find("messages");
    if (msgs == nullptr || !msgs->is_arr()) { err = "Missing messages array"; return false; }
    for (const Json& m : msgs->arr) {
        if (!m.is_obj()) { err = "messages element is not an object"; return false; }
        const Json* role = m.find("role");
        if (role == nullptr || !role->is_str()) { err = "Missing role string"; return false; }
        const std::string& r = role->str;
        if (r != "user" && r != "assistant" && r != "tool") {
            err = "role accepts only user/assistant/tool";
            return false;
        }
        const Json* content = m.find("content");
        if (content == nullptr) { err = "Missing content"; return false; }
        std::string text;
        std::vector<Json> calls;   // tool_use blocks in the assistant message (template shape, see prompt.hpp)
        if (content->is_str()) {
            text = content->str;
        } else if (content->is_arr()) {
            for (const Json& blk : content->arr) {
                if (!blk.is_obj()) { err = "content block is not an object"; return false; }
                const Json* ty = blk.find("type");
                if (ty == nullptr || !ty->is_str()) { err = "content block missing type"; return false; }
                if (ty->str == "text") {
                    const Json* t = blk.find("text");
                    if (t == nullptr || !t->is_str()) { err = "Text block missing text"; return false; }
                    text += t->str;
                } else if (ty->str == "thinking") {
                    // Thinking blocks are skipped (not put into the template; the reasoning_content
                    // handoff is deferred to M2)
                } else if (ty->str == "tool_use") {
                    if (r != "assistant") { err = "tool_use block only appears in assistant messages"; return false; }
                    const Json* name = blk.find("name");
                    if (name == nullptr || !name->is_str()) { err = "tool_use block missing name"; return false; }
                    Json fn = Json::objv();
                    fn.set("name", Json::strv(name->str));
                    // input missing/malformed JSON → tool_arguments falls back to {} or
                    // {"arguments": raw} (#510)
                    fn.set("arguments", tool_arguments(blk.find("input")));
                    Json call = Json::objv();
                    call.set("function", std::move(fn));
                    calls.push_back(std::move(call));
                } else if (ty->str == "tool_result") {
                    if (r != "user") { err = "tool_result block only appears in user messages"; return false; }
                    std::string rt;
                    const Json* rc = blk.find("content");
                    if (rc == nullptr) { err = "tool_result block missing content"; return false; }
                    if (rc->is_str()) {
                        rt = rc->str;
                    } else if (rc->is_arr()) {
                        for (const Json& rb : rc->arr) {
                            if (!rb.is_obj()) { err = "tool_result content block is not an object"; return false; }
                            const Json* rty = rb.find("type");
                            if (rty == nullptr || !rty->is_str() || rty->str != "text") {
                                err = "tool_result content accepts only text blocks (images and other multimodal deferred to M2)";
                                return false;
                            }
                            const Json* t = rb.find("text");
                            if (t == nullptr || !t->is_str()) { err = "tool_result text block missing text"; return false; }
                            rt += t->str;
                        }
                    } else {
                        err = "tool_result content must be a string or block array";
                        return false;
                    }
                    // Inline tool message (each one immediately follows, same as Strata
                    // anthropic_to_messages): at render time the template emits <tool_response> for
                    // role=tool; empty results are skipped by the uniform empty filter.
                    if (rt.empty()) continue;
                    conv.emplace_back(Message{"tool", std::move(rt)});
                } else {
                    err = "content block accepts only text/thinking/tool_use/tool_result (images and other multimodal deferred to M2)";
                    return false;
                }
            }
        } else {
            err = "content must be a string or block array";
            return false;
        }
        // Uniform empty filter: the whole message is skipped only when both text and calls are empty
        // (same as the OpenAI domain).
        if (text.empty() && calls.empty()) continue;
        conv.emplace_back(Message{.role = r, .content = std::move(text), .tool_calls = std::move(calls)});
    }
    return true;
}

// v/unit, rounded to digits decimal places; nullopt → null.
static Json scaled_mib_gib(const std::optional<double>& v, double unit, int digits) {
    if (!v) return Json::nullv();
    const double p = std::pow(10.0, digits);
    return Json::numv(std::round(*v / unit * p) / p);
}

}  // namespace

Serve::Serve(ServeConfig cfg, std::shared_ptr<Backend> eng)
    : cfg_(std::move(cfg)),
      eng_(std::move(eng)),
      sched_(cfg_.slots, cfg_.queue_capacity == 0 ? 2 * cfg_.slots : cfg_.queue_capacity),
      prompt_cache_(cfg_.conversation_cache),
      svr_(std::make_unique<httplib::Server>()),
      started_(std::time(nullptr)) {
    const std::filesystem::path base = std::filesystem::path(__FILE__).parent_path() / "web";
    web_index_ = read_file((base / "index.html").string());
    web_monitor_ = read_file((base / "monitor.html").string());
    // T13: telemetry starts in the constructor (launch is the per-request entry point; placed there,
    // tel_ would be null when there are no requests and the first frame of T14's /metrics would have
    // no data); sampling spans the process lifetime, destruction joins via tel_'s dtor.
    tel_ = std::make_unique<Telemetry>([this] { return telemetry_extra(); });
    tel_->start();
}

Serve::~Serve() { stop(); }

void Serve::worker_main() {
    for (;;) {
        std::shared_ptr<Job> job = sched_.take();
        if (!job) break;
        std::string err;
        std::vector<std::int32_t> ids;
        GenResult res;
        bool ok = false;
        {
            // Encoding consults the cache (conversation_cache of spec §6): lookup happens outside the
            // lock; on a hit the cached ids are used without BPE (and without writing the
            // worker-shared piece_cache_); only on a miss is encode run inside the lock and the cache
            // backfilled. generate, as in v0, still serializes inside engine_mu_ (it writes the shared
            // KV); what a hit saves is tokenization, while the lock is still taken. The cache is
            // looked up only outside the lock and backfilled inside it: the lock-order direction is
            // always engine → cache, with no reverse path.
            const bool hit = prompt_cache_.lookup(job->prompt, ids);
            std::lock_guard<std::mutex> lock(engine_mu_);
            if (!hit) {
                if (!eng_->tokenizer().encode(job->prompt, /*parse_special=*/true, ids, err)) {
                    job->finish = "error";
                    job->error = "Tokenization failed: " + err;
                    // Set the completion time before stopping the stream: when the HTTP side reads
                    // done_at it is guaranteed ready (isomorphic to the other error paths); otherwise
                    // the HTTP thread might call record_request before done_at and have it skipped
                    // entirely by the guard.
                    job->done_at = std::chrono::steady_clock::now();
                    job->events->stop_with_error(job->error);
                    sched_.done(job);
                    continue;
                }
                prompt_cache_.store(job->prompt, ids);
            }
            job->prompt_ids = std::move(ids);
            job->n_prompt = job->prompt_ids.size();
            if (hit) reused_.fetch_add(job->n_prompt);   // hit = this whole prompt was encoded for free (totals.reused)
            ok = eng_->generate(job->prompt_ids, job->opts,
                                [&](const std::string& piece) -> bool {
                                    if (job->first_at == std::chrono::steady_clock::time_point{})
                                        job->first_at = std::chrono::steady_clock::now();
                                    // A stalled (not disconnected) SSE client can block the worker on
                                    // push while holding engine_mu_, freezing all slots — an M2 tuning
                                    // item (lock granularity/timeouts), not preventable in M1.
                                    return job->events->push(JobEvent{EventKind::kToken, piece, {}, {}});
                                },
                                res, err);
        }
        if (ok) {
            job->n_gen = res.n_gen;
            job->text = res.text;
            job->finish = res.finish == Finish::kLength ? "length" : "stop";
            job->done_at = std::chrono::steady_clock::now();
            const double span = std::chrono::duration<double>(job->done_at - job->first_at).count();
            job->tok_s = span > 0 ? static_cast<double>(res.n_gen) / span : 0.0;
            job->events->push(JobEvent{EventKind::kDone, {}, job->finish, {}});
            job->events->stop();
        } else {
            job->finish = "error";
            job->error = err;
            job->done_at = std::chrono::steady_clock::now();
            job->events->stop_with_error(err);
        }
        // When all pushes fail (client disconnected), slots are still released as usual; the record
        // is marked disconnected on the HTTP side.
        sched_.done(job);
    }
}

std::shared_ptr<Job> Serve::launch(const std::string& path, std::string prompt,
                                   const GenOptions& opts, bool stream) {
    // reject_after: from the Nth request on, all requests are rejected at enqueue (503). The read of
    // req_id_ happens before job->id = req_id_++, ensuring the Nth request is by arrival ordinal.
    if (cfg_.reject_after > 0 && req_id_.load() >= cfg_.reject_after) return nullptr;
    auto job = std::make_shared<Job>();
    job->id = req_id_++;
    job->path = path;
    job->model = eng_->model_name();
    job->prompt = std::move(prompt);
    job->opts = opts;
    job->stream = stream;
    return sched_.enqueue(job) ? job : nullptr;
}

void Serve::handle_chat(const httplib::Request& req, httplib::Response& res, bool stream) {
    Json body;
    std::string jerr;
    if (!json_parse(req.body, body, jerr) || !body.is_obj()) {
        send_json(res, 400, openai_err("Request body is not a valid JSON object: " + jerr, "invalid_request_error", nullptr));
        return;
    }
    const Json* msgs = body.find("messages");
    if (msgs == nullptr || !msgs->is_arr()) {
        send_json(res, 400, openai_err("Missing messages array", "invalid_request_error", "messages"));
        return;
    }
    // messages → Message list (OpenAI domain): role accepts system/user/assistant/tool (developer is
    // folded into system); content accepts null/string/array of text blocks (images and other
    // multimodal open in M2); the assistant's tool_calls history is normalized to the template shape
    // via parse_openai_tool_calls. Semantics cross-check Strata's openai_to_messages: only messages
    // with both text and calls empty are filtered (an assistant tool_calls message may have empty content).
    std::vector<Message> conv;
    for (const Json& m : msgs->arr) {
        if (!m.is_obj()) { send_json(res, 400, openai_err("messages element is not an object", "invalid_request_error", "messages")); return; }
        const Json* role = m.find("role");
        if (role == nullptr || !role->is_str()) {
            send_json(res, 400, openai_err("Each message requires a string role", "invalid_request_error", "messages"));
            return;
        }
        const std::string r = role->str == "developer" ? "system" : role->str;
        if (r != "system" && r != "user" && r != "assistant" && r != "tool") {
            send_json(res, 400, openai_err("role accepts only system/user/assistant/tool", "invalid_request_error", "messages"));
            return;
        }
        std::string text;
        const Json* content = m.find("content");
        if (content != nullptr && !content->is_null()) {
            if (content->is_str()) {
                text = content->str;
            } else if (content->is_arr()) {
                for (const Json& blk : content->arr) {
                    if (!blk.is_obj() || !blk.find("type") || !blk.find("type")->is_str() ||
                        blk.find("type")->str != "text") {
                        send_json(res, 400, openai_err("content block accepts only {type: text} (images and other multimodal deferred to M2)", "invalid_request_error", "messages"));
                        return;
                    }
                    const Json* t = blk.find("text");
                    if (t == nullptr || !t->is_str()) {
                        send_json(res, 400, openai_err("Text block missing text", "invalid_request_error", "messages"));
                        return;
                    }
                    text += t->str;
                }
            } else {
                send_json(res, 400, openai_err("content must be a string or block array", "invalid_request_error", "messages"));
                return;
            }
        }
        std::vector<Json> calls;
        const Json* tcs = m.find("tool_calls");
        if (tcs != nullptr) {
            std::string cerr;
            if (!parse_openai_tool_calls(*tcs, calls, cerr)) {
                send_json(res, 400, openai_err("tool_calls parse failed: " + cerr, "invalid_request_error", "tool_calls"));
                return;
            }
        }
        if (text.empty() && calls.empty()) continue;   // empty messages (tool placeholders etc.) are filtered; the templater ignores them
        conv.emplace_back(Message{.role = r, .content = std::move(text), .tool_calls = std::move(calls)});
    }
    // tools → internal tool shape (normalize_tools' OpenAI unwrapping); a bad one is 400 (errors must come before enqueue).
    std::vector<Json> tools;
    const Json* tv = body.find("tools");
    if (tv != nullptr) {
        std::string terr;
        if (!normalize_tools(*tv, /*anthropic=*/false, tools, terr)) {
            send_json(res, 400, openai_err("tools parse failed: " + terr, "invalid_request_error", "tools"));
            return;
        }
    }
    // Ruling ⑤: empty prompt rejected — after all messages are filtered there is still nothing renderable → 400.
    if (conv.empty()) {
        send_json(res, 400, openai_err("messages has no usable content", "invalid_request_error", "messages"));
        return;
    }
    GenOptions opts;
    double t = 0.0, p = 0.0, mt = 0.0;
    std::int64_t seed = -1;
    bool numeric_ok = num_param(body, "temperature", 0.8, t) && num_param(body, "top_p", 0.95, p) &&
                      num_param(body, "max_tokens", 512, mt);
    const Json* seedv = body.find("seed");
    if (seedv != nullptr) {
        if (!seedv->is_num()) numeric_ok = false;
        else seed = static_cast<std::int64_t>(seedv->num);
    }
    if (!numeric_ok) { send_json(res, 400, openai_err("temperature/top_p/max_tokens/seed must be numbers", "invalid_request_error", nullptr)); return; }
    opts.temperature = static_cast<float>(t);
    opts.top_p = static_cast<float>(p);
    opts.seed = seed;
    if (mt <= 0) { send_json(res, 400, openai_err("max_tokens must be positive", "invalid_request_error", "max_tokens")); return; }
    opts.max_tokens = static_cast<std::uint64_t>(mt);
    if (!stop_param(body, opts.stop)) { send_json(res, 400, openai_err("stop must be a string or array of strings", "invalid_request_error", "stop")); return; }
    SamplerParams sp; sp.temperature = opts.temperature; sp.top_p = opts.top_p;
    if (!sp.valid()) { send_json(res, 400, openai_err("temperature must be non-negative and top_p in (0,1]", "invalid_request_error", nullptr)); return; }
    // T10: stream is read from the body by the routing layer and passed in; the body is no longer
    // read here, dispatch is by parameter only (SSE goes through serve_sse).
    std::string prompt;
    std::string rerr;
    Json tj = tools_array(tools);
    if (!render_chat(eng_->chat_template(), conv, /*add_generation_prompt=*/true, prompt, rerr,
                     tools.empty() ? nullptr : &tj)) {
        // The templater is a trimmed subset (fallback per spec §11): no silent mis-rendering, an
        // explicit 500 with explanation.
        send_json(res, 500, openai_err("Structure not supported by the chat template: " + rerr, "server_error", nullptr));
        return;
    }
    std::shared_ptr<Job> job = launch(req.path, prompt, opts, stream);
    if (!job) {
        res.status = 503;
        res.set_content(std::string(kQueueFullBody), "application/json; charset=utf-8");
        // 503 not enqueued: no timing data, not entered into the records ring (the ring only accepts
        // accepted requests).
        return;
    }
    job->raw_body = req.body;   // T14: the raw request body goes into Records (streaming requests too)
    // T10: streaming dispatch — include_usage is accepted leniently (Ruling ①): it counts as true
    // only when stream_options exists, is an object, and its include_usage is boolean true; anything
    // else is false, with no extra 400.
    if (stream) {
        bool include_usage = false;
        const Json* so = body.find("stream_options");
        if (so != nullptr && so->is_obj()) {
            const Json* iu = so->find("include_usage");
            if (iu != nullptr && iu->kind == Json::kBool) include_usage = iu->b;
        }
        serve_sse(job, include_usage, tools, res);
        return;
    }
    std::string text, drain_err;
    const bool ok = drain_nonstream(job, text, drain_err);
    if (!ok) {
        const int st = job->events->cancelled() ? 499 : 500;   // 499 = client disconnected, httplib convention
        if (st == 500) {
            send_json(res, st, openai_err(drain_err, "server_error", nullptr));
        }
        record_request(job, st, job->events->cancelled());
        return;
    }
    // Split generated text into events: <tool_call> blocks → calls; malformed markers are merged back
    // into text (split_generation semantics). Calls and text keep generation order (same as Strata's
    // OutputParser); content is set to null when entirely empty (OpenAI convention).
    std::vector<GenEvent> events;
    split_generation(text, tools, events);
    Json r = Json::objv();
    r.set("id", Json::strv("chatcmpl-" + std::to_string(job->id)));
    r.set("object", Json::strv("chat.completion"));
    r.set("created", Json::numv(static_cast<double>(std::time(nullptr))));
    r.set("model", Json::strv(job->model));
    Json message = Json::objv();
    message.set("role", Json::strv("assistant"));
    std::string content_str;
    Json calls = Json::arrv();
    int call_idx = 0;
    for (const GenEvent& ev : events) {
        if (ev.call) {
            calls.push(openai_tool_call("call_" + std::to_string(job->id) + "_" + std::to_string(call_idx),
                                        ev.call->name, ev.call->arguments));
            ++call_idx;
        } else {
            content_str += ev.text;
        }
    }
    message.set("content", content_str.empty() ? Json::nullv() : Json::strv(content_str));
    const bool used_tool = call_idx > 0;
    if (used_tool) message.set("tool_calls", std::move(calls));
    Json choice = Json::objv();
    choice.set("index", Json::numv(0.0));
    choice.set("message", std::move(message));
    // finish_reason: has calls and stopped naturally → tool_calls (per Strata's openai_chunks).
    choice.set("finish_reason", Json::strv(
        used_tool && job->finish == "stop" ? "tool_calls" : job->finish));
    Json choices = Json::arrv();
    choices.push(std::move(choice));
    r.set("choices", std::move(choices));
    Json usage = Json::objv();
    usage.set("prompt_tokens", Json::numv(static_cast<double>(job->n_prompt)));
    usage.set("completion_tokens", Json::numv(static_cast<double>(job->n_gen)));
    usage.set("total_tokens", Json::numv(static_cast<double>(job->n_prompt + job->n_gen)));
    r.set("usage", std::move(usage));
    send_json(res, 200, r);
    record_request(job, 200, false);
}

// OpenAI-compatible SSE. The provider is called repeatedly by httplib after the handler returns; each
// call writes exactly one frame, and an event can expand into multiple frames (first role frame, text
// deltas, tool-call blocks, final frame/usage/[DONE]); frames are batched into pending as events are
// consumed and drained one at a time. Generated text is split incrementally by ToolCallStreamer — a
// half <tool_call> spanning increments is buffered until complete, and malformed chunks (parse
// failures) are released as text. Frame shapes cross-check Strata's openai_chunks: first frame
// {delta:{role,content:""}}, one frame per whole tool call
// {delta:{tool_calls:[{index,id,type,function}]}}, final frame finish_reason (has calls and stopped
// naturally → tool_calls). Connection lost (write false) → cancel → 499 bookkeeping → return false
// to wrap up.
void Serve::serve_sse(const std::shared_ptr<Job>& job, bool include_usage,
                      const std::vector<Json>& tools, httplib::Response& res) {
    const std::string id = "chatcmpl-" + std::to_string(job->id);
    const double created = static_cast<double>(std::time(nullptr));
    const std::string model = eng_->model_name();
    struct State {
        ToolCallStreamer st;
        // Tool definitions live by value with State on the heap: the provider outlives the
        // handler (httplib calls the chunk provider repeatedly only after the route function
        // returns), so a reference argument would dangle.
        std::vector<Json> tools;
        std::vector<std::string> pending;   // frames pending write (popped from front)
        bool first = true;                  // the first role chunk is not yet written
        int calls = 0;                      // tool-call blocks already emitted (finish_reason criterion + block index)
        bool done = false;                  // kDone/kError already consumed
        int final_status = 200;
        bool recorded = false;
    };
    auto state = std::make_shared<State>();
    state->tools = tools;
    auto provider = [this, job, include_usage, id, created, model, state](
                        std::size_t, httplib::DataSink& sink) -> bool {
        const auto w = [&](const std::string& s) {
            // The vendored DataSink::write is std::function<bool(const char*, size_t)> (two-argument call,
            // see httplib.h:538).
            if (!sink.write(s.data(), s.size())) {
                job->events->cancel();
                if (!state->recorded) { record_request(job, 499, true); state->recorded = true; }
                return false;
            }
            return true;
        };
        const auto chunk = [&](Json delta) {
            Json c = Json::objv();
            c.set("id", Json::strv(id));
            c.set("object", Json::strv("chat.completion.chunk"));
            c.set("created", Json::numv(created));
            c.set("model", Json::strv(model));
            Json choice = Json::objv();
            choice.set("index", Json::numv(0.0));
            choice.set("delta", std::move(delta));
            choice.set("finish_reason", Json::nullv());
            Json choices = Json::arrv();
            choices.push(std::move(choice));
            c.set("choices", std::move(choices));
            return "data: " + json_dump(c) + "\n\n";
        };
        const auto emit_streamer_event = [&](const ToolCallStreamer::Event& sev) {
            if (sev.kind == ToolCallStreamer::Kind::kText) {
                if (sev.text.empty()) return;
                Json d = Json::objv();
                d.set("content", Json::strv(sev.text));
                state->pending.push_back(chunk(std::move(d)));
                return;
            }
            // kBlock: complete call block → tool-call frame (one frame per whole block, the
            // stream_tools=false shape); parse failure (a hallucinated malformed marker) → released
            // as text.
            ToolCall tc;
            std::string perr;
            if (!parse_tool_block(sev.text, state->tools, tc, perr)) {
                Json d = Json::objv();
                d.set("content", Json::strv(sev.text));
                state->pending.push_back(chunk(std::move(d)));
                return;
            }
            Json d = Json::objv();
            Json tcs = Json::arrv();
            tcs.push(openai_tool_call(
                "call_" + std::to_string(job->id) + "_" + std::to_string(state->calls),
                tc.name, tc.arguments));
            d.set("tool_calls", std::move(tcs));
            state->pending.push_back(chunk(std::move(d)));
            ++state->calls;
        };
        // Write one frame; when no frame is writable, first consume job events to accumulate
        // frames (a single event can expand into multiple frames, drained one at a time).
        for (;;) {
            if (!state->pending.empty() || state->done) break;
            JobEvent e;
            if (!job->events->pop(e)) {
                if (!state->recorded) { record_request(job, 499, job->events->cancelled()); state->recorded = true; }
                return false;
            }
            if (e.kind == EventKind::kToken) {
                if (state->first) {
                    Json d = Json::objv();
                    d.set("role", Json::strv("assistant"));
                    d.set("content", Json::strv(""));
                    state->pending.push_back(chunk(std::move(d)));
                    state->first = false;
                }
                std::vector<ToolCallStreamer::Event> evs;
                state->st.feed(e.text, evs);
                for (const auto& sev : evs) emit_streamer_event(sev);
                continue;
            }
            if (e.kind == EventKind::kDone) {
                // Before wrap-up, flush the buffered remainder (an unclosed call is released as its
                // raw text, #211).
                std::vector<ToolCallStreamer::Event> evs;
                state->st.finish(evs);
                for (const auto& sev : evs) emit_streamer_event(sev);
                // Final frame: finish_reason (has calls and stopped naturally → tool_calls); when usage
                // is requested an extra frame comes first, then [DONE].
                Json dc = Json::objv();
                Json choice = Json::objv();
                choice.set("index", Json::numv(0.0));
                choice.set("delta", std::move(dc));
                choice.set("finish_reason", Json::strv(
                    state->calls > 0 && e.finish == "stop" ? "tool_calls"
                                                           : (e.finish.empty() ? "stop" : e.finish)));
                Json choices = Json::arrv();
                choices.push(std::move(choice));
                Json c = Json::objv();
                c.set("id", Json::strv(id));
                c.set("object", Json::strv("chat.completion.chunk"));
                c.set("created", Json::numv(created));
                c.set("model", Json::strv(model));
                c.set("choices", std::move(choices));
                state->pending.push_back("data: " + json_dump(c) + "\n\n");
                if (include_usage) {
                    Json u = Json::objv();
                    u.set("prompt_tokens", Json::numv(static_cast<double>(job->n_prompt)));
                    u.set("completion_tokens", Json::numv(static_cast<double>(job->n_gen)));
                    u.set("total_tokens", Json::numv(static_cast<double>(job->n_prompt + job->n_gen)));
                    Json uc = Json::objv();
                    uc.set("id", Json::strv(id));
                    uc.set("object", Json::strv("chat.completion.chunk"));
                    uc.set("created", Json::numv(created));
                    uc.set("model", Json::strv(model));
                    uc.set("choices", Json::arrv());
                    uc.set("usage", std::move(u));
                    state->pending.push_back("data: " + json_dump(uc) + "\n\n");
                }
                state->pending.push_back("data: [DONE]\n\n");
                state->done = true;
                continue;
            }
            // kError (appears as an error frame in the event stream so clients can read the reason
            // consistently).
            Json inner = Json::objv();
            inner.set("message", Json::strv(e.err.empty() ? "Generation failed" : e.err));
            inner.set("type", Json::strv("server_error"));
            inner.set("param", Json::nullv());
            Json er = Json::objv();
            er.set("error", std::move(inner));
            state->pending.push_back("data: " + json_dump(er) + "\n\n");
            state->done = true;
            state->final_status = 500;
            continue;
        }
        if (state->pending.empty()) {
            // kDone consumed, all frames sent: normal completion. sink.done() must be called to write
            // the chunked terminator (0\r\n\r\n) — when the provider returns false httplib only
            // closes the connection without writing the terminator, and strict chunked clients
            // (browsers, http.client) report ERR_INCOMPLETE_CHUNKED_ENCODING / IncompleteRead; curl
            // tolerates EOF, which is why earlier integration tests did not catch this. done() writes
            // the terminator internally and marks the stream finished, so the while loop exits
            // normally.
            if (!state->recorded) { record_request(job, state->final_status, false); state->recorded = true; }
            if (state->done) { sink.done(); return true; }
            // Event buffer terminated abnormally (kDone never arrived, pop failure already recorded as
            // 499): the provider ends, closing the connection.
            return false;
        }
        const std::string frame = state->pending.front();
        state->pending.erase(state->pending.begin());
        if (!w(frame)) return false;
        // Only account after the final frame lands: on disconnect/mid-stream failure w already
        // recorded 499, so no double-counting here.
        if (state->pending.empty() && state->done && !state->recorded) {
            record_request(job, state->final_status, false);
            state->recorded = true;
        }
        return true;
    };
    res.set_chunked_content_provider("text/event-stream", provider);
}

// Anthropic-compatible SSE. Event order: message_start → per-block content_block_start/delta/stop →
// message_delta → message_stop; kError ends with an error event. A text block opens at the first
// non-empty text segment (text_delta per segment); a tool block opens with tool_use
// (id/name/empty input) immediately followed by an input_json_delta carrying the full arguments (the
// stream_tools=false shape); the current block is unconditionally closed before a tool_use (blocks
// share one incrementing index). stop_reason "tool_use" only when there is a tool block and a natural
// stop (same as Strata anthropic_events); generated text is split incrementally by ToolCallStreamer,
// malformed chunks are released as text, and unclosed calls are released as their raw text at
// wrap-up (#211). The top level of the event data carries a type field matching the event name
// (Anthropic's official shape, checked item by item by the integration tests).
void Serve::serve_anthropic_sse(const std::shared_ptr<Job>& job, const std::vector<Json>& tools,
                                httplib::Response& res) {
    const std::string id = "msg_" + std::to_string(job->id);
    const std::string model = eng_->model_name();
    struct State {
        ToolCallStreamer st;
        // Tool definitions live by value with State on the heap (same reason as serve_sse: the
        // provider outlives the handler).
        std::vector<Json> tools;
        std::vector<std::string> pending;   // frames pending write (popped from front)
        bool started = false;               // message_start not yet written
        bool block_open = false;            // a block is currently open (last block index = block_idx - 1)
        int block_idx = 0;                  // next block ordinal (shared by text/tool_use)
        bool used_tool = false;
        bool done = false;
        int final_status = 200;
        bool recorded = false;
    };
    auto state = std::make_shared<State>();
    state->tools = tools;
    auto provider = [this, job, id, model, state](std::size_t, httplib::DataSink& sink) -> bool {
        const auto w = [&](const std::string& s) {
            // The vendored DataSink::write is std::function<bool(const char*, size_t)> (two-argument call,
            // see httplib.h:538).
            if (!sink.write(s.data(), s.size())) {
                job->events->cancel();
                if (!state->recorded) { record_request(job, 499, true); state->recorded = true; }
                return false;
            }
            return true;
        };
        const auto enqueue = [&](const std::string& name, Json data) {
            state->pending.push_back("event: " + name + "\ndata: " + json_dump(data) + "\n\n");
        };
        const auto open_block = [&](Json blk) {
            Json cb = Json::objv();
            cb.set("type", Json::strv("content_block_start"));
            cb.set("index", Json::numv(static_cast<double>(state->block_idx)));
            cb.set("content_block", std::move(blk));
            enqueue("content_block_start", std::move(cb));
            state->block_open = true;
        };
        const auto close_block = [&]() {
            if (!state->block_open) return;
            Json cb = Json::objv();
            cb.set("type", Json::strv("content_block_stop"));
            cb.set("index", Json::numv(static_cast<double>(state->block_idx - 1)));
            enqueue("content_block_stop", std::move(cb));
            state->block_open = false;
        };
        const auto text_delta = [&](const std::string& t) {
            Json d = Json::objv();
            d.set("type", Json::strv("content_block_delta"));
            d.set("index", Json::numv(static_cast<double>(state->block_idx - 1)));
            Json delta = Json::objv();
            delta.set("type", Json::strv("text_delta"));
            delta.set("text", Json::strv(t));
            d.set("delta", std::move(delta));
            enqueue("content_block_delta", std::move(d));
        };
        // Streaming split events → block state machine (kToken and kDone wrap-up share the same
        // semantics).
        const auto emit_streamer_event = [&](const ToolCallStreamer::Event& sev) {
            if (sev.kind == ToolCallStreamer::Kind::kText) {
                if (sev.text.empty()) return;
                if (!state->block_open) {
                    Json blk = Json::objv();
                    blk.set("type", Json::strv("text"));
                    blk.set("text", Json::strv(""));
                    open_block(std::move(blk));
                    ++state->block_idx;
                }
                text_delta(sev.text);
                return;
            }
            ToolCall tc;
            std::string perr;
            if (!parse_tool_block(sev.text, state->tools, tc, perr)) {
                // Malformed marker: merged back into text and released, not treated as an error that
                // kills the request.
                if (sev.text.empty()) return;
                if (!state->block_open) {
                    Json blk = Json::objv();
                    blk.set("type", Json::strv("text"));
                    blk.set("text", Json::strv(""));
                    open_block(std::move(blk));
                    ++state->block_idx;
                }
                text_delta(sev.text);
                return;
            }
            close_block();   // tool_use always closes the current block before opening a new one
            Json blk = Json::objv();
            blk.set("type", Json::strv("tool_use"));
            blk.set("id", Json::strv("toolu_" + std::to_string(job->id) + "_" + std::to_string(state->block_idx)));
            blk.set("name", Json::strv(tc.name));
            blk.set("input", Json::objv());
            open_block(std::move(blk));
            ++state->block_idx;
            Json d = Json::objv();
            d.set("type", Json::strv("content_block_delta"));
            d.set("index", Json::numv(static_cast<double>(state->block_idx - 1)));
            Json delta = Json::objv();
            delta.set("type", Json::strv("input_json_delta"));
            delta.set("partial_json", Json::strv(json_dump(tc.arguments)));
            d.set("delta", std::move(delta));
            enqueue("content_block_delta", std::move(d));
            state->used_tool = true;
        };
        // Write one frame; when no frame is writable, first consume job events to accumulate frames.
        for (;;) {
            if (!state->pending.empty() || state->done) break;
            JobEvent e;
            if (!job->events->pop(e)) {
                if (!state->recorded) { record_request(job, 499, job->events->cancelled()); state->recorded = true; }
                return false;
            }
            if (e.kind == EventKind::kToken) {
                if (!state->started) {
                    Json m = Json::objv();
                    m.set("type", Json::strv("message_start"));
                    m.set("id", Json::strv(id));
                    m.set("role", Json::strv("assistant"));
                    m.set("model", Json::strv(model));
                    m.set("content", Json::arrv());
                    m.set("stop_reason", Json::nullv());
                    m.set("stop_sequence", Json::nullv());
                    Json us = Json::objv();
                    us.set("input_tokens", Json::numv(static_cast<double>(job->n_prompt)));
                    us.set("output_tokens", Json::numv(0.0));
                    m.set("usage", std::move(us));
                    enqueue("message_start", std::move(m));
                    state->started = true;
                }
                std::vector<ToolCallStreamer::Event> evs;
                state->st.feed(e.text, evs);
                for (const auto& sev : evs) emit_streamer_event(sev);
                continue;
            }
            if (e.kind == EventKind::kDone) {
                // Wrap-up: first flush the buffered remainder, then close the current block, finally
                // message_delta/message_stop.
                std::vector<ToolCallStreamer::Event> evs;
                state->st.finish(evs);
                for (const auto& sev : evs) emit_streamer_event(sev);
                close_block();
                Json md = Json::objv();
                md.set("type", Json::strv("message_delta"));
                Json dlt = Json::objv();
                dlt.set("stop_reason", Json::strv(
                    state->used_tool && e.finish == "stop" ? "tool_use"
                                                           : (e.finish == "length" ? "max_tokens" : "end_turn")));
                dlt.set("stop_sequence", Json::nullv());
                md.set("delta", std::move(dlt));
                Json us = Json::objv();
                us.set("output_tokens", Json::numv(static_cast<double>(job->n_gen)));
                md.set("usage", std::move(us));
                enqueue("message_delta", std::move(md));
                Json stop = Json::objv();
                stop.set("type", Json::strv("message_stop"));
                enqueue("message_stop", std::move(stop));
                state->done = true;
                continue;
            }
            // kError: Anthropic ends with an error event (same shape as the other event-stream frames).
            Json er = Json::objv();
            Json inner = Json::objv();
            inner.set("type", Json::strv("server_error"));
            inner.set("message", Json::strv(e.err.empty() ? "Generation failed" : e.err));
            er.set("type", Json::strv("error"));
            er.set("error", std::move(inner));
            enqueue("error", std::move(er));
            state->done = true;
            state->final_status = 500;
            continue;
        }
        if (state->pending.empty()) {
            // kDone consumed, all frames sent: normal completion. sink.done() must be called to write
            // the chunked terminator (0\r\n\r\n) — when the provider returns false httplib only
            // closes the connection without writing the terminator, and strict chunked clients
            // (browsers, http.client) report ERR_INCOMPLETE_CHUNKED_ENCODING / IncompleteRead; curl
            // tolerates EOF, which is why earlier integration tests did not catch this. done() writes
            // the terminator internally and marks the stream finished, so the while loop exits
            // normally.
            if (!state->recorded) { record_request(job, state->final_status, false); state->recorded = true; }
            if (state->done) { sink.done(); return true; }
            // Event buffer terminated abnormally (kDone never arrived, pop failure already recorded as
            // 499): the provider ends, closing the connection.
            return false;
        }
        const std::string frame = state->pending.front();
        state->pending.erase(state->pending.begin());
        if (!w(frame)) return false;
        // Only account after the final frame lands: on disconnect/mid-stream failure w already
        // recorded 499, so no double-counting here.
        if (state->pending.empty() && state->done && !state->recorded) {
            record_request(job, state->final_status, false);
            state->recorded = true;
        }
        return true;
    };
    res.set_chunked_content_provider("text/event-stream", provider);
}

void Serve::handle_completions(const httplib::Request& req, httplib::Response& res) {
    Json body;
    std::string jerr;
    if (!json_parse(req.body, body, jerr) || !body.is_obj()) {
        send_json(res, 400, openai_err("Request body is not a valid JSON object: " + jerr, "invalid_request_error", nullptr));
        return;
    }
    std::string prompt;
    bool present = false;
    if (!str_param(body, "prompt", prompt, present) || !present) {
        send_json(res, 400, openai_err("Missing prompt string", "invalid_request_error", "prompt"));
        return;
    }
    // Ruling ⑤: empty prompt rejected.
    if (prompt.empty()) {
        send_json(res, 400, openai_err("prompt must not be an empty string", "invalid_request_error", "prompt"));
        return;
    }
    // Parameter reading same as handle_chat (temperature/top_p/seed/max_tokens/stop; stream same 400).
    GenOptions opts;
    double t = 0.0, p = 0.0, mt = 0.0;
    bool numeric_ok = num_param(body, "temperature", 0.8, t) && num_param(body, "top_p", 0.95, p) &&
                      num_param(body, "max_tokens", 512, mt);
    if (!numeric_ok) { send_json(res, 400, openai_err("temperature/top_p/max_tokens must be numbers", "invalid_request_error", nullptr)); return; }
    opts.temperature = static_cast<float>(t);
    opts.top_p = static_cast<float>(p);
    opts.max_tokens = static_cast<std::uint64_t>(mt);
    if (mt <= 0) { send_json(res, 400, openai_err("max_tokens must be positive", "invalid_request_error", "max_tokens")); return; }
    if (!stop_param(body, opts.stop)) { send_json(res, 400, openai_err("stop must be a string or array of strings", "invalid_request_error", "stop")); return; }
    // Ruling ④: completions likewise reads stream:true from the body → 400.
    const Json* sv = body.find("stream");
    if (sv != nullptr && sv->kind == Json::kBool && sv->b) {
        send_json(res, 400, openai_err("completions does not yet support stream:true (supported in the chat/messages domains)", "invalid_request_error", "stream"));
        return;
    }
    std::shared_ptr<Job> job = launch(req.path, prompt, opts, false);
    if (!job) {
        res.status = 503;
        res.set_content(std::string(kQueueFullBody), "application/json; charset=utf-8");
        // 503 not enqueued: no timing data, not entered into the records ring (the ring only accepts
        // accepted requests).
        return;
    }
    job->raw_body = req.body;   // T14: the raw request body goes into Records
    std::string text, drain_err;
    if (!drain_nonstream(job, text, drain_err)) {
        const int st = job->events->cancelled() ? 499 : 500;
        if (st == 500) send_json(res, 500, openai_err(drain_err, "server_error", nullptr));
        record_request(job, st, job->events->cancelled());   // bookkeeping same as the chat/messages domains: exactly once per exit path
        return;
    }
    Json r = Json::objv();
    r.set("id", Json::strv("cmpl-" + std::to_string(job->id)));
    r.set("object", Json::strv("text_completion"));
    r.set("created", Json::numv(static_cast<double>(std::time(nullptr))));
    r.set("model", Json::strv(job->model));
    Json choice = Json::objv();
    choice.set("text", Json::strv(text));
    choice.set("index", Json::numv(0.0));
    choice.set("logprobs", Json::nullv());
    choice.set("finish_reason", Json::strv(job->finish));
    Json choices = Json::arrv();
    choices.push(std::move(choice));
    r.set("choices", std::move(choices));
    Json usage = Json::objv();
    usage.set("prompt_tokens", Json::numv(static_cast<double>(job->n_prompt)));
    usage.set("completion_tokens", Json::numv(static_cast<double>(job->n_gen)));
    usage.set("total_tokens", Json::numv(static_cast<double>(job->n_prompt + job->n_gen)));
    r.set("usage", std::move(usage));
    send_json(res, 200, r);
    record_request(job, 200, false);
}

void Serve::handle_messages(const httplib::Request& req, httplib::Response& res) {
    Json body;
    std::string jerr;
    if (!json_parse(req.body, body, jerr) || !body.is_obj()) {
        send_json(res, 400, anthropic_err("invalid_request_error", "Request body is not a valid JSON object: " + jerr));
        return;
    }
    double mt = 0.0;
    if (!num_param(body, "max_tokens", 0, mt) || mt <= 0) {
        send_json(res, 400, anthropic_err("invalid_request_error", "max_tokens is required and must be positive"));
        return;
    }
    std::vector<Message> conv;
    if (!messages_to_conversation(body, conv, jerr)) {
        send_json(res, 400, anthropic_err("invalid_request_error", jerr));
        return;
    }
    // Ruling ⑦: empty conversation 400 — same stance as the chat domain's empty-prompt rejection;
    // the two compatibility domains validate consistently.
    if (conv.empty()) {
        send_json(res, 400, anthropic_err("invalid_request_error", "No message content to render"));
        return;
    }
    // tools → internal tool shape (normalize_tools' Anthropic renaming: input_schema → parameters).
    std::vector<Json> tools;
    const Json* tv = body.find("tools");
    if (tv != nullptr) {
        std::string terr;
        if (!normalize_tools(*tv, /*anthropic=*/true, tools, terr)) {
            send_json(res, 400, anthropic_err("invalid_request_error", "tools parse failed: " + terr));
            return;
        }
    }
    GenOptions opts;
    double t = 0.0, p = 0.0;
    bool numeric_ok = num_param(body, "temperature", 0.8, t) && num_param(body, "top_p", 0.95, p);
    if (!numeric_ok) { send_json(res, 400, anthropic_err("invalid_request_error", "temperature/top_p must be numbers")); return; }
    opts.temperature = static_cast<float>(t);
    opts.top_p = static_cast<float>(p);
    opts.max_tokens = static_cast<std::uint64_t>(mt);
    // stop_sequences → stop (array of strings; empty by default).
    const Json* stops = body.find("stop_sequences");
    if (stops != nullptr) {
        if (!stops->is_arr()) { send_json(res, 400, anthropic_err("invalid_request_error", "stop_sequences must be an array")); return; }
        for (const Json& s : stops->arr) {
            if (!s.is_str()) { send_json(res, 400, anthropic_err("invalid_request_error", "stop_sequences elements must be strings")); return; }
            opts.stop.push_back(s.str);
        }
    }
    std::string prompt, rerr;
    Json tj = tools_array(tools);
    if (!render_chat(eng_->chat_template(), conv, /*add_generation_prompt=*/true, prompt, rerr,
                     tools.empty() ? nullptr : &tj)) {
        send_json(res, 500, anthropic_err("server_error", "Structure not supported by the chat template: " + rerr));
        return;
    }
    // Ruling ②: Json has no is_bool — use kind == kBool.
    const Json* s = body.find("stream");
    const bool stream = s != nullptr && s->kind == Json::kBool && s->b;
    std::shared_ptr<Job> job = launch(req.path, prompt, opts, stream);
    if (!job) {
        send_json(res, 503, anthropic_err("overloaded_error", "queue full"));
        // 503 not enqueued: no timing data, not entered into the records ring (the ring only accepts
        // accepted requests).
        return;
    }
    job->raw_body = req.body;   // T14: the raw request body goes into Records (streaming requests too)
    if (stream) { serve_anthropic_sse(job, tools, res); return; }
    std::string text, drain_err;
    if (!drain_nonstream(job, text, drain_err)) {
        // Bookkeeping same as the chat domain: exactly once per exit path (kError records 500 and writes
        // the body; a disconnected client records 499 without writing a body).
        const int st = job->events->cancelled() ? 499 : 500;
        if (st == 500) send_json(res, 500, anthropic_err("server_error", drain_err));
        record_request(job, st, job->events->cancelled());
        return;
    }
    Json r = Json::objv();
    r.set("id", Json::strv("msg_" + std::to_string(job->id)));
    r.set("type", Json::strv("message"));
    r.set("role", Json::strv("assistant"));
    r.set("model", Json::strv(job->model));
    // Split generated text into events: text blocks and tool_use blocks keep generation order
    // (Anthropic's official block shape).
    std::vector<GenEvent> events;
    split_generation(text, tools, events);
    Json content = Json::arrv();
    bool used_tool = false;
    int call_idx = 0;
    for (const GenEvent& ev : events) {
        if (ev.call) {
            used_tool = true;
            content.push(anthropic_tool_use(
                "toolu_" + std::to_string(job->id) + "_" + std::to_string(call_idx),
                ev.call->name, ev.call->arguments));
            ++call_idx;
        } else if (!ev.text.empty()) {
            Json blk = Json::objv();
            blk.set("type", Json::strv("text"));
            blk.set("text", Json::strv(ev.text));
            content.push(std::move(blk));
        }
    }
    r.set("content", std::move(content));
    // stop_reason: has calls and stopped naturally → tool_use (per Strata's anthropic semantics).
    r.set("stop_reason", Json::strv(
        used_tool && job->finish == "stop" ? "tool_use"
                                           : (job->finish == "length" ? "max_tokens" : "end_turn")));
    r.set("stop_sequence", Json::nullv());
    Json usage = Json::objv();
    usage.set("input_tokens", Json::numv(static_cast<double>(job->n_prompt)));
    usage.set("output_tokens", Json::numv(static_cast<double>(job->n_gen)));
    r.set("usage", std::move(usage));
    send_json(res, 200, r);
    record_request(job, 200, false);
}

void Serve::handle_count_tokens(const httplib::Request& req, httplib::Response& res) {
    Json body;
    std::string jerr;
    if (!json_parse(req.body, body, jerr) || !body.is_obj()) {
        send_json(res, 400, anthropic_err("invalid_request_error", "Request body is not a valid JSON object: " + jerr));
        return;
    }
    std::vector<Message> conv;
    if (!messages_to_conversation(body, conv, jerr)) {
        send_json(res, 400, anthropic_err("invalid_request_error", jerr));
        return;
    }
    // Same as handle_messages: empty conversation 400, both paths validate consistently.
    if (conv.empty()) {
        send_json(res, 400, anthropic_err("invalid_request_error", "No message content to render"));
        return;
    }
    // tools are normalized the same way (Anthropic shape); only at render is the same basis used as a
    // real request (count and completion share the prompt).
    std::vector<Json> tools;
    const Json* tv = body.find("tools");
    if (tv != nullptr) {
        std::string terr;
        if (!normalize_tools(*tv, /*anthropic=*/true, tools, terr)) {
            send_json(res, 400, anthropic_err("invalid_request_error", "tools parse failed: " + terr));
            return;
        }
    }
    std::string prompt, rerr;
    Json tj = tools_array(tools);
    if (!render_chat(eng_->chat_template(), conv, /*add_generation_prompt=*/true, prompt, rerr,
                     tools.empty() ? nullptr : &tj)) {
        send_json(res, 500, anthropic_err("server_error", "Structure not supported by the chat template: " + rerr));
        return;
    }
    std::vector<std::int32_t> ids;
    if (!prompt_cache_.lookup(prompt, ids)) {
        // Same engine_mu_ as worker_main: encode writes the worker-shared piece_cache_, and out of
        // the lock it would be a data race with concurrent requests (see the header comment in
        // server.hpp); chat_template() is read-only and needs no lock. The lock only covers mutable
        // engine access: the 500 response on the failure path is moved outside the lock, so writing
        // a response for a slow-reading client cannot hold the lock and freeze every slot. A hit
        // never enters the lock; a miss still encodes and backfills inside the lock (lock order is
        // always engine → cache, no reverse path).
        std::lock_guard<std::mutex> lock(engine_mu_);
        if (!eng_->tokenizer().encode(prompt, /*parse_special=*/true, ids, rerr)) {
            send_json(res, 500, anthropic_err("server_error", "Tokenization failed: " + rerr));
            return;
        }
        prompt_cache_.store(prompt, ids);
    }
    Json r = Json::objv();
    r.set("input_tokens", Json::numv(static_cast<double>(ids.size())));
    send_json(res, 200, r);
}

void Serve::record_request(const std::shared_ptr<Job>& job, int http_status, bool disconnected) {
    (void) http_status;
    if (!job) return;
    // Requests that never reached the completed state (done_at unset) are not accounted: the only
    // remaining case is on the client-disconnect (499) path, where there is no happens-before edge
    // between cancel() and the worker's write of done_at — the HTTP thread can read the default
    // value before the worker sets done_at. An unfinished request is not the last completed request,
    // so it is skipped entirely, avoiding a default time_point becoming a bogus reading of about
    // -1.7e9 s that would pollute last_timings in /v1/status.
    if (job->done_at == std::chrono::steady_clock::time_point{}) return;
    const auto enq = job->enqueued_at;   // a default-constructed time_point (if never enqueued) → all zero
    const double wall = std::chrono::duration<double>(job->done_at - enq).count();
    LastTimings lt;
    lt.have = true;
    lt.queue_s = std::chrono::duration<double>(job->leased_at - enq).count();
    lt.first_token_s = job->first_at != std::chrono::steady_clock::time_point{}
                           ? std::chrono::duration<double>(job->first_at - enq).count() : 0.0;
    if (lt.first_token_s > 0) lt.first_token_s = lt.first_token_s - lt.queue_s;
    lt.wallclock_s = wall;
    lt.decode_tok_s = job->tok_s;
    lt.prompt_tokens = job->n_prompt;
    lt.output_tokens = job->n_gen;
    lt.finish = job->finish;
    {
        std::lock_guard<std::mutex> lk(last_mu_);
        last_ = lt;
        totals_.prompt_tokens += job->n_prompt;
        totals_.completion_tokens += job->n_gen;
        totals_.requests += 1;   // completion count (per Strata's totals.requests: +1 at each completion time)
        // Timing basis: prefill ≈ first_at - leased_at, decode ≈ done_at - first_at.
        const auto ms = [](std::chrono::steady_clock::duration d) {
            return std::chrono::duration<double, std::milli>(d).count();
        };
        if (job->first_at != std::chrono::steady_clock::time_point{})
            totals_.prompt_ms += ms(job->first_at - job->leased_at);
        const auto start = job->first_at != std::chrono::steady_clock::time_point{}
                               ? job->first_at : job->leased_at;
        if (job->done_at > start) totals_.decode_ms += ms(job->done_at - start);
        // T13: full-run accumulation of tok_s_mean (generated tokens ÷ decode time; same lock as
        // telemetry_extra, and there is a happens-before edge between the sampling thread's previous
        // reading and this accumulation). On encode failure n_gen is always 0, filtered out by the
        // n_gen > 0 check; the disconnect race (done_at not yet set) is caught by the done_at guard
        // above.
        if (job->n_gen > 0 && job->done_at > job->leased_at &&
            job->done_at != std::chrono::steady_clock::time_point{}) {
            const auto start = job->first_at != std::chrono::steady_clock::time_point{}
                                   ? job->first_at : job->leased_at;
            if (job->done_at > start) {
                gen_done_ += job->n_gen;
                gen_us_ += static_cast<std::uint64_t>(
                    std::chrono::duration_cast<std::chrono::microseconds>(job->done_at - start).count());
            }
        }
    }
    // ---- Latency histogram (the ttft/itl/e2e families on the Prometheus text endpoint) ----
    // Like totals, counted once at completion time; the span is uniformly enq (enqueue) → first
    // token → done_at, sharing e2e's start point. Failed requests without a first token pass ttft as
    // 0, and the Latencies-side >0 guard skips them so buckets stay clean.
    const auto start_of_decode = job->first_at != std::chrono::steady_clock::time_point{}
                                     ? job->first_at : job->leased_at;
    lat_.observe(job->first_at != std::chrono::steady_clock::time_point{}
                     ? std::chrono::duration<double>(job->first_at - enq).count() : 0.0,
                 job->n_gen,
                 job->done_at > start_of_decode
                     ? std::chrono::duration<double>(job->done_at - start_of_decode).count() : 0.0,
                 wall);
    // ---- T14: Records ring (one entry per job; disconnects/errors all finish in the error state) ----
    if (job->record_id.empty()) {
        RequestBegin b;
        b.path = job->path;
        b.model = job->model;
        b.input = job->raw_body;
        b.stream = job->stream;
        Json raw;   // response_format read from the raw request body (a parse failure is harmless; it lands as an empty string)
        std::string jerr;
        if (json_parse(job->raw_body, raw, jerr) && raw.is_obj()) {
            const Json* rf = raw.find("response_format");
            if (rf != nullptr && rf->is_obj()) {
                const Json* ty = rf->find("type");
                if (ty != nullptr && ty->is_str()) b.response_format = ty->str;
            }
        }
        b.created = static_cast<double>(std::time(nullptr)) - wall;   // ≈ request start
        job->record_id = records_.begin(b);
    }
    if (!job->record_id.empty()) {
        RequestFinish f;
        f.text = job->text;
        f.error = disconnected ? "disconnected" : job->error;
        f.queue_s = lt.queue_s;
        f.first_token_s = lt.first_token_s;
        f.wallclock_s = wall;
        f.decode_tok_s = job->tok_s;
        f.prompt_tokens = job->n_prompt;
        f.output_tokens = job->n_gen;
        records_.finish(job->record_id, f);
    }
}

TelemetryExtra Serve::telemetry_extra() const {
    std::lock_guard<std::mutex> lk(last_mu_);
    TelemetryExtra ex;
    if (last_.have && last_.decode_tok_s > 0.0) ex.tok_s = last_.decode_tok_s;
    if (last_.have && last_.first_token_s > 0.001)
        ex.prefill_tok_s_mean = static_cast<double>(last_.prompt_tokens) / last_.first_token_s;
    if (gen_us_ > 0) ex.tok_s_mean = 1e6 * static_cast<double>(gen_done_) /
                                     static_cast<double>(gen_us_);
    return ex;
}

void Serve::register_routes() {
    httplib::Server& svr = *svr_;
    // Auth (spec §3): /web, /api-monitor, / and /health are reachable, and the pages carry the key
    // themselves; /v1/*, /api/* and /metrics with a missing or wrong key → 401 (the error shape of
    // the corresponding domain).
    svr.set_pre_routing_handler([this](const httplib::Request& req, httplib::Response& res) {
        if (cfg_.api_key.empty()) return httplib::Server::HandlerResponse::Unhandled;
        const std::string want = "Bearer " + cfg_.api_key;
        if (req.get_header_value("Authorization") == want) return httplib::Server::HandlerResponse::Unhandled;
        if (req.path == "/" || req.path.rfind("/web/", 0) == 0 || req.path == "/api-monitor" ||
            req.path == "/health" || req.path.rfind("/fonts/", 0) == 0) {
            return httplib::Server::HandlerResponse::Unhandled;
        }
        if (anthropic_path(req.path)) {
            send_json(res, 401, anthropic_err("authentication_error", "Missing or wrong API key"));
        } else {
            send_json(res, 401, openai_err("Missing or wrong API key", "authentication_error", nullptr));
        }
        return httplib::Server::HandlerResponse::Handled;
    });
    // Paths without a registered route always get a 404 JSON (spec §3's static domain: /v1/responses,
    // /load etc. all go through here).
    svr.set_error_handler([](const httplib::Request& req, httplib::Response& res) {
        (void) req;
        if (res.status == 404) send_json(res, 404, openai_err("not found", "not_found", nullptr));
    });

    svr.Get("/health", [this](const httplib::Request&, httplib::Response& res) {
        Json h = Json::objv();
        h.set("status", Json::strv("ok"));
        h.set("model", Json::strv(eng_->model_name()));
        h.set("max_context", Json::numv(static_cast<double>(eng_->config().max_context)));
        h.set("images", Json::boolv(false));
        h.set("api_key", Json::boolv(!cfg_.api_key.empty()));
        h.set("loaded", Json::boolv(true));
        h.set("service", Json::strv("qinfer"));
        send_json(res, 200, h);
    });
    svr.Get("/v1/models", [this](const httplib::Request&, httplib::Response& res) {
        Json model = Json::objv();
        model.set("id", Json::strv(eng_->model_name()));
        model.set("object", Json::strv("model"));
        Json st = Json::objv();
        st.set("value", Json::strv("loaded"));
        model.set("status", std::move(st));
        Json meta = Json::objv();
        meta.set("n_ctx", Json::numv(static_cast<double>(eng_->config().max_context)));
        model.set("meta", std::move(meta));
        Json arch = Json::objv();
        Json in = Json::arrv(); in.push(Json::strv("text"));
        Json out = Json::arrv(); out.push(Json::strv("text"));
        arch.set("input_modalities", std::move(in));
        arch.set("output_modalities", std::move(out));
        model.set("architecture", std::move(arch));
        Json data = Json::arrv();
        data.push(std::move(model));
        Json r = Json::objv();
        r.set("object", Json::strv("list"));
        r.set("data", std::move(data));
        send_json(res, 200, r);
    });
    // T10: stream detection moved to the routing layer (invalid/non-object request body → treated as
    // non-streaming); handle_chat's stream is the final truth.
    svr.Post("/v1/chat/completions", [this](const httplib::Request& req, httplib::Response& res) {
        Json body;
        std::string jerr;
        bool stream = false;
        if (json_parse(req.body, body, jerr) && body.is_obj()) {
            const Json* s = body.find("stream");
            if (s != nullptr && s->kind == Json::kBool) stream = s->b;
        }
        handle_chat(req, res, stream);
    });
    svr.Post("/v1/completions",
             [this](const httplib::Request& req, httplib::Response& res) { handle_completions(req, res); });
    // Anthropic domain: /v1/messages (including streaming) + count_tokens. The 401 domain split
    // already happens in pre-routing (anthropic_path).
    svr.Post("/v1/messages", [this](const httplib::Request& req, httplib::Response& res) { handle_messages(req, res); });
    svr.Post("/v1/messages/count_tokens", [this](const httplib::Request& req, httplib::Response& res) { handle_count_tokens(req, res); });
    // Status and property static domain (spec §3; /metrics and /api/requests are reserved for T14):
    // /v1/status self-describes externally, /status is the stripped-down probe, /props capability
    // discovery, /slots the context ring.
    svr.Get("/v1/status", [this](const httplib::Request&, httplib::Response& res) {
        const auto now = std::time(nullptr);
        const std::uint64_t ctx = eng_->config().max_context;
        const bool busy = sched_.any_busy();
        Json r = Json::objv();
        r.set("service", Json::strv("qinfer"));
        r.set("model", Json::strv(eng_->model_name()));
        r.set("loaded", Json::boolv(true));
        r.set("auto_load", Json::boolv(false));
        Json so = Json::objv();
        Json fmts = Json::arrv(); fmts.push(Json::strv("json_object"));
        so.set("formats", std::move(fmts));
        so.set("method", Json::strv("prompt_and_validate"));
        so.set("constrained_decoding", Json::boolv(false));
        so.set("stream_buffered", Json::boolv(true));
        r.set("structured_output", std::move(so));
        r.set("engine", Json::strv("qinfer-m1"));
        r.set("started", Json::numv(static_cast<double>(started_)));
        r.set("uptime_s", Json::numv(static_cast<double>(now - started_)));
        r.set("cache_max_tokens", Json::numv(static_cast<double>(ctx)));
        Json cx = Json::objv();
        cx.set("native", Json::numv(static_cast<double>(ctx)));
        cx.set("max_positions", Json::numv(static_cast<double>(ctx)));
        r.set("context", std::move(cx));
        Json cc = Json::objv();
        cc.set("serving", Json::numv(static_cast<double>(cfg_.slots)));
        cc.set("requested", Json::numv(static_cast<double>(cfg_.slots)));
        r.set("concurrency", std::move(cc));
        Json dl = Json::arrv();
        dl.push(Json::strv("/v1/chat/completions"));
        dl.push(Json::strv("/v1/messages"));
        r.set("dialects", std::move(dl));
        Json vis = Json::objv();
        vis.set("enabled", Json::boolv(false));
        vis.set("available", Json::boolv(false));
        vis.set("error", Json::nullv());
        r.set("vision", std::move(vis));
        Json act = Json::objv();
        act.set("requests", Json::numv(static_cast<double>(req_id_.load())));
        act.set("in_flight", Json::numv((sched_.queued() + (busy ? 1 : 0))));
        act.set("last_request_at", Json::nullv());
        r.set("activity", std::move(act));
        {
            std::lock_guard<std::mutex> lk(last_mu_);
            if (last_.have) {
                Json lt = Json::objv();
                lt.set("queue_s", Json::numv(last_.queue_s));
                lt.set("first_token_s", Json::numv(last_.first_token_s));
                lt.set("wallclock_s", Json::numv(last_.wallclock_s));
                lt.set("decode_tok_s", Json::numv(last_.decode_tok_s));
                lt.set("prompt_tokens", Json::numv(static_cast<double>(last_.prompt_tokens)));
                lt.set("output_tokens", Json::numv(static_cast<double>(last_.output_tokens)));
                r.set("last_timings", std::move(lt));
            } else {
                r.set("last_timings", Json::nullv());
            }
        }
        r.set("vram", Json::objv());
        Json mach = Json::objv();
        mach.set("at", Json::numv(static_cast<double>(now)));
        if (tel_) {
            const auto ts = tel_->snapshot();
            const auto& hw = ts.now;
            const auto& st = ts.stat;
            auto get = [&hw](const char* k) -> std::optional<double> {
                return hw.count(k) ? std::optional<double>(hw.at(k)) : std::nullopt;
            };
            if (!st.gpu_name.empty()) {   // Strata: machine.gpu is entirely null when there is no GPU
                Json g = Json::objv();
                g.set("name", Json::strv(st.gpu_name));
                g.set("used_mib", scaled_mib_gib(get("gpu_mem_used"), 1 << 20, 0));
                g.set("total_mib", scaled_mib_gib(get("gpu_mem_total"), 1 << 20, 0));
                g.set("util_pct", get("gpu_util").has_value() ? Json::numv(*get("gpu_util"))
                                                              : Json::nullv());
                g.set("temp_c", get("gpu_temp").has_value() ? Json::numv(*get("gpu_temp"))
                                                            : Json::nullv());
                g.set("power_w", get("gpu_power").has_value() ? Json::numv(*get("gpu_power"))
                                                              : Json::nullv());
                mach.set("gpu", std::move(g));
            } else {
                mach.set("gpu", Json::nullv());
            }
            if (get("ram_total").has_value()) {
                Json rm = Json::objv();
                rm.set("used_gib", scaled_mib_gib(get("ram_used"), 1 << 30, 1));
                rm.set("total_gib", scaled_mib_gib(get("ram_total"), 1 << 30, 1));
                mach.set("ram", std::move(rm));
            } else {
                mach.set("ram", Json::nullv());
            }
        } else {
            mach.set("gpu", Json::nullv());
            mach.set("ram", Json::nullv());
        }
        r.set("machine", std::move(mach));
        send_json(res, 200, r);
    });
    svr.Get("/status", [this](const httplib::Request&, httplib::Response& res) {
        const bool busy = sched_.any_busy();
        Json r = Json::objv();
        r.set("busy", Json::boolv(busy));
        r.set("queued", Json::numv(static_cast<double>(sched_.queued())));
        r.set("phase", busy ? Json::strv("generating") : Json::nullv());
        r.set("slots", Json::numv(static_cast<double>(sched_.slots())));
        send_json(res, 200, r);
    });
    svr.Get("/props", [this](const httplib::Request&, httplib::Response& res) {
        Json dg = Json::objv();
        dg.set("n_ctx", Json::numv(static_cast<double>(eng_->config().max_context)));
        Json prm = Json::objv();
        prm.set("temperature", Json::numv(0.8));
        prm.set("top_p", Json::numv(0.95));
        prm.set("seed", Json::numv(-1.0));
        prm.set("n_predict", Json::numv(512.0));
        dg.set("params", std::move(prm));
        Json r = Json::objv();
        r.set("default_generation_settings", std::move(dg));
        r.set("total_slots", Json::numv(static_cast<double>(cfg_.slots)));
        r.set("compute_threads", Json::numv(static_cast<double>(cfg_.compute_threads)));
        // GPU device side: offload_enabled = --gpu was given in the config; gpu_devices = auto or an
        // explicit list.
        r.set("offload_enabled", Json::boolv(cfg_.gpu_auto || !cfg_.gpu_devices.empty()));
        {
            Json gd = Json::arrv();
            if (cfg_.gpu_auto) {
                gd.push(Json::strv("auto"));
            } else {
                for (int d : cfg_.gpu_devices) gd.push(Json::numv(static_cast<double>(d)));
            }
            r.set("gpu_devices", std::move(gd));
        }
        // Device KV residency cap (tokens); 0 = all-resident (--kv-resident).
        r.set("kv_resident", Json::numv(static_cast<double>(cfg_.kv_resident)));
        r.set("model_alias", Json::strv(eng_->model_name()));
        r.set("chat_template", Json::strv(eng_->chat_template()));
        Json caps = Json::objv();
        caps.set("templater", Json::strv("lexer-subset"));
        caps.set("rich_text", Json::boolv(false));
        caps.set("date", Json::boolv(false));
        r.set("chat_template_caps", std::move(caps));
        Json mod = Json::objv();
        mod.set("vision", Json::boolv(false));
        r.set("modalities", std::move(mod));
        r.set("models_autoload", Json::boolv(false));
        r.set("is_sleeping", Json::boolv(false));
        r.set("build_info", Json::strv("Q-Infer M1"));
        send_json(res, 200, r);
    });
    svr.Get("/slots", [this](const httplib::Request&, httplib::Response& res) {
        const auto views = sched_.slots_view();
        Json list = Json::arrv();
        for (const SlotView& v : views) {
            Json s = Json::objv();
            s.set("id", Json::numv(static_cast<double>(v.index)));
            s.set("n_ctx", Json::numv(static_cast<double>(eng_->config().max_context)));
            s.set("is_processing", Json::boolv(v.busy));
            s.set("n_prompt_tokens", Json::numv(0.0));
            list.push(std::move(s));
        }
        send_json(res, 200, list);
    });
    svr.Get("/metrics", [this](const httplib::Request& req, httplib::Response& res) {
        const auto now = std::time(nullptr);
        const std::uint64_t ctx = eng_->config().max_context;
        const bool busy = sched_.any_busy();
        Json r = Json::objv();
        Json eng = Json::objv();
        eng.set("model", Json::strv(eng_->model_name()));
        eng.set("max_context", Json::numv(static_cast<double>(ctx)));
        eng.set("images", Json::boolv(false));
        r.set("engine", std::move(eng));
        Json live = Json::objv();
        live.set("state", Json::strv(busy ? "generating" : "idle"));
        live.set("queued", Json::numv(static_cast<double>(sched_.queued())));
        live.set("phase", Json::nullv());          // v0 has no phase tracking
        live.set("prompt_tokens", Json::nullv());  // v0 has no per-item view of in-flight requests
        live.set("prompt_read", Json::nullv());
        live.set("prompt_total", Json::nullv());
        live.set("generated", Json::nullv());
        live.set("max_tokens", Json::nullv());
        live.set("elapsed_s", Json::nullv());
        {
            std::lock_guard<std::mutex> lk(last_mu_);
            if (last_.have) {
                live.set("tok_s", Json::numv(round1(last_.decode_tok_s)));
            } else {
                live.set("tok_s", Json::nullv());
            }
            if (last_.have && last_.first_token_s > 0.001) {
                live.set("prefill_tok_s_mean", Json::numv(round1(
                    static_cast<double>(last_.prompt_tokens) / last_.first_token_s)));
            } else {
                live.set("prefill_tok_s_mean", Json::nullv());
            }
            if (gen_us_ > 0) {
                live.set("tok_s_mean", Json::numv(round1(
                    1e6 * static_cast<double>(gen_done_) / static_cast<double>(gen_us_))));
            } else {
                live.set("tok_s_mean", Json::nullv());
            }
        }
        live.set("tok_s_window_s", Json::nullv());   // v0 has no sliding window (M2 to add)
        const auto views = sched_.slots_view();
        live.set("parallel", Json::numv(static_cast<double>(cfg_.slots)));
        int running = 0;
        Json slots = Json::arrv();
        for (const SlotView& v : views) {
            if (v.busy) ++running;
            Json s = Json::objv();
            s.set("id", Json::numv(static_cast<double>(v.index)));
            s.set("n_ctx", Json::numv(static_cast<double>(ctx)));
            s.set("is_processing", Json::boolv(v.busy));
            s.set("n_prompt_tokens", Json::numv(0.0));
            slots.push(std::move(s));
        }
        live.set("running", Json::numv(static_cast<double>(running)));
        live.set("slots", std::move(slots));
        live.set("waiting", Json::numv(static_cast<double>(sched_.queued())));
        r.set("live", std::move(live));
        const auto list = records_.list();   // newest first
        Json reqs = Json::arrv();
        for (std::size_t i = 0; i < list.size() && i < 12; ++i) reqs.push(record_summary_json(list[i]));
        r.set("requests", std::move(reqs));
        Json cc = Json::objv();
        {
            // Cache live view (spec §6's conversation_cache filled in): per-request counters, all real values.
            const PromptCache::Stats cs = prompt_cache_.stats();
            cc.set("enabled", Json::boolv(prompt_cache_.enabled()));
            cc.set("capacity", Json::numv(static_cast<double>(cfg_.conversation_cache)));
            cc.set("hits", Json::numv(static_cast<double>(cs.hits)));
            cc.set("misses", Json::numv(static_cast<double>(cs.misses)));
            cc.set("entries", Json::numv(static_cast<double>(cs.entries)));
            cc.set("tokens", Json::numv(static_cast<double>(cs.tokens)));
        }
        r.set("conversation_cache", std::move(cc));
        r.set("requests_kept", Json::numv(static_cast<double>(Records::kCap)));
        Json total = Json::objv();
        {
            std::lock_guard<std::mutex> lk(last_mu_);
            total.set("requests", Json::numv(static_cast<double>(totals_.requests)));
            total.set("prompt_tokens", Json::numv(static_cast<double>(totals_.prompt_tokens)));
            total.set("reused", Json::numv(static_cast<double>(reused_.load())));
            total.set("completion_tokens", Json::numv(static_cast<double>(totals_.completion_tokens)));
            total.set("prompt_ms", Json::numv(totals_.prompt_ms));
            total.set("decode_ms", Json::numv(totals_.decode_ms));
        }
        r.set("totals", std::move(total));
        const auto ts = tel_ ? tel_->snapshot() : TelemetrySnapshot{};
        Json hw = Json::objv();
        for (const auto& [k, v] : ts.now) hw.set(k, v ? Json::numv(*v) : Json::nullv());
        if (ts.gpus.size() > 1) {   // Strata: gpus[] only when there are multiple GPUs
            Json gpu_list = Json::arrv();
            for (const GpuReading& gr : ts.gpus) {
                Json g = Json::objv();
                g.set("index", Json::numv(static_cast<double>(gr.index)));
                g.set("name", gr.name.empty() ? Json::nullv() : Json::strv(gr.name));
                g.set("util", gr.util ? Json::numv(*gr.util) : Json::nullv());
                g.set("mem_used", gr.mem_used ? Json::numv(*gr.mem_used) : Json::nullv());
                g.set("mem_total", gr.mem_total ? Json::numv(*gr.mem_total) : Json::nullv());
                g.set("temp", gr.temp ? Json::numv(*gr.temp) : Json::nullv());
                g.set("power", gr.power ? Json::numv(*gr.power) : Json::nullv());
                gpu_list.push(std::move(g));
            }
            hw.set("gpus", std::move(gpu_list));
        }
        r.set("hardware", std::move(hw));
        Json hs = Json::objv();
        hs.set("gpu_name", Json::strv(ts.stat.gpu_name));
        hs.set("gpu_count", Json::numv(static_cast<double>(ts.stat.gpu_count)));
        hs.set("gpu_note", Json::nullv());   // always empty in v0 (see the comment in telemetry.hpp)
        hs.set("cpu_name", Json::strv(ts.stat.cpu_name));
        hs.set("cores", Json::numv(static_cast<double>(ts.stat.cores)));
        hs.set("threads", Json::numv(static_cast<double>(ts.stat.threads)));
        // Stage C: the integrated GPU (Vulkan compute path for unhit experts).
        hs.set("igpu_name", Json::strv(ts.stat.igpu_name));
        hs.set("igpu_available", Json::boolv(ts.stat.igpu_available));
        hs.set("igpu_note", Json::strv(ts.stat.igpu_note));
        r.set("hardware_static", std::move(hs));
        Json hist = Json::objv();
        for (const auto& [k, vec] : ts.history) hist.set(k, series_json(vec));
        r.set("history", std::move(hist));
        r.set("time", Json::numv(static_cast<double>(now)));
        // Content negotiation (Strata contract, wants_prometheus in serve/prometheus.py): Accept with
        // text/plain or openmetrics, or ?format=prometheus → Prometheus text; the frontend fetch and
        // headless curl (default Accept */*) stay on JSON. text and JSON share one source — rendering
        // only reads the just-built r, a single source of truth. The predicate follows Strata's
        // "query contains format=prometheus" semantics, so the full k=v fragment is passed back
        // rather than a bare value.
        const std::string fmt = "format=" + req.get_param_value("format");
        if (wants_prometheus(req.get_header_value("Accept"), fmt)) {
            res.set_content(render_prometheus(r, lat_.snapshot(), eng_->model_name()),
                            "text/plain; version=0.0.4; charset=utf-8");
            return;
        }
        send_json(res, 200, r);
    });
    svr.Get("/api/requests", [this](const httplib::Request& req, httplib::Response& res) {
        const std::string id = req.get_param_value("id");
        if (!id.empty()) {
            RequestRecord rec;
            if (!records_.detail(id, rec)) {
                send_json(res, 404, openai_err("request no longer retained", "invalid_request_error", nullptr));
                return;
            }
            send_json(res, 200, record_detail_json(rec));
            return;
        }
        const auto list = records_.list();
        Json arr = Json::arrv();
        for (const RequestRecord& r : list) arr.push(record_summary_json(r));
        Json out = Json::objv();
        out.set("requests", std::move(arr));
        out.set("retention", Json::numv(static_cast<double>(Records::kCap)));
        out.set("persistent", Json::boolv(false));
        out.set("loaded", Json::boolv(true));   // loaded at engine startup (no hot loading in v0)
        out.set("auto_load", Json::boolv(false));
        send_json(res, 200, out);
    });
    // Static pages (the files landed by T15/T16; 404 is fine at this point — T9 only verifies the API
    // surface).
    svr.Get("/", [this](const httplib::Request&, httplib::Response& res) {
        res.set_header("Cache-Control", "no-cache");
        res.set_content(web_index_, "text/html; charset=utf-8");
    });
    svr.Get("/api-monitor", [this](const httplib::Request&, httplib::Response& res) {
        res.set_header("Cache-Control", "no-cache");
        res.set_content(web_monitor_, "text/html; charset=utf-8");
    });
    // /web/* is served here rather than through set_base_dir so every asset
    // carries Cache-Control: no-cache: the vendored static handler left the
    // browsers free to heuristically cache the CSS/JS, which hid frontend
    // edits behind a stale reload. This is a local tool; revalidation costs
    // nothing.
    const std::filesystem::path base = std::filesystem::path(__FILE__).parent_path() / "web";
    svr.Get(R"(/web/(.+))", [base](const httplib::Request& req, httplib::Response& res) {
        std::string rel = req.matches[1].str();
        if (rel.find("..") != std::string::npos) {
            res.status = 400;
            return;
        }
        std::ifstream f(base / rel, std::ios::binary);
        if (!f) {
            res.status = 404;
            return;
        }
        std::string body((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        const std::size_t dot = rel.rfind('.');
        const std::string ext = dot == std::string::npos ? "" : rel.substr(dot);
        const char* ct = ext == ".css"    ? "text/css; charset=utf-8"
                         : ext == ".js"   ? "text/javascript; charset=utf-8"
                         : ext == ".html" ? "text/html; charset=utf-8"
                         : ext == ".svg"  ? "image/svg+xml"
                         : ext == ".woff2" ? "font/woff2"
                         : ext == ".json" ? "application/json; charset=utf-8"
                                          : "application/octet-stream";
        res.set_header("Cache-Control", "no-cache");
        res.set_content(body, ct);
    });
}

bool Serve::listen(std::string& err) {
    register_routes();
    // HTTP thread pool: 2 × slots + 2 (spec §10: pool size configured by the entry point from the
    // slot count). This vendored httplib hooks the pool through the new_task_queue member (no
    // set_thread_pool_size), wired per the vendored API.
    const std::size_t tp = cfg_.threads > 0 ? static_cast<std::size_t>(cfg_.threads) : 2 * cfg_.slots + 2;
    svr_->new_task_queue = [tp] { return new httplib::ThreadPool(tp); };
    for (std::size_t i = 0; i < cfg_.slots; ++i) workers_.emplace_back([this] { worker_main(); });
    if (!svr_->bind_to_port(cfg_.host, cfg_.port)) {
        sched_.stop();
        err = "Failed to bind " + cfg_.host + ":" + std::to_string(cfg_.port);
        return false;
    }
    (void) svr_->listen_after_bind();
    return true;
}

void Serve::stop() {
    if (svr_) svr_->stop();
    sched_.stop();
    for (auto& w : workers_) {
        if (w.joinable()) w.join();
    }
    workers_.clear();
}

}  // namespace qinfer::serve