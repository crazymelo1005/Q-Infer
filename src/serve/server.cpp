#include "serve/server.hpp"

#include <chrono>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <sstream>

#include "httplib.h"
#include "serve/json.hpp"

namespace qinfer::serve {

namespace {

// ---- 请求体参数读取（全部按规格 §5 的非法参数 → 400 口径）----
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

// stop：缺省 → 空；字符串 → 一个；数组 → 全字符串；其他 → 假（400）。
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

// ---- 错误体（规格 §5 的两个域形状）----
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

bool anthropic_path(const std::string& p) {
    return p == "/v1/messages" || p == "/v1/messages/count_tokens";
}

std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    std::ostringstream s;
    s << f.rdbuf();
    return s.str();
}

// 把停等协议滚一遍（worker 侧语义见 worker_main）：读 kToken 攒文本，kDone/kError 收尾。
// HTTP 线程专用（阻塞 pop）。返回 false = 出错（err 给出消息）。
bool drain_nonstream(const std::shared_ptr<Job>& job, std::string& text, std::string& err) {
    JobEvent e;
    for (;;) {
        if (!job->events->pop(e)) { err = "事件缓冲异常终止"; return false; }
        if (e.kind == EventKind::kToken) { text += e.text; continue; }
        if (e.kind == EventKind::kDone) return true;
        // kError
        err = e.err;
        return false;
    }
}

// 消息列表组装（Anthropic 域）：system + messages（只收 user/assistant；content 收字符串或文本块
// 数组，出现图像块等一律拒绝）→ 与 OpenAI 域同一套 Message 列表交给 render_chat。产出与 OpenAI 域
// 同源的 prompt——两个兼容域对同一请求出同一 token 序列（M1.3 硬指标）就靠这条单一渲染路径。
bool messages_to_conversation(const Json& body, std::vector<Message>& conv, std::string& err) {
    const Json* sys = body.find("system");
    if (sys != nullptr) {
        if (!sys->is_str()) { err = "system 必须是字符串"; return false; }
        conv.emplace_back(Message{"system", sys->str});
    }
    const Json* msgs = body.find("messages");
    if (msgs == nullptr || !msgs->is_arr()) { err = "缺少 messages 数组"; return false; }
    for (const Json& m : msgs->arr) {
        if (!m.is_obj()) { err = "messages 元素不是对象"; return false; }
        const Json* role = m.find("role");
        const Json* content = m.find("content");
        if (role == nullptr || !role->is_str() ||
            (role->str != "user" && role->str != "assistant")) {
            err = "role 只收 user/assistant";
            return false;
        }
        std::string text;
        if (content == nullptr) { err = "缺 content"; return false; }
        if (content->is_str()) {
            text = content->str;
        } else if (content->is_arr()) {
            for (const Json& blk : content->arr) {
                if (!blk.is_obj() || !blk.find("type") || !blk.find("type")->is_str() ||
                    blk.find("type")->str != "text") {
                    err = "content 块只收 {type: text}（图像等多模态 M2）";
                    return false;
                }
                const Json* t = blk.find("text");
                if (t == nullptr || !t->is_str()) { err = "文本块缺 text"; return false; }
                text += t->str;
            }
        } else {
            err = "content 必须是字符串或块数组";
            return false;
        }
        if (text.empty()) continue;
        conv.emplace_back(Message{role->str, text});
    }
    return true;
}

}  // namespace

Serve::Serve(ServeConfig cfg, std::shared_ptr<Engine> eng)
    : cfg_(std::move(cfg)),
      eng_(std::move(eng)),
      sched_(cfg_.slots, cfg_.queue_capacity == 0 ? 2 * cfg_.slots : cfg_.queue_capacity),
      svr_(std::make_unique<httplib::Server>()),
      started_(std::time(nullptr)) {
    const std::filesystem::path base = std::filesystem::path(__FILE__).parent_path() / "web";
    web_index_ = read_file((base / "index.html").string());
    web_monitor_ = read_file((base / "monitor.html").string());
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
            // v0 串行化：encode 与 generate 都在锁内。encode 写各 worker 共享的 piece_cache_
            // （tokenizer.cpp 的 encode_plain），脱离锁是两个 worker 并发改同一张表（见 server.hpp 头注释）。
            std::lock_guard<std::mutex> lock(engine_mu_);
            if (!eng_->tokenizer().encode(job->prompt, /*parse_special=*/true, ids, err)) {
                job->finish = "error";
                job->error = "分词失败：" + err;
                job->events->stop_with_error(job->error);
                sched_.done(job);
                continue;
            }
            job->prompt_ids = std::move(ids);
            job->n_prompt = job->prompt_ids.size();
            ok = eng_->generate(job->prompt_ids, job->opts,
                                [&](const std::string& piece) -> bool {
                                    if (job->first_at == std::chrono::steady_clock::time_point{})
                                        job->first_at = std::chrono::steady_clock::now();
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
        // push 全失败（客户端已断开）时也照常释放槽位；记录由 HTTP 侧标 disconnected。
        sched_.done(job);
    }
}

std::shared_ptr<Job> Serve::launch(const std::string& path, std::string prompt,
                                   const GenOptions& opts, bool stream) {
    // reject_after：第 N 个请求起一律拒绝入队（503）。req_id_ 的读在 job->id = req_id_++ 前，
    // 保证「第 N 个请求」按到达序数。
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
        send_json(res, 400, openai_err("请求体不是合法 JSON 对象：" + jerr, "invalid_request_error", nullptr));
        return;
    }
    const Json* msgs = body.find("messages");
    if (msgs == nullptr || !msgs->is_arr()) {
        send_json(res, 400, openai_err("缺少 messages 数组", "invalid_request_error", "messages"));
        return;
    }
    // messages → [{role, content}]；content 只收字符串（规格 §1 范围外无多模态）。
    // 收集类型用实仓 prompt.hpp 的 Message（render_chat 的签名以实仓为准，见裁定①）。
    std::vector<Message> conv;
    for (const Json& m : msgs->arr) {
        if (!m.is_obj()) { send_json(res, 400, openai_err("messages 元素不是对象", "invalid_request_error", "messages")); return; }
        const Json* role = m.find("role");
        const Json* content = m.find("content");
        if (role == nullptr || !role->is_str() || content == nullptr || !content->is_str()) {
            send_json(res, 400, openai_err("每条消息需要字符串 role 与 content", "invalid_request_error", "messages"));
            return;
        }
        // content 为空的消息（tool 占位）过滤掉，模板器不管
        if (content->str.empty()) continue;
        conv.emplace_back(Message{role->str, content->str});
    }
    // 裁定⑤：空 prompt 拒绝——所有消息过滤后仍无可渲染内容 → 400。
    if (conv.empty()) {
        send_json(res, 400, openai_err("messages 没有可用内容", "invalid_request_error", "messages"));
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
    if (!numeric_ok) { send_json(res, 400, openai_err("temperature/top_p/max_tokens/seed 必须为数字", "invalid_request_error", nullptr)); return; }
    opts.temperature = static_cast<float>(t);
    opts.top_p = static_cast<float>(p);
    opts.seed = seed;
    if (mt <= 0) { send_json(res, 400, openai_err("max_tokens 必须为正", "invalid_request_error", "max_tokens")); return; }
    opts.max_tokens = static_cast<std::uint64_t>(mt);
    if (!stop_param(body, opts.stop)) { send_json(res, 400, openai_err("stop 必须是字符串或字符串数组", "invalid_request_error", "stop")); return; }
    SamplerParams sp; sp.temperature = opts.temperature; sp.top_p = opts.top_p;
    if (!sp.valid()) { send_json(res, 400, openai_err("temperature 非负、top_p 在 (0,1]", "invalid_request_error", nullptr)); return; }
    // T10：stream 由路由层从 body 读入并传入；此处不再读 body，只按参数分派（SSE 走 serve_sse）。
    std::string prompt;
    std::string rerr;
    if (!render_chat(eng_->chat_template(), conv, /*add_generation_prompt=*/true, prompt, rerr)) {
        // 模板器是裁剪子集（规格 §11 兜底）：不静默错译，明确 500 并说明。
        send_json(res, 500, openai_err("chat 模板不支持的结构：" + rerr, "server_error", nullptr));
        return;
    }
    std::shared_ptr<Job> job = launch(req.path, prompt, opts, stream);
    if (!job) {
        res.status = 503;
        res.set_content(std::string(kQueueFullBody), "application/json; charset=utf-8");
        record_request(nullptr, 503, false);
        return;
    }
    // T10：流式分派——include_usage 宽松收（裁定①）：stream_options 存在且为对象、其下
    // include_usage 为布尔真才算真；其余一律 false，不额外 400。
    if (stream) {
        bool include_usage = false;
        const Json* so = body.find("stream_options");
        if (so != nullptr && so->is_obj()) {
            const Json* iu = so->find("include_usage");
            if (iu != nullptr && iu->kind == Json::kBool) include_usage = iu->b;
        }
        serve_sse(job, include_usage, res);
        return;
    }
    std::string text, drain_err;
    const bool ok = drain_nonstream(job, text, drain_err);
    if (!ok) {
        const int st = job->events->cancelled() ? 499 : 500;   // 499 = 客户端断开，httplib 惯例
        if (st == 500) {
            send_json(res, st, openai_err(drain_err, "server_error", nullptr));
        }
        record_request(job, st, job->events->cancelled());
        return;
    }
    Json r = Json::objv();
    r.set("id", Json::strv("chatcmpl-" + std::to_string(job->id)));
    r.set("object", Json::strv("chat.completion"));
    r.set("created", Json::numv(static_cast<double>(std::time(nullptr))));
    r.set("model", Json::strv(job->model));
    Json choice = Json::objv();
    Json message = Json::objv();
    message.set("role", Json::strv("assistant"));
    message.set("content", Json::strv(text));
    choice.set("index", Json::numv(0.0));
    choice.set("message", std::move(message));
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

// OpenAI 兼容 SSE。provider 由 httplib 在 handler 返回后反复调用；每次调用恰好消费一个
// 事件并写一帧（多帧合并写进同一次 sink.write）。连接断开（write 假）→ cancel → 返回假收尾。
void Serve::serve_sse(const std::shared_ptr<Job>& job, bool include_usage, httplib::Response& res) {
    const std::string id = "chatcmpl-" + std::to_string(job->id);
    const double created = static_cast<double>(std::time(nullptr));
    const std::string model = eng_->model_name();
    const auto first = std::make_shared<bool>(true);
    auto provider = [this, job, include_usage, id, created, model, first](
                        size_t, httplib::DataSink& sink) -> bool {
        JobEvent e;
        const bool got = job->events->pop(e);
        if (!got) { record_request(job, 499, job->events->cancelled()); return false; }
        const auto w = [&](const std::string& s) {
            // 实仓 DataSink::write 是 std::function<bool(const char*, size_t)>（两参调用，见 httplib.h:538）。
            if (!sink.write(s.data(), s.size())) { job->events->cancel(); record_request(job, 499, true); return false; }
            return true;
        };
        Json chunk = Json::objv();
        chunk.set("id", Json::strv(id));
        chunk.set("object", Json::strv("chat.completion.chunk"));
        chunk.set("created", Json::numv(created));
        chunk.set("model", Json::strv(model));
        Json choice = Json::objv();
        Json delta = Json::objv();
        if (e.kind == EventKind::kToken) {
            Json c = Json::arrv();
            choice.set("index", Json::numv(0.0));
            if (*first) { delta.set("role", Json::strv("assistant")); *first = false; }
            if (!e.text.empty()) delta.set("content", Json::strv(e.text));
            choice.set("delta", std::move(delta));
            choice.set("finish_reason", Json::nullv());
            c.push(std::move(choice));
            chunk.set("choices", std::move(c));
            return w("data: " + json_dump(chunk) + "\n\n");
        }
        // 收尾帧：finish_reason；需 usage 时先补一条 usage 帧，再 [DONE]。
        Json c = Json::arrv();
        if (e.kind == EventKind::kDone) {
            choice.set("index", Json::numv(0.0));
            choice.set("delta", Json::objv());
            choice.set("finish_reason", Json::strv(e.finish.empty() ? "stop" : e.finish));
            c.push(std::move(choice));
            chunk.set("choices", std::move(c));
            if (!w("data: " + json_dump(chunk) + "\n\n")) return false;
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
                if (!w("data: " + json_dump(uc) + "\n\n")) return false;
            }
            if (!w("data: [DONE]\n\n")) return false;
            record_request(job, 200, false);
            return false;
        }
        // kError（事件流里以 error 帧出现，一致性让客户端能读到原因）。
        Json inner = Json::objv();
        inner.set("message", Json::strv(e.err.empty() ? "生成失败" : e.err));
        inner.set("type", Json::strv("server_error"));
        inner.set("param", Json::nullv());
        Json er = Json::objv();
        er.set("error", std::move(inner));
        record_request(job, 500, false);
        if (!w("data: " + json_dump(er) + "\n\n")) return false;
        return false;
    };
    res.set_chunked_content_provider("text/event-stream", provider);
}

// Anthropic 兼容 SSE。事件序钉死：message_start → content_block_start → content_block_delta×n →
// content_block_stop → message_delta → message_stop；kError 走 error 事件收尾。形状与 serve_sse
// 同构（provider 每次调用恰好消费一个事件），事件 data 顶层带与事件名一致的 type 字段（Anthropic
// 官方形状，集成测试逐条核对）。
void Serve::serve_anthropic_sse(const std::shared_ptr<Job>& job, httplib::Response& res) {
    const std::string id = "msg_" + std::to_string(job->id);
    const std::string model = eng_->model_name();
    const auto first = std::make_shared<bool>(true);
    auto provider = [this, job, id, model, first](size_t, httplib::DataSink& sink) -> bool {
        const auto w = [&](const std::string& s) {
            // 实仓 DataSink::write 是 std::function<bool(const char*, size_t)>（两参调用，见 httplib.h:538）。
            if (!sink.write(s.data(), s.size())) { job->events->cancel(); record_request(job, 499, true); return false; }
            return true;
        };
        const auto ev = [&](const std::string& name, Json data) {
            return w("event: " + name + "\ndata: " + json_dump(data) + "\n\n");
        };
        JobEvent e;
        if (!job->events->pop(e)) { record_request(job, 499, job->events->cancelled()); return false; }
        if (e.kind == EventKind::kToken) {
            if (*first) {
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
                if (!ev("message_start", std::move(m))) return false;
                Json cb = Json::objv();
                cb.set("type", Json::strv("content_block_start"));
                cb.set("index", Json::numv(0.0));
                Json blk = Json::objv();
                blk.set("type", Json::strv("text"));
                blk.set("text", Json::strv(""));
                cb.set("content_block", std::move(blk));
                if (!ev("content_block_start", std::move(cb))) return false;
                *first = false;
            }
            Json d = Json::objv();
            d.set("type", Json::strv("content_block_delta"));
            d.set("index", Json::numv(0.0));
            Json delta = Json::objv();
            delta.set("type", Json::strv("text_delta"));
            delta.set("text", Json::strv(e.text));
            d.set("delta", std::move(delta));
            return ev("content_block_delta", std::move(d));
        }
        if (e.kind == EventKind::kDone) {
            Json s1 = Json::objv();
            s1.set("type", Json::strv("content_block_stop"));
            s1.set("index", Json::numv(0.0));
            if (!ev("content_block_stop", std::move(s1))) return false;
            Json md = Json::objv();
            md.set("type", Json::strv("message_delta"));
            Json dlt = Json::objv();
            dlt.set("stop_reason", Json::strv(job->finish == "length" ? "max_tokens" : "end_turn"));
            dlt.set("stop_sequence", Json::nullv());
            md.set("delta", std::move(dlt));
            Json us = Json::objv();
            us.set("output_tokens", Json::numv(static_cast<double>(job->n_gen)));
            md.set("usage", std::move(us));
            if (!ev("message_delta", std::move(md))) return false;
            record_request(job, 200, false);
            Json stop = Json::objv();
            stop.set("type", Json::strv("message_stop"));
            return ev("message_stop", std::move(stop));
        }
        // kError：Anthropic 用 error 事件收尾（形状与事件流其他帧一致）。
        Json er = Json::objv();
        Json inner = Json::objv();
        inner.set("type", Json::strv("server_error"));
        inner.set("message", Json::strv(e.err.empty() ? "生成失败" : e.err));
        er.set("type", Json::strv("error"));
        er.set("error", std::move(inner));
        record_request(job, 500, false);
        return ev("error", std::move(er));
    };
    res.set_chunked_content_provider("text/event-stream", provider);
}

void Serve::handle_completions(const httplib::Request& req, httplib::Response& res) {
    Json body;
    std::string jerr;
    if (!json_parse(req.body, body, jerr) || !body.is_obj()) {
        send_json(res, 400, openai_err("请求体不是合法 JSON 对象：" + jerr, "invalid_request_error", nullptr));
        return;
    }
    std::string prompt;
    bool present = false;
    if (!str_param(body, "prompt", prompt, present) || !present) {
        send_json(res, 400, openai_err("缺 prompt 字符串", "invalid_request_error", "prompt"));
        return;
    }
    // 裁定⑤：空 prompt 拒绝。
    if (prompt.empty()) {
        send_json(res, 400, openai_err("prompt 不能为空串", "invalid_request_error", "prompt"));
        return;
    }
    // 参数读取与 handle_chat 同款（temperature/top_p/seed/max_tokens/stop；stream 同 400）。
    GenOptions opts;
    double t = 0.0, p = 0.0, mt = 0.0;
    bool numeric_ok = num_param(body, "temperature", 0.8, t) && num_param(body, "top_p", 0.95, p) &&
                      num_param(body, "max_tokens", 512, mt);
    if (!numeric_ok) { send_json(res, 400, openai_err("temperature/top_p/max_tokens 必须为数字", "invalid_request_error", nullptr)); return; }
    opts.temperature = static_cast<float>(t);
    opts.top_p = static_cast<float>(p);
    opts.max_tokens = static_cast<std::uint64_t>(mt);
    if (mt <= 0) { send_json(res, 400, openai_err("max_tokens 必须为正", "invalid_request_error", "max_tokens")); return; }
    if (!stop_param(body, opts.stop)) { send_json(res, 400, openai_err("stop 必须是字符串或字符串数组", "invalid_request_error", "stop")); return; }
    // 裁定④：completions 同样从 body 读 stream:true → 400。
    const Json* sv = body.find("stream");
    if (sv != nullptr && sv->kind == Json::kBool && sv->b) {
        send_json(res, 400, openai_err("stream:true 尚未支持（SSE 在 T10 落地）", "invalid_request_error", "stream"));
        return;
    }
    std::shared_ptr<Job> job = launch(req.path, prompt, opts, false);
    if (!job) {
        res.status = 503;
        res.set_content(std::string(kQueueFullBody), "application/json; charset=utf-8");
        record_request(nullptr, 503, false);   // 裁定⑧：503 记录与 chat 侧对齐
        return;
    }
    std::string text, drain_err;
    if (!drain_nonstream(job, text, drain_err)) {
        if (!job->events->cancelled()) send_json(res, 500, openai_err(drain_err, "server_error", nullptr));
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
        send_json(res, 400, anthropic_err("invalid_request_error", "请求体不是合法 JSON 对象：" + jerr));
        return;
    }
    double mt = 0.0;
    if (!num_param(body, "max_tokens", 0, mt) || mt <= 0) {
        send_json(res, 400, anthropic_err("invalid_request_error", "max_tokens 必填且为正"));
        return;
    }
    std::vector<Message> conv;
    if (!messages_to_conversation(body, conv, jerr)) {
        send_json(res, 400, anthropic_err("invalid_request_error", jerr));
        return;
    }
    // 裁定⑦：空会话 400——与 chat 域的空 prompt 拒绝同姿态，两个兼容域校验一致。
    if (conv.empty()) {
        send_json(res, 400, anthropic_err("invalid_request_error", "没有可渲染的消息内容"));
        return;
    }
    GenOptions opts;
    double t = 0.0, p = 0.0;
    bool numeric_ok = num_param(body, "temperature", 0.8, t) && num_param(body, "top_p", 0.95, p);
    if (!numeric_ok) { send_json(res, 400, anthropic_err("invalid_request_error", "temperature/top_p 必须为数字")); return; }
    opts.temperature = static_cast<float>(t);
    opts.top_p = static_cast<float>(p);
    opts.max_tokens = static_cast<std::uint64_t>(mt);
    // stop_sequences → stop（字符串数组；缺省空）。
    const Json* stops = body.find("stop_sequences");
    if (stops != nullptr) {
        if (!stops->is_arr()) { send_json(res, 400, anthropic_err("invalid_request_error", "stop_sequences 必须是数组")); return; }
        for (const Json& s : stops->arr) {
            if (!s.is_str()) { send_json(res, 400, anthropic_err("invalid_request_error", "stop_sequences 元素必须是字符串")); return; }
            opts.stop.push_back(s.str);
        }
    }
    std::string prompt, rerr;
    if (!render_chat(eng_->chat_template(), conv, /*add_generation_prompt=*/true, prompt, rerr)) {
        send_json(res, 500, anthropic_err("server_error", "chat 模板不支持的结构：" + rerr));
        return;
    }
    // 裁定②：Json 没有 is_bool——用 kind == kBool 判。
    const Json* s = body.find("stream");
    const bool stream = s != nullptr && s->kind == Json::kBool && s->b;
    std::shared_ptr<Job> job = launch(req.path, prompt, opts, stream);
    if (!job) {
        send_json(res, 503, anthropic_err("overloaded_error", "queue full"));
        record_request(nullptr, 503, false);   // 裁定⑧：503 记录与 chat/completions 侧对齐
        return;
    }
    if (stream) { serve_anthropic_sse(job, res); return; }
    std::string text, drain_err;
    if (!drain_nonstream(job, text, drain_err)) {
        // 记账与 chat 域同款：每个退出路径恰好一次（kError 记 500 写体；客户端断开记 499 不写体）。
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
    Json block = Json::objv();
    block.set("type", Json::strv("text"));
    block.set("text", Json::strv(text));
    Json content = Json::arrv();
    content.push(std::move(block));
    r.set("content", std::move(content));
    r.set("stop_reason", Json::strv(job->finish == "length" ? "max_tokens" : "end_turn"));
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
        send_json(res, 400, anthropic_err("invalid_request_error", "请求体不是合法 JSON 对象：" + jerr));
        return;
    }
    std::vector<Message> conv;
    if (!messages_to_conversation(body, conv, jerr)) {
        send_json(res, 400, anthropic_err("invalid_request_error", jerr));
        return;
    }
    // 与 handle_messages 同款空会话 400，两路校验一致。
    if (conv.empty()) {
        send_json(res, 400, anthropic_err("invalid_request_error", "没有可渲染的消息内容"));
        return;
    }
    std::string prompt, rerr;
    if (!render_chat(eng_->chat_template(), conv, /*add_generation_prompt=*/true, prompt, rerr)) {
        send_json(res, 500, anthropic_err("server_error", "chat 模板不支持的结构：" + rerr));
        return;
    }
    std::vector<std::int32_t> ids;
    if (!eng_->tokenizer().encode(prompt, /*parse_special=*/true, ids, rerr)) {
        send_json(res, 500, anthropic_err("server_error", "分词失败：" + rerr));
        return;
    }
    Json r = Json::objv();
    r.set("input_tokens", Json::numv(static_cast<double>(ids.size())));
    send_json(res, 200, r);
}

void Serve::record_request(const std::shared_ptr<Job>& job, int http_status, bool disconnected) {
    (void) http_status; (void) disconnected;
    if (!job) return;
    const auto enq = job->enqueued_at;   // 缺省构造的 time_point（若从未入队）→ 全零
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
    }
}

void Serve::register_routes() {
    httplib::Server& svr = *svr_;
    // 鉴权（规格 §3）：/web、/api-monitor、/ 与 /health 可达，页面自带 key；
    // /v1/*、/api/*、/metrics 缺失或错误 → 401（对应域的错误形状）。
    svr.set_pre_routing_handler([this](const httplib::Request& req, httplib::Response& res) {
        if (cfg_.api_key.empty()) return httplib::Server::HandlerResponse::Unhandled;
        const std::string want = "Bearer " + cfg_.api_key;
        if (req.get_header_value("Authorization") == want) return httplib::Server::HandlerResponse::Unhandled;
        if (req.path == "/" || req.path.rfind("/web/", 0) == 0 || req.path == "/api-monitor" ||
            req.path == "/health" || req.path.rfind("/fonts/", 0) == 0) {
            return httplib::Server::HandlerResponse::Unhandled;
        }
        if (anthropic_path(req.path)) {
            send_json(res, 401, anthropic_err("authentication_error", "缺少或错误的 API key"));
        } else {
            send_json(res, 401, openai_err("缺少或错误的 API key", "authentication_error", nullptr));
        }
        return httplib::Server::HandlerResponse::Handled;
    });
    // 未挂路由的路径一律 404 JSON（规格 §3 的静态域：/v1/responses、/load 等全走这里）。
    svr.set_error_handler([](const httplib::Request& req, httplib::Response& res) {
        (void) req;
        if (res.status == 404) send_json(res, 404, openai_err("not found", "not_found", nullptr));
    });

    svr.Get("/health", [this](const httplib::Request&, httplib::Response& res) {
        Json h = Json::objv();
        h.set("status", Json::strv("ok"));
        h.set("model", Json::strv(eng_->model_name()));
        h.set("max_context", Json::numv(static_cast<double>(cfg_.facade.max_context)));
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
        meta.set("n_ctx", Json::numv(static_cast<double>(cfg_.facade.max_context)));
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
    // T10：stream 判定移到路由层（请求体非法/非对象 → 视作非流式）；handle_chat 的 stream 即最终真值。
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
    // Anthropic 域：/v1/messages（含流式）+ count_tokens。401 域分流已在 pre-routing（anthropic_path）。
    svr.Post("/v1/messages", [this](const httplib::Request& req, httplib::Response& res) { handle_messages(req, res); });
    svr.Post("/v1/messages/count_tokens", [this](const httplib::Request& req, httplib::Response& res) { handle_count_tokens(req, res); });
    // 状态与属性静态域（规格 §3；/metrics 与 /api/requests 留给 T14）：/v1/status 对外自述、
    // /status 探针简版、/props 能力发现、/slots 上下文环。
    svr.Get("/v1/status", [this](const httplib::Request&, httplib::Response& res) {
        const auto now = std::time(nullptr);
        const std::uint64_t ctx = cfg_.facade.max_context;
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
        mach.set("gpu", Json::nullv());      // T13 接遥测后填
        mach.set("ram", Json::nullv());
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
        dg.set("n_ctx", Json::numv(static_cast<double>(cfg_.facade.max_context)));
        Json prm = Json::objv();
        prm.set("temperature", Json::numv(0.8));
        prm.set("top_p", Json::numv(0.95));
        prm.set("seed", Json::numv(-1.0));
        prm.set("n_predict", Json::numv(512.0));
        dg.set("params", std::move(prm));
        Json r = Json::objv();
        r.set("default_generation_settings", std::move(dg));
        r.set("total_slots", Json::numv(static_cast<double>(cfg_.slots)));
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
            s.set("n_ctx", Json::numv(static_cast<double>(cfg_.facade.max_context)));
            s.set("is_processing", Json::boolv(v.busy));
            s.set("n_prompt_tokens", Json::numv(0.0));
            list.push(std::move(s));
        }
        send_json(res, 200, list);
    });
    // 静态页（T15/T16 落地文件；此刻 404 也正常——T9 只验 API 面）。
    svr.Get("/", [this](const httplib::Request&, httplib::Response& res) {
        res.set_content(web_index_, "text/html; charset=utf-8");
    });
    svr.Get("/api-monitor", [this](const httplib::Request&, httplib::Response& res) {
        res.set_content(web_monitor_, "text/html; charset=utf-8");
    });
    const std::filesystem::path base = std::filesystem::path(__FILE__).parent_path() / "web";
    svr.set_base_dir(base.string().c_str(), "/web");
}

bool Serve::listen(std::string& err) {
    register_routes();
    // HTTP 线程池：2 × 槽位 + 2（规格 §10：线程池大小由入口按槽位数配置）。
    // 本入库的 httplib 用 new_task_queue 成员挂线程池（无 set_thread_pool_size），按实仓 API 接。
    const std::size_t tp = cfg_.threads > 0 ? static_cast<std::size_t>(cfg_.threads) : 2 * cfg_.slots + 2;
    svr_->new_task_queue = [tp] { return new httplib::ThreadPool(tp); };
    for (std::size_t i = 0; i < cfg_.slots; ++i) workers_.emplace_back([this] { worker_main(); });
    if (!svr_->bind_to_port(cfg_.host, cfg_.port)) {
        sched_.stop();
        err = "绑定 " + cfg_.host + ":" + std::to_string(cfg_.port) + " 失败";
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