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

// ---- 记录序列化（列表视图不带大字段，详情全带；Strata request_records 同款）----
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

// 历史系列（log. 每点：值或 null）。
Json series_json(const std::vector<std::optional<double>>& s) {
    Json a = Json::arrv();
    for (const auto& v : s) a.push(v ? Json::numv(*v) : Json::nullv());
    return a;
}

// 一位小数（Strata 的 round(x, 1)），仅展示用。
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

// v/unit，四舍五入 digits 位；nullopt → null。
static Json scaled_mib_gib(const std::optional<double>& v, double unit, int digits) {
    if (!v) return Json::nullv();
    const double p = std::pow(10.0, digits);
    return Json::numv(std::round(*v / unit * p) / p);
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
    // T13：遥测在构造函数启动（launch 是每请求入口，放那无请求时 tel_ 恒 null、T14 的 /metrics
    // 首帧无数据）；进程生存期采样，析构经 tel_ 的 dtor join。
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
            // v0 串行化：encode 与 generate 都在锁内。encode 写各 worker 共享的 piece_cache_
            // （tokenizer.cpp 的 encode_plain），脱离锁是两个 worker 并发改同一张表（见 server.hpp 头注释）。
            std::lock_guard<std::mutex> lock(engine_mu_);
            if (!eng_->tokenizer().encode(job->prompt, /*parse_special=*/true, ids, err)) {
                job->finish = "error";
                job->error = "分词失败：" + err;
                job->events->stop_with_error(job->error);
                job->done_at = std::chrono::steady_clock::now();   // T14：编码失败也要收尾成记录（与其他错误路径同构）
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
    job->raw_body = req.body;   // T14：原始请求体进 Records（流式请求也要带）
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
    job->raw_body = req.body;   // T14：原始请求体进 Records
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
    job->raw_body = req.body;   // T14：原始请求体进 Records（流式请求也要带）
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
    (void) http_status;
    if (!job) return;
    // 未达完成态（done_at 缺省）不入账：worker 的编码失败分支不写 done_at 就 sched_.done()，
    // 客户端断开的 499 路径里 cancel() 与 worker 的 done_at 写入之间也无 happens-before 边，
    // 两种情况都会留下全零 done_at。未完成的请求不是「最近一次完成的请求」，
    // 整体跳过，避免缺省 time_point 化成约 -1.7e9 s 的假读数污染 /v1/status 的 last_timings。
    if (job->done_at == std::chrono::steady_clock::time_point{}) return;
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
        totals_.prompt_tokens += job->n_prompt;
        totals_.completion_tokens += job->n_gen;
        // 计时口径：prefill ≈ first_at- leased_at，decode ≈ done_at - first_at。
        const auto ms = [](std::chrono::steady_clock::duration d) {
            return std::chrono::duration<double, std::milli>(d).count();
        };
        if (job->first_at != std::chrono::steady_clock::time_point{})
            totals_.prompt_ms += ms(job->first_at - job->leased_at);
        if (job->done_at > job->first_at) totals_.decode_ms += ms(job->done_at - job->first_at);
        // T13：tok_s_mean 的全程累计（生成 token ÷ 解码耗时；与 telemetry_extra 同锁，采样线程前一次
        // 读数与本次累加之间有 happens-before 边）。编码失败/断开路径已在上面的早退拦掉。
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
    // ---- T14：Records 环（同一 job 只建一条；断开/错误都收尾成 error 态）----
    if (job->record_id.empty()) {
        RequestBegin b;
        b.path = job->path;
        b.model = job->model;
        b.input = job->raw_body;
        b.stream = job->stream;
        Json raw;   // response_format 从原始请求体取（解析失败也不碍事，落空串）
        std::string jerr;
        if (json_parse(job->raw_body, raw, jerr) && raw.is_obj()) {
            const Json* rf = raw.find("response_format");
            if (rf != nullptr && rf->is_obj()) {
                const Json* ty = rf->find("type");
                if (ty != nullptr && ty->is_str()) b.response_format = ty->str;
            }
        }
        b.created = static_cast<double>(std::time(nullptr)) - wall;   // ≈ 请求开始
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
        if (tel_) {
            const auto ts = tel_->snapshot();
            const auto& hw = ts.now;
            const auto& st = ts.stat;
            auto get = [&hw](const char* k) -> std::optional<double> {
                return hw.count(k) ? std::optional<double>(hw.at(k)) : std::nullopt;
            };
            if (!st.gpu_name.empty()) {   // Strata: machine.gpu 在无卡时整个为 null
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
    svr.Get("/metrics", [this](const httplib::Request&, httplib::Response& res) {
        const auto now = std::time(nullptr);
        const std::uint64_t ctx = cfg_.facade.max_context;
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
        live.set("phase", Json::nullv());          // v0 无阶段跟踪
        live.set("prompt_tokens", Json::nullv());  // v0 不做进行中请求的逐项视图
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
        live.set("tok_s_window_s", Json::nullv());   // v0 无滑动窗口（M2 补）
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
        const auto list = records_.list();   // 最新在前
        Json reqs = Json::arrv();
        for (std::size_t i = 0; i < list.size() && i < 12; ++i) reqs.push(record_summary_json(list[i]));
        r.set("requests", std::move(reqs));
        Json cc = Json::objv();
        cc.set("enabled", Json::boolv(false));   // 索引器统计 M2 接（spec §6）
        r.set("conversation_cache", std::move(cc));
        r.set("requests_kept", Json::numv(static_cast<double>(Records::kCap)));
        Json total = Json::objv();
        {
            std::lock_guard<std::mutex> lk(last_mu_);
            total.set("prompt_tokens", Json::numv(static_cast<double>(totals_.prompt_tokens)));
            total.set("reused", Json::numv(0.0));
            total.set("completion_tokens", Json::numv(static_cast<double>(totals_.completion_tokens)));
            total.set("prompt_ms", Json::numv(totals_.prompt_ms));
            total.set("decode_ms", Json::numv(totals_.decode_ms));
        }
        r.set("totals", std::move(total));
        const auto ts = tel_ ? tel_->snapshot() : TelemetrySnapshot{};
        Json hw = Json::objv();
        for (const auto& [k, v] : ts.now) hw.set(k, v ? Json::numv(*v) : Json::nullv());
        if (ts.gpus.size() > 1) {   // Strata：仅多卡时带 gpus[]
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
        hs.set("gpu_note", Json::nullv());   // v0 恒空（见 telemetry.hpp 注释）
        hs.set("cpu_name", Json::strv(ts.stat.cpu_name));
        hs.set("cores", Json::numv(static_cast<double>(ts.stat.cores)));
        hs.set("threads", Json::numv(static_cast<double>(ts.stat.threads)));
        r.set("hardware_static", std::move(hs));
        Json hist = Json::objv();
        for (const auto& [k, vec] : ts.history) hist.set(k, series_json(vec));
        r.set("history", std::move(hist));
        r.set("time", Json::numv(static_cast<double>(now)));
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
        out.set("loaded", Json::boolv(true));   // 引擎启动即装载（v0 无热装载）
        out.set("auto_load", Json::boolv(false));
        send_json(res, 200, out);
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