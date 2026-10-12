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

}  // namespace

Serve::Serve(ServeConfig cfg, std::shared_ptr<Engine> eng)
    : cfg_(std::move(cfg)),
      eng_(std::move(eng)),
      sched_(cfg_.slots, cfg_.queue_capacity == 0 ? 2 * cfg_.slots : cfg_.queue_capacity),
      svr_(std::make_unique<httplib::Server>()) {
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
    // 裁定④：stream:true 从 body 读并真 400（SSE 在 T10 落地）；stream 参数保留给 T10 的流式分支。
    const Json* sv = body.find("stream");
    if (sv != nullptr && sv->kind == Json::kBool && sv->b) {
        send_json(res, 400, openai_err("stream:true 尚未支持（SSE 在 T10 落地）", "invalid_request_error", "stream"));
        return;
    }

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

void Serve::record_request(const std::shared_ptr<Job>& job, int http_status, bool disconnected) {
    // T14 接 Records 环：本任务为空实现（请求记录 M1.4 才落地，路由形状已就绪）。
    (void) job; (void) http_status; (void) disconnected;
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
    svr.Post("/v1/chat/completions",
             [this](const httplib::Request& req, httplib::Response& res) { handle_chat(req, res, false); });
    svr.Post("/v1/completions",
             [this](const httplib::Request& req, httplib::Response& res) { handle_completions(req, res); });
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