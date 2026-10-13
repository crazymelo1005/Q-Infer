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

// vector<Json> → Json 数组（render_chat 的 tools 实参形状；空数组与 nullptr 语义等价，
// 调用处统一用 tools.empty() 判——缺省不注入，模板 tools 分支不进入）。
Json tools_array(const std::vector<Json>& tools) {
    Json a = Json::arrv();
    for (const Json& t : tools) a.push(t);
    return a;
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

// 消息列表组装（Anthropic 域）：system + messages（收 user/assistant/tool；content 收字符串或
// 块数组——text 块拼接、thinking 块跳过、tool_use 块→调用、tool_result 块→内联 tool 消息；
// 出现图像块等一律拒绝）→ 与 OpenAI 域同一套 Message 列表交给 render_chat。两个兼容域对同一
// 请求出同一 token 序列（M1.3 硬指标）就靠这条单一渲染路径。工具调用语义对拍 Strata
// anthropic_to_messages：tool_use 的 input 经 tool_arguments 转成模板参数形状（坏 JSON 拒绝干净
// 而非难为整段对话，#510）、tool_result 拆成逐条 {"role":"tool"}。
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
        if (role == nullptr || !role->is_str()) { err = "缺 role 字符串"; return false; }
        const std::string& r = role->str;
        if (r != "user" && r != "assistant" && r != "tool") {
            err = "role 只收 user/assistant/tool";
            return false;
        }
        const Json* content = m.find("content");
        if (content == nullptr) { err = "缺 content"; return false; }
        std::string text;
        std::vector<Json> calls;   // assistant 消息里的 tool_use 块（模板形状，见 prompt.hpp）
        if (content->is_str()) {
            text = content->str;
        } else if (content->is_arr()) {
            for (const Json& blk : content->arr) {
                if (!blk.is_obj()) { err = "content 块不是对象"; return false; }
                const Json* ty = blk.find("type");
                if (ty == nullptr || !ty->is_str()) { err = "content 块缺 type"; return false; }
                if (ty->str == "text") {
                    const Json* t = blk.find("text");
                    if (t == nullptr || !t->is_str()) { err = "文本块缺 text"; return false; }
                    text += t->str;
                } else if (ty->str == "thinking") {
                    // 思考块跳过（不入模板；reasoning_content 的交接 M2 一并做）
                } else if (ty->str == "tool_use") {
                    if (r != "assistant") { err = "tool_use 块只出现在 assistant 消息"; return false; }
                    const Json* name = blk.find("name");
                    if (name == nullptr || !name->is_str()) { err = "tool_use 块缺 name"; return false; }
                    Json fn = Json::objv();
                    fn.set("name", Json::strv(name->str));
                    // input 缺省/坏 JSON → tool_arguments 兜成 {} 或 {"arguments": 原文}（#510）
                    fn.set("arguments", tool_arguments(blk.find("input")));
                    Json call = Json::objv();
                    call.set("function", std::move(fn));
                    calls.push_back(std::move(call));
                } else if (ty->str == "tool_result") {
                    if (r != "user") { err = "tool_result 块只出现在 user 消息"; return false; }
                    std::string rt;
                    const Json* rc = blk.find("content");
                    if (rc == nullptr) { err = "tool_result 块缺 content"; return false; }
                    if (rc->is_str()) {
                        rt = rc->str;
                    } else if (rc->is_arr()) {
                        for (const Json& rb : rc->arr) {
                            if (!rb.is_obj()) { err = "tool_result content 块不是对象"; return false; }
                            const Json* rty = rb.find("type");
                            if (rty == nullptr || !rty->is_str() || rty->str != "text") {
                                err = "tool_result 内容只收文本块（图像等多模态 M2）";
                                return false;
                            }
                            const Json* t = rb.find("text");
                            if (t == nullptr || !t->is_str()) { err = "tool_result 文本块缺 text"; return false; }
                            rt += t->str;
                        }
                    } else {
                        err = "tool_result content 必须是字符串或块数组";
                        return false;
                    }
                    // 内联 tool 消息（逐条紧跟，Strata anthropic_to_messages 同款）：
                    // 渲染时模板按 role=tool 出 <tool_response>；空结果按统一空过滤跳过。
                    if (rt.empty()) continue;
                    conv.emplace_back(Message{"tool", std::move(rt)});
                } else {
                    err = "content 块只收 text/thinking/tool_use/tool_result（图像等多模态 M2）";
                    return false;
                }
            }
        } else {
            err = "content 必须是字符串或块数组";
            return false;
        }
        // 统一空过滤：文本与调用都空才整条跳过（与 OpenAI 域同款）。
        if (text.empty() && calls.empty()) continue;
        conv.emplace_back(Message{r, std::move(text), std::move(calls)});
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
      prompt_cache_(cfg_.conversation_cache),
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
            // 编码查缓存（spec §6 的 conversation_cache）：查询在锁外，命中拿缓存的 ids 免做
            // BPE（也不写各 worker 共享的 piece_cache_）；未命中才在锁内 encode 并回填。
            // generate 一如 v0 仍须在 engine_mu_ 内串行（写共享 KV），命中省的是分词、锁照进。
            // 缓存只在锁外查、锁内回填：持锁方向恒为 engine→cache，无反向路径。
            const bool hit = prompt_cache_.lookup(job->prompt, ids);
            std::lock_guard<std::mutex> lock(engine_mu_);
            if (!hit) {
                if (!eng_->tokenizer().encode(job->prompt, /*parse_special=*/true, ids, err)) {
                    job->finish = "error";
                    job->error = "分词失败：" + err;
                    // 先落完成时刻再停流：HTTP 侧读到 done_at 时必然已就绪（与其他错误路径同构），
                    // 否则 HTTP 线程可能赶在 done_at 之前调 record_request、被其守卫整条跳过。
                    job->done_at = std::chrono::steady_clock::now();
                    job->events->stop_with_error(job->error);
                    sched_.done(job);
                    continue;
                }
                prompt_cache_.store(job->prompt, ids);
            }
            job->prompt_ids = std::move(ids);
            job->n_prompt = job->prompt_ids.size();
            if (hit) reused_.fetch_add(job->n_prompt);   // 命中 = 这整段 prompt 免编码（totals.reused）
            ok = eng_->generate(job->prompt_ids, job->opts,
                                [&](const std::string& piece) -> bool {
                                    if (job->first_at == std::chrono::steady_clock::time_point{})
                                        job->first_at = std::chrono::steady_clock::now();
                                    // 停滞（未断开）的 SSE 客户端可让 worker 阻塞在 push 上并持有 engine_mu_，
                                    // 冻结全部槽位——M2 调优项（锁粒度/超时），M1 记录不住。
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
    // messages → Message 列表（OpenAI 域）：role 收 system/user/assistant/tool（developer 并入
    // system）；content 收 null/字符串/文本块数组（图像等多模态 M2 再开）；assistant 的
    // tool_calls 历史经 parse_openai_tool_calls 归一成模板形状。语义对拍 Strata
    // openai_to_messages：只滤文本与调用都空的消息（assistant 的 tool_calls 消息 content 可空）。
    std::vector<Message> conv;
    for (const Json& m : msgs->arr) {
        if (!m.is_obj()) { send_json(res, 400, openai_err("messages 元素不是对象", "invalid_request_error", "messages")); return; }
        const Json* role = m.find("role");
        if (role == nullptr || !role->is_str()) {
            send_json(res, 400, openai_err("每条消息需要字符串 role", "invalid_request_error", "messages"));
            return;
        }
        const std::string r = role->str == "developer" ? "system" : role->str;
        if (r != "system" && r != "user" && r != "assistant" && r != "tool") {
            send_json(res, 400, openai_err("role 只收 system/user/assistant/tool", "invalid_request_error", "messages"));
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
                        send_json(res, 400, openai_err("content 块只收 {type: text}（图像等多模态 M2）", "invalid_request_error", "messages"));
                        return;
                    }
                    const Json* t = blk.find("text");
                    if (t == nullptr || !t->is_str()) {
                        send_json(res, 400, openai_err("文本块缺 text", "invalid_request_error", "messages"));
                        return;
                    }
                    text += t->str;
                }
            } else {
                send_json(res, 400, openai_err("content 必须是字符串或块数组", "invalid_request_error", "messages"));
                return;
            }
        }
        std::vector<Json> calls;
        const Json* tcs = m.find("tool_calls");
        if (tcs != nullptr) {
            std::string cerr;
            if (!parse_openai_tool_calls(*tcs, calls, cerr)) {
                send_json(res, 400, openai_err("tool_calls 解析失败：" + cerr, "invalid_request_error", "tool_calls"));
                return;
            }
        }
        if (text.empty() && calls.empty()) continue;   // 空消息（tool 占位等）过滤，模板器不管
        conv.emplace_back(Message{r, std::move(text), std::move(calls)});
    }
    // tools → 内部工具形状（normalize_tools 的 OpenAI 解包）；坏了 400（错误要先于入队）。
    std::vector<Json> tools;
    const Json* tv = body.find("tools");
    if (tv != nullptr) {
        std::string terr;
        if (!normalize_tools(*tv, /*anthropic=*/false, tools, terr)) {
            send_json(res, 400, openai_err("tools 解析失败：" + terr, "invalid_request_error", "tools"));
            return;
        }
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
    Json tj = tools_array(tools);
    if (!render_chat(eng_->chat_template(), conv, /*add_generation_prompt=*/true, prompt, rerr,
                     tools.empty() ? nullptr : &tj)) {
        // 模板器是裁剪子集（规格 §11 兜底）：不静默错译，明确 500 并说明。
        send_json(res, 500, openai_err("chat 模板不支持的结构：" + rerr, "server_error", nullptr));
        return;
    }
    std::shared_ptr<Job> job = launch(req.path, prompt, opts, stream);
    if (!job) {
        res.status = 503;
        res.set_content(std::string(kQueueFullBody), "application/json; charset=utf-8");
        // 503 未入队：无计时数据，不入记录环（环只收已接受请求）。
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
        serve_sse(job, include_usage, tools, res);
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
    // 生成文本拆事件：<tool_call> 块 → 调用；坏标记并回文本（split_generation 语义）。调用与
    // 文本保持生成顺序（Strata OutputParser 同款）；content 全空时置 null（OpenAI 惯例）。
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
    // finish_reason：有调用且自然停 → tool_calls（Strata openai_chunks 口径）。
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

// OpenAI 兼容 SSE。provider 由 httplib 在 handler 返回后反复调用；每次调用恰好写一帧，事件
// 可展开成多帧（首帧 role、文本增量、工具调用块、终帧/usage/[DONE]），帧在事件消费时整批攒入
// pending 逐次回落。生成文本经 ToolCallStreamer 增量切分——半个 <tool_call> 跨增量留缓冲等齐，
// 坏块（解析失败）按文本放行。帧形状对拍 Strata openai_chunks：首帧 {delta:{role,content:""}}、
// 工具调用整块一帧 {delta:{tool_calls:[{index,id,type,function}]}}、终帧 finish_reason（有调用
// 且自然停 → tool_calls）。连接断开（write 假）→ cancel → 499 记账 → 返回假收尾。
void Serve::serve_sse(const std::shared_ptr<Job>& job, bool include_usage,
                      const std::vector<Json>& tools, httplib::Response& res) {
    const std::string id = "chatcmpl-" + std::to_string(job->id);
    const double created = static_cast<double>(std::time(nullptr));
    const std::string model = eng_->model_name();
    struct State {
        ToolCallStreamer st;
        // 工具定义按值随 State 留在堆上：provider 的生命周期长于 handler（httplib 在路由函数
        // 返回后才反复调 chunk provider），引用实参会悬垂。
        std::vector<Json> tools;
        std::vector<std::string> pending;   // 待写帧（front 弹出）
        bool first = true;                  // 首帧 role chunk 未写
        int calls = 0;                      // 已发出的工具调用块数（finish_reason 判据 + 块 index）
        bool done = false;                  // kDone/kError 已消费
        int final_status = 200;
        bool recorded = false;
    };
    auto state = std::make_shared<State>();
    state->tools = tools;
    auto provider = [this, job, include_usage, id, created, model, state](
                        std::size_t, httplib::DataSink& sink) -> bool {
        const auto w = [&](const std::string& s) {
            // 实仓 DataSink::write 是 std::function<bool(const char*, size_t)>（两参调用，见 httplib.h:538）。
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
            // kBlock：完整调用块 → 工具调用帧（整块一帧，stream_tools=false 形态）；解析失败
            // （幻觉出的坏标记）→ 按文本放行。
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
        // 写一帧；无帧可写时先消费作业事件攒帧（一个事件可展开成多帧，逐次回落）。
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
                // 收尾前把缓冲里等齐的残余切出（未闭合调用按原样文本释放，#211）。
                std::vector<ToolCallStreamer::Event> evs;
                state->st.finish(evs);
                for (const auto& sev : evs) emit_streamer_event(sev);
                // 终帧：finish_reason（有调用且自然停 → tool_calls）；需 usage 时先补帧，再 [DONE]。
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
            // kError（事件流里以 error 帧出现，一致性让客户端能读到原因）。
            Json inner = Json::objv();
            inner.set("message", Json::strv(e.err.empty() ? "生成失败" : e.err));
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
            // kDone 已消费、帧全部送出：正常完成。必须调 sink.done() 写 chunked 终止块
            // （0\r\n\r\n）——httplib 对 provider 返回 false 只是关连接、不写终止块，严格
            // 的 chunked 客户端（浏览器、http.client）会报 ERR_INCOMPLETE_CHUNKED_ENCODING
            // / IncompleteRead；curl 容忍 EOF，故此前集成测试未暴露。done() 内部写终止块
            // 并把流置为 finished，while 循环随之正常退出。
            if (!state->recorded) { record_request(job, state->final_status, false); state->recorded = true; }
            if (state->done) { sink.done(); return true; }
            // 事件缓冲异常终止（kDone 未到、pop 失败已 499 记账）：provider 结束关连接。
            return false;
        }
        const std::string frame = state->pending.front();
        state->pending.erase(state->pending.begin());
        if (!w(frame)) return false;
        // 终帧落地后才记账：断连/中途失败时 w 内已按 499 记，这里不得叠记。
        if (state->pending.empty() && state->done && !state->recorded) {
            record_request(job, state->final_status, false);
            state->recorded = true;
        }
        return true;
    };
    res.set_chunked_content_provider("text/event-stream", provider);
}

// Anthropic 兼容 SSE。事件序：message_start → 按需逐块 content_block_start/delta/stop → message_delta
// → message_stop；kError 走 error 事件收尾。文本块在首段非空文本处打开（text_delta 逐段）；工具块以
// tool_use 打开（id/name/空 input）并紧跟一条 input_json_delta 带整份参数（stream_tools=false 形态），
// tool_use 前无条件关掉当前块（块序共享一个递增 index）。stop_reason "tool_use" 仅当有工具块且自然停
// （Strata anthropic_events 同款）；生成文本经 ToolCallStreamer 增量切分，坏块按文本放行，未闭合
// 调用在收尾时按原样文本释放（#211）。事件 data 顶层带与事件名一致的 type 字段（Anthropic 官方
// 形状，集成测试逐条核对）。
void Serve::serve_anthropic_sse(const std::shared_ptr<Job>& job, const std::vector<Json>& tools,
                                httplib::Response& res) {
    const std::string id = "msg_" + std::to_string(job->id);
    const std::string model = eng_->model_name();
    struct State {
        ToolCallStreamer st;
        // 工具定义按值随 State 留在堆上（理由同 serve_sse：provider 生命周期长于 handler）。
        std::vector<Json> tools;
        std::vector<std::string> pending;   // 待写帧（front 弹出）
        bool started = false;               // message_start 未写
        bool block_open = false;            // 当前有打开的块（末块 index = block_idx - 1）
        int block_idx = 0;                  // 下一块序号（text/tool_use 共享）
        bool used_tool = false;
        bool done = false;
        int final_status = 200;
        bool recorded = false;
    };
    auto state = std::make_shared<State>();
    state->tools = tools;
    auto provider = [this, job, id, model, state](std::size_t, httplib::DataSink& sink) -> bool {
        const auto w = [&](const std::string& s) {
            // 实仓 DataSink::write 是 std::function<bool(const char*, size_t)>（两参调用，见 httplib.h:538）。
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
        // 流式切分事件 → 块状态机（kToken 与 kDone 收尾共用同一份语义）。
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
                // 坏标记：并回文本放行，不当错误毁掉请求。
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
            close_block();   // tool_use 永远先关当前块再开新块
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
        // 写一帧；无帧可写时先消费作业事件攒帧。
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
                // 收尾：先释放缓冲里等齐的残余，再关当前块，最后 message_delta/message_stop。
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
            // kError：Anthropic 用 error 事件收尾（形状与事件流其他帧一致）。
            Json er = Json::objv();
            Json inner = Json::objv();
            inner.set("type", Json::strv("server_error"));
            inner.set("message", Json::strv(e.err.empty() ? "生成失败" : e.err));
            er.set("type", Json::strv("error"));
            er.set("error", std::move(inner));
            enqueue("error", std::move(er));
            state->done = true;
            state->final_status = 500;
            continue;
        }
        if (state->pending.empty()) {
            // kDone 已消费、帧全部送出：正常完成。必须调 sink.done() 写 chunked 终止块
            // （0\r\n\r\n）——httplib 对 provider 返回 false 只是关连接、不写终止块，严格
            // 的 chunked 客户端（浏览器、http.client）会报 ERR_INCOMPLETE_CHUNKED_ENCODING
            // / IncompleteRead；curl 容忍 EOF，故此前集成测试未暴露。done() 内部写终止块
            // 并把流置为 finished，while 循环随之正常退出。
            if (!state->recorded) { record_request(job, state->final_status, false); state->recorded = true; }
            if (state->done) { sink.done(); return true; }
            // 事件缓冲异常终止（kDone 未到、pop 失败已 499 记账）：provider 结束关连接。
            return false;
        }
        const std::string frame = state->pending.front();
        state->pending.erase(state->pending.begin());
        if (!w(frame)) return false;
        // 终帧落地后才记账：断连/中途失败时 w 内已按 499 记，这里不得叠记。
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
        send_json(res, 400, openai_err("completions 暂不支持 stream:true（chat/messages 域已支持）", "invalid_request_error", "stream"));
        return;
    }
    std::shared_ptr<Job> job = launch(req.path, prompt, opts, false);
    if (!job) {
        res.status = 503;
        res.set_content(std::string(kQueueFullBody), "application/json; charset=utf-8");
        // 503 未入队：无计时数据，不入记录环（环只收已接受请求）。
        return;
    }
    job->raw_body = req.body;   // T14：原始请求体进 Records
    std::string text, drain_err;
    if (!drain_nonstream(job, text, drain_err)) {
        const int st = job->events->cancelled() ? 499 : 500;
        if (st == 500) send_json(res, 500, openai_err(drain_err, "server_error", nullptr));
        record_request(job, st, job->events->cancelled());   // 记账与 chat/messages 域同款：每个退出路径恰好一次
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
    // tools → 内部工具形状（normalize_tools 的 Anthropic 改名：input_schema → parameters）。
    std::vector<Json> tools;
    const Json* tv = body.find("tools");
    if (tv != nullptr) {
        std::string terr;
        if (!normalize_tools(*tv, /*anthropic=*/true, tools, terr)) {
            send_json(res, 400, anthropic_err("invalid_request_error", "tools 解析失败：" + terr));
            return;
        }
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
    Json tj = tools_array(tools);
    if (!render_chat(eng_->chat_template(), conv, /*add_generation_prompt=*/true, prompt, rerr,
                     tools.empty() ? nullptr : &tj)) {
        send_json(res, 500, anthropic_err("server_error", "chat 模板不支持的结构：" + rerr));
        return;
    }
    // 裁定②：Json 没有 is_bool——用 kind == kBool 判。
    const Json* s = body.find("stream");
    const bool stream = s != nullptr && s->kind == Json::kBool && s->b;
    std::shared_ptr<Job> job = launch(req.path, prompt, opts, stream);
    if (!job) {
        send_json(res, 503, anthropic_err("overloaded_error", "queue full"));
        // 503 未入队：无计时数据，不入记录环（环只收已接受请求）。
        return;
    }
    job->raw_body = req.body;   // T14：原始请求体进 Records（流式请求也要带）
    if (stream) { serve_anthropic_sse(job, tools, res); return; }
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
    // 生成文本拆事件：文本块与 tool_use 块保持生成顺序（Anthropic 官方块形状）。
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
    // stop_reason：有调用且自然停 → tool_use（Strata anthropic 口径）。
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
    // tools 同样归一（Anthropic 形状），渲染时才与真实请求同口径（count 与补全同 prompt）。
    std::vector<Json> tools;
    const Json* tv = body.find("tools");
    if (tv != nullptr) {
        std::string terr;
        if (!normalize_tools(*tv, /*anthropic=*/true, tools, terr)) {
            send_json(res, 400, anthropic_err("invalid_request_error", "tools 解析失败：" + terr));
            return;
        }
    }
    std::string prompt, rerr;
    Json tj = tools_array(tools);
    if (!render_chat(eng_->chat_template(), conv, /*add_generation_prompt=*/true, prompt, rerr,
                     tools.empty() ? nullptr : &tj)) {
        send_json(res, 500, anthropic_err("server_error", "chat 模板不支持的结构：" + rerr));
        return;
    }
    std::vector<std::int32_t> ids;
    if (!prompt_cache_.lookup(prompt, ids)) {
        // 与 worker_main 同一把 engine_mu_：encode 写各 worker 共享的 piece_cache_，
        // 脱离锁与并发请求是数据竞争（见 server.hpp 头注释）；chat_template() 是只读访问不需锁。
        // 锁只罩引擎可变访问：失败路径的 500 响应移到锁外，慢读客户端写响应不得持锁冻结全槽。
        // 命中免进锁；未命中仍在锁内 encode 与回填（锁序恒定 engine→cache，无反向路径）。
        std::lock_guard<std::mutex> lock(engine_mu_);
        if (!eng_->tokenizer().encode(prompt, /*parse_special=*/true, ids, rerr)) {
            send_json(res, 500, anthropic_err("server_error", "分词失败：" + rerr));
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
    // 未达完成态（done_at 缺省）不入账：唯一剩余场景是客户端断开（499）路径里 cancel() 与
    // worker 的 done_at 写入之间没有 happens-before 边——HTTP 线程在 worker 落 done_at 之前
    // 读到缺省值。未完成的请求不是「最近一次完成的请求」，整体跳过，避免缺省 time_point
    // 化成约 -1.7e9 s 的假读数污染 /v1/status 的 last_timings。
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
        totals_.requests += 1;   // 完成计数（Strata 口径的 totals.requests：每次完成时刻 +1）
        // 计时口径：prefill ≈ first_at- leased_at，decode ≈ done_at - first_at。
        const auto ms = [](std::chrono::steady_clock::duration d) {
            return std::chrono::duration<double, std::milli>(d).count();
        };
        if (job->first_at != std::chrono::steady_clock::time_point{})
            totals_.prompt_ms += ms(job->first_at - job->leased_at);
        const auto start = job->first_at != std::chrono::steady_clock::time_point{}
                               ? job->first_at : job->leased_at;
        if (job->done_at > start) totals_.decode_ms += ms(job->done_at - start);
        // T13：tok_s_mean 的全程累计（生成 token ÷ 解码耗时；与 telemetry_extra 同锁，采样线程前一次
        // 读数与本次累加之间有 happens-before 边）。编码失败 n_gen 恒 0 被 n_gen > 0 拦掉，
        // 断开竞态（done_at 未落）由上面的 done_at 守卫拦。
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
    // ---- 延迟直方图（Prometheus text 端口的 ttft/itl/e2e 家族）----
    // 与 totals 同口径只在完成时刻计一次；起止统一为 enq（入队）→ 首 token → done_at，与 e2e
    // 同一起点。无首 token 的失败请求 ttft 传 0，Latencies 端 >0 守卫跳过，不污染桶。
    const auto start_of_decode = job->first_at != std::chrono::steady_clock::time_point{}
                                     ? job->first_at : job->leased_at;
    lat_.observe(job->first_at != std::chrono::steady_clock::time_point{}
                     ? std::chrono::duration<double>(job->first_at - enq).count() : 0.0,
                 job->n_gen,
                 job->done_at > start_of_decode
                     ? std::chrono::duration<double>(job->done_at - start_of_decode).count() : 0.0,
                 wall);
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
        r.set("compute_threads", Json::numv(static_cast<double>(cfg_.compute_threads)));
        // GPU 设备侧：offload_enabled = 配置里给了 --gpu；gpu_devices = auto 或显式列表。
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
    svr.Get("/metrics", [this](const httplib::Request& req, httplib::Response& res) {
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
        {
            // 缓存实况（spec §6 的 conversation_cache 填实）：逐请求计数，全是真实值。
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
        // 内容协商（Strata 契约，serve/prometheus.py 的 wants_prometheus）：Accept 带 text/plain
        // 或 openmetrics，或 ?format=prometheus → Prometheus text；前端 fetch 与无头 curl
        // （Accept 缺省 */*）保持 JSON。text 与 JSON 同源——渲染只读刚构建的 r，单一事实源。
        // 谓词按 Strata 的「query 含 format=prometheus」口径写，故传回完整的 k=v 片段而非裸值。
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