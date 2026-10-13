#include "serve/backend_strata.hpp"

#include <utility>

namespace qinfer::serve {

bool StrataBackend::open(StrataBackendConfig cfg, StrataBackend& out, std::string& err) {
    err.clear();
    if (!engine::EngineDriver::start(cfg.driver, out.driver_, err)) return false;
    out.tokenizer_ = std::move(cfg.tokenizer);
    out.chat_template_ = std::move(cfg.chat_template);
    out.model_name_ = std::move(cfg.model_name);
    out.cfg_ = std::move(cfg.facade);
    out.cfg_.max_context = out.driver_.max_context();  // the engine's real capacity, not the default
    return true;
}

bool StrataBackend::generate(const std::vector<std::int32_t>& prompt, const GenOptions& opt,
                             const std::function<bool(const std::string&)>& on_token,
                             GenResult& res, std::string& err) {
    res = GenResult{};
    err.clear();
    res.n_prompt = prompt.size();

    engine::GenRequest req;
    req.prompt = prompt;
    req.max_new = static_cast<std::int64_t>(opt.max_tokens);
    req.sampling.temperature = opt.temperature;
    req.sampling.top_p = opt.top_p;
    req.sampling.seed = opt.seed;

    std::string text;
    std::vector<std::int32_t> ids;
    bool client_stop = false;

    engine::GenSummary sum;
    const bool ok = driver_.generate(
        req,
        [&](std::int32_t id) {
            ids.push_back(id);
            std::string piece;
            if (!tokenizer_.token_bytes(id, piece)) piece.clear();
            text += piece;
            if (on_token && !on_token(piece)) {
                client_stop = true;
                return false;
            }
            return true;
        },
        sum, err);
    if (!ok) return false;
    if (client_stop) {
        err = "client disconnected";
        return false;
    }

    res.text = text;
    res.tokens = std::move(ids);
    res.n_gen = sum.generated;
    if (sum.finish == "stop") {
        res.finish = Finish::kStop;
    } else if (sum.finish == "cancel") {
        err = "engine cancelled the request";
        return false;
    } else {
        res.finish = Finish::kLength;  // length | nonfinite | anything else the engine reports
    }
    return true;
}

}  // namespace qinfer::serve
