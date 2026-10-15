#include "serve/backend_strata.hpp"

#include <utility>

namespace qinfer::serve {

std::size_t utf8_complete_prefix(const std::string& buf) {
    std::size_t i = 0;
    while (i < buf.size()) {
        const unsigned char b = static_cast<unsigned char>(buf[i]);
        std::size_t need = 1;
        if (b < 0x80) {
            need = 1;
        } else if ((b & 0xE0) == 0xC0) {
            need = 2;
        } else if ((b & 0xF0) == 0xE0) {
            need = 3;
        } else if ((b & 0xF8) == 0xF0) {
            need = 4;
        }
        // else: invalid lead byte -> need stays 1, so one byte is emitted and the stream moves on.
        if (i + need > buf.size()) break;  // a partial sequence at the tail: hold it back
        i += need;
    }
    return i;
}

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
    // Extended sampling parameters: only set when explicitly provided (non-default)
    if (opt.top_k > 0) req.sampling.top_k = static_cast<int>(opt.top_k);
    if (opt.min_p >= 0) req.sampling.min_p = opt.min_p;
    if (opt.repeat_penalty != 1.0f) req.sampling.penalty_repeat = opt.repeat_penalty;
    if (opt.frequency_penalty != 0.0f) req.sampling.penalty_freq = opt.frequency_penalty;
    if (opt.presence_penalty != 0.0f) req.sampling.penalty_present = opt.presence_penalty;

    std::string text;
    std::vector<std::int32_t> ids;
    bool client_stop = false;
    bool stop_hit = false;
    pending_.clear();

    engine::GenSummary sum;
    const bool ok = driver_.generate(
        req,
        [&](std::int32_t id) {
            // Once we have decided to stop, ignore tokens the engine already had in flight.
            if (stop_hit || client_stop) return false;
            ids.push_back(id);
            std::string piece;
            if (!tokenizer_.token_bytes(id, piece)) piece.clear();
            pending_ += piece;

            // Emit only complete characters: a multi-byte piece split across two tokens waits.
            const std::size_t done = utf8_complete_prefix(pending_);
            if (done == 0) return true;
            std::string chunk = pending_.substr(0, done);
            pending_.erase(0, done);

            // stop strings: truncate at the first hit and stop the engine.
            if (!opt.stop.empty()) {
                const std::string joined = text + chunk;
                std::size_t hit_at = std::string::npos;
                for (const std::string& s : opt.stop) {
                    if (s.empty()) continue;
                    const std::size_t p = joined.find(s);
                    if (p != std::string::npos && (hit_at == std::string::npos || p < hit_at)) {
                        hit_at = p;
                    }
                }
                if (hit_at != std::string::npos) {
                    const std::size_t keep_in_chunk =
                        hit_at > text.size() ? hit_at - text.size() : 0;
                    const std::string emit = chunk.substr(0, keep_in_chunk);
                    text += emit;
                    if (!emit.empty() && on_token) on_token(emit);
                    if (!ids.empty()) ids.pop_back();  // the cut token does not count as produced
                    stop_hit = true;
                    return false;  // ask the engine to stop
                }
            }

            text += chunk;
            if (on_token && !on_token(chunk)) {
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
    res.n_gen = res.tokens.size();
    if (stop_hit) {
        res.finish = Finish::kStopString;
    } else if (sum.finish == "stop") {
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
