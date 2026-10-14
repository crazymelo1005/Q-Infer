// NativeBackend: self-written backend implementing the Backend interface.
// Wraps NativeSession (per-inference session loop) and provides the same
// generate/tokenizer/chat_template contract as Engine, but using the new
// session loop architecture (Phase 1 of native engine).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "artifact/gguf_table.hpp"
#include "dense/token_embed.hpp"
#include "serve/facade.hpp"
#include "serve/native_session.hpp"
#include "text/tokenizer.hpp"

namespace qinfer::serve {

// Self-written backend: owns model, tokenizer, and session.
// Implements Backend interface for server integration.
class NativeBackend : public Backend {
public:
    NativeBackend() = default;
    ~NativeBackend() override;
    NativeBackend(const NativeBackend&) = delete;
    NativeBackend& operator=(const NativeBackend&) = delete;

    // Load model, tokenizer, chat template. Builds internal StepConfig.
    static bool open(const FacadeConfig& cfg, NativeBackend& out,
                     std::string& err);

    // Backend interface
    bool generate(const std::vector<std::int32_t>& prompt,
                  const GenOptions& opt,
                  const std::function<bool(const std::string&)>& on_token,
                  GenResult& res, std::string& err) override;
    const text::Tokenizer& tokenizer() const override { return tokenizer_; }
    const std::string& chat_template() const override { return chat_template_; }
    const std::string& model_name() const override { return model_name_; }
    const FacadeConfig& config() const override { return cfg_; }
    std::uint64_t kv_misses() const override;
    std::uint64_t kv_evictions() const override;
    std::uint64_t kv_bytes_moved() const override;

private:
    FacadeConfig cfg_;
    artifact::GgufFile gguf_;
    dense::TokenEmbed te_;
    text::Tokenizer tokenizer_;
    std::string chat_template_;
    std::string model_name_;
    std::int32_t eos_ = -1;
    NativeSession session_;

    void build_step_config();
};

}  // namespace qinfer::serve
