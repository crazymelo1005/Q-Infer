// Vendored-engine backend (P2, ADR-017): drives the Strata engine through EngineDriver but
// presents the same Backend interface as the in-house Engine, so the service layer is agnostic.
//
// Token ids cross the process boundary; this class owns the tokenizer and is the only place
// that turns ids into text. The engine keeps its own KV, so kv_*() report 0 here.
#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "engine/engine_driver.hpp"
#include "serve/facade.hpp"
#include "text/tokenizer.hpp"

namespace qinfer::serve {

// The tokenizer and chat template are built by the caller (from the same GGUF) and moved in, so
// a unit test can hand in a tiny vocabulary and needs no weights.
struct StrataBackendConfig {
    FacadeConfig facade;
    engine::DriverConfig driver;
    text::Tokenizer tokenizer;   // moved in
    std::string chat_template;   // moved in
    std::string model_name;      // moved in
};

class StrataBackend : public Backend {
public:
    // Spawns the engine and completes the READY handshake; max_context comes from READY.
    static bool open(StrataBackendConfig cfg, StrataBackend& out, std::string& err);

    bool generate(const std::vector<std::int32_t>& prompt, const GenOptions& opt,
                  const std::function<bool(const std::string&)>& on_token, GenResult& res,
                  std::string& err) override;

    const text::Tokenizer& tokenizer() const override { return tokenizer_; }
    const std::string& chat_template() const override { return chat_template_; }
    const std::string& model_name() const override { return model_name_; }
    const FacadeConfig& config() const override { return cfg_; }
    std::uint64_t kv_misses() const override { return 0; }
    std::uint64_t kv_evictions() const override { return 0; }
    std::uint64_t kv_bytes_moved() const override { return 0; }

private:
    engine::EngineDriver driver_;
    text::Tokenizer tokenizer_;
    std::string chat_template_, model_name_;
    FacadeConfig cfg_{};
};

}  // namespace qinfer::serve
