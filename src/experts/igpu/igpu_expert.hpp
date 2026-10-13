// IgpuExpertDevice (stage C task 4): one batch of experts on the iGPU.
//
// Per expert: gate = W_gate·x (Q2_0 GEMV) -> up = W_up·x -> h = silu(gate)·up
// quantized to Q8_0 (fused shader) -> out = W_down·h (Q2_0 GEMV), written to
// out + e*hidden exactly like CpuExpertWorker. Activation quantization stays on
// the host (quantize_input), the same convention as the CUDA expert path.
//
// First slice: all three matrices must be Q2_0 (the shader set we have); other
// formats are an explicit error, never a silent fallback. The capability table
// grows with the shader set (task 6).
#pragma once

#include <cstdint>
#include <string>

#include "experts/expert_formats.hpp"
#include "experts/ffn.hpp"
#include "experts/igpu/vulkan_abi.hpp"

namespace qinfer::experts::igpu {

struct IgpuExpertDevice {
    // True when the Vulkan runtime found the iGPU; err explains otherwise.
    bool init(std::string& err);
    bool ready() const { return ready_; }

    // Runs `n` experts' FFN for one layer; `ws` is the per-expert weight set,
    // `x` the hidden activation (host), out length n*hidden (host).
    bool run_batch(const LayerSpec& spec, const ExpertWeights* ws, int n, const float* x,
                   std::uint64_t hidden, float* out, std::string& err);

private:
    bool ready_ = false;
    std::string reason_;
    // Pipelines created once in init (Mesa compiles the SPIR-V at creation).
    vk::DescriptorSetLayout sl_ = nullptr;
    vk::PipelineLayout pl_ = nullptr;
    vk::Pipeline gemv_ = nullptr;
    vk::Pipeline swiglu_ = nullptr;
};

}  // namespace qinfer::experts::igpu