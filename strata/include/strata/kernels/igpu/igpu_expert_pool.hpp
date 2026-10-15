// IgpuExpertPool + hybrid dispatcher for the Strata backend.
//
// The pool processes batches of experts using Vulkan compute shaders:
//   - strata_q2_0_gemv: planar Q2_0 weight x ActQ activation GEMV
//   - strata_swiglu_quant: fused silu(gate)*up + ActQ quantization
//
// The hybrid dispatcher routes large batches (k > 4) to iGPU and small
// batches to CPU. It matches Strata's PoolFn signature so it can replace
// expert_pool_dispatch at the call site.
//
// Strata integration (Task 5): namespaces migrated from qinfer::experts::igpu
// to strata::kernels::igpu. The activation quantization calls
// strata::kernels::cpu::act_quant_q8_1 directly. The CPU fallback is wired
// to strata::core::expert_pool_dispatch from generate.cpp.
#pragma once

#include <cstdint>
#include <string>

#include "strata/kernels/igpu/vulkan_abi.hpp"

namespace strata::kernels::igpu {

// ActQ activation format (matches strata::kernels::cpu::ActQ binary layout).
// The first five fields are read by the GEMV shader; the remaining fields
// (bit-plane image) are unused by the iGPU path.
struct PoolActQ {
    alignas(64) int8_t q[2560];
    float scale[80];
    int32_t sum[80];
    float hx[80];
    int nchunks;
};

// One expert job (matches strata::kernels::cpu::ExpertJob prefix layout).
// The pool only reads these five fields; the Strata struct has additional
// fields (weight, slot) that are ignored by the iGPU path.
struct PoolExpertJob {
    const uint8_t* blob;     // one 1,382,400-byte expert in planar layout
    const PoolActQ* act;     // the layer's quantized activation (shared)
    float* out;              // H=2560 floats, written by the pool
    float weight;            // router weight (applied by the caller, not here)
    int slot;                // job index for diagnostics
};

// Quantization function type: float[H] -> PoolActQ.
// Default is strata::kernels::cpu::act_quant_q8_1 (wrapped).
using ActQuantFn = void (*)(const float* x, int n, PoolActQ& a);

// Set the activation quantization function. Default wraps Strata's act_quant_q8_1.
void set_act_quant_fn(ActQuantFn fn);

// Get the current quantization function.
ActQuantFn act_quant_fn();

class IgpuExpertPool {
public:
    // Load Vulkan, create pipelines for both Strata shaders.
    // Returns true on success; err describes the failure otherwise.
    bool init(std::string& err);

    // True when init() succeeded and the pool is ready to run.
    bool available() const { return ready_; }

    // Run n experts. `jobs` points to an array of PoolExpertJob (or equivalently
    // strata::kernels::cpu::ExpertJob, since the prefix layout matches).
    // For each expert: quantize activation, run gate/up/down GEMVs with
    // SwiGLU fusion in between, download results.
    // Returns true on success; on failure, returns false and writes to `err`.
    bool run(void* jobs, int n, std::string& err);

private:
    bool ready_ = false;
    std::string reason_;

    // Pipelines created once in init().
    vk::DescriptorSetLayout sl_gemv_ = nullptr;   // 4 bindings: wcodes, wscales, act, out
    vk::DescriptorSetLayout sl_swiglu_ = nullptr;  // 3 bindings: gate, up, out
    vk::PipelineLayout pl_gemv_ = nullptr;
    vk::PipelineLayout pl_swiglu_ = nullptr;
    vk::Pipeline pipe_gemv_ = nullptr;
    vk::Pipeline pipe_swiglu_ = nullptr;
};

// CPU fallback function type. Matches PoolFn signature.
// Set from generate.cpp to a wrapper around drive_pool (which extracts
// ExpertDispatch from Drive and calls strata::core::expert_pool_dispatch).
using CpuDispatchFn = void (*)(void* user, const float* x_f, const int32_t* ids,
                               const float* weights, int64_t n_embd, int64_t k, float* out);

// Set the CPU fallback function for the hybrid dispatcher.
// Default is nullptr (calls abort if invoked with no fallback set).
// generate.cpp sets this to a drive_pool wrapper when STRATA_IGPU_POOL=1.
void set_cpu_dispatch_fn(CpuDispatchFn fn);

// Get the current CPU fallback function.
CpuDispatchFn cpu_dispatch_fn();

// Hybrid dispatcher: matches PoolFn signature.
// If iGPU is available and k > 4, use iGPU; else call CPU fallback.
void hybrid_expert_dispatch(void* user, const float* x_f, const int32_t* ids,
                            const float* weights, int64_t n_embd, int64_t k, float* out);

// CPU fallback function type for multi-token path. Matches PoolMultiFn signature.
using CpuDispatchMultiFn = void (*)(void* user, const float* x_f, const int32_t* ids,
                                    int64_t n_tok, int64_t k, float* out, int64_t layer);

// Set the CPU fallback function for the multi-token hybrid dispatcher.
void set_cpu_dispatch_multi_fn(CpuDispatchMultiFn fn);

// Get the current CPU fallback function for multi-token path.
CpuDispatchMultiFn cpu_dispatch_multi_fn();

// Multi-token hybrid dispatcher: matches PoolMultiFn signature.
// Iterates over n_tok tokens, calling hybrid_expert_dispatch for each.
// If iGPU is available and k > 4, use iGPU; else call CPU fallback.
void hybrid_expert_dispatch_multi(void* user, const float* x_f, const int32_t* ids,
                                  int64_t n_tok, int64_t k, float* out, int64_t layer);

}  // namespace strata::kernels::igpu
