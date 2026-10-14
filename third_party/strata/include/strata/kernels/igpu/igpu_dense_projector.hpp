// iGPU dense projection accelerator for large GDN matrices.
//
// DESIGN (2026-10-14):
// The GDN layer has three large dense projections that dominate CPU time:
//   - attn_qkv:  [10240, 2560]  ~25 MiB per tensor
//   - attn_gate: [10240, 2560]  ~25 MiB per tensor
//   - ssm_out:   [2560, 10240]  ~25 MiB per tensor
//
// P1 spike measured iGPU 3.9x faster than CPU for these (CPU 6.4ms → iGPU 1.6ms).
// The iGPU excels at large matrices (10240 rows fills 4 Xe cores) but is slow at
// small matrices (640 rows only partially occupies).
//
// INTEGRATION:
// - NativeDense uploads weights to both GPU (CUDA) and iGPU (Vulkan) when STRATA_IGPU_DENSE=1
// - gemv_quantized() checks w.igpu_data and routes to iGPU instead of native_mmvq()
// - Activation quantization (Q8_1) happens on CPU, then uploads to iGPU
// - Result downloads from iGPU to CPU output buffer
//
// THREAD SAFETY:
// Single-threaded per inference session. The Vulkan runtime is not thread-safe.
#pragma once

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace strata::kernels::igpu {

// Forward declaration
struct VulkanRuntime;

/// Convert fp32 to fp16 (IEEE 754 half-precision).
/// Matches the conversion in kernels/fp16.hpp.
inline std::uint16_t f32_to_f16(float f) {
    std::uint32_t x;
    std::memcpy(&x, &f, 4);
    const std::uint32_t sign = (x >> 16) & 0x8000u;
    int32_t e = (int32_t)((x >> 23) & 0xff) - 127 + 15;
    std::uint32_t m = x & 0x7fffffu;
    if (e <= 0) {
        if (e < -10) return (std::uint16_t)sign;
        m |= 0x800000u;
        const std::uint32_t shift = (std::uint32_t)(14 - e);
        std::uint32_t h = m >> shift;
        const std::uint32_t rem = m & ((1u << shift) - 1), half = 1u << (shift - 1);
        if (rem > half || (rem == half && (h & 1u))) ++h;
        return (std::uint16_t)(sign | h);
    }
    if (e >= 31) return (std::uint16_t)(sign | 0x7c00u);
    std::uint32_t h = ((std::uint32_t)e << 10) | (m >> 13);
    const std::uint32_t rem = m & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) ++h;
    return (std::uint16_t)(sign | h);
}

/// Q8_1 buffer size for n_in elements (matches native_mmvq layout).
inline std::size_t native_q8_1_bytes(int n_in) {
    // Q8_1: 36 bytes per 32-element block (32 codes + 2 scale + 2 sum + 2 padding)
    // Actually it's 32 + 2 + 2 = 36 bytes, but aligned to 4 bytes per block
    const int n_blocks = (n_in + 31) / 32;
    return (std::size_t)n_blocks * 36;
}

/// Dense projection accelerator for large GDN matrices on Intel iGPU.
///
/// Usage:
///   IgpuDenseProjector proj;
///   if (!proj.init(err)) return false;
///   if (!proj.load("blk.0.attn_qkv.weight", weights_ptr, bytes, n_in, n_out, err)) return false;
///   if (!proj.run(weights_ptr, x_q8_1, y, n_in, n_out, err)) return false;
class IgpuDenseProjector {
public:
    IgpuDenseProjector() = default;
    ~IgpuDenseProjector();

    IgpuDenseProjector(const IgpuDenseProjector&) = delete;
    IgpuDenseProjector& operator=(const IgpuDenseProjector&) = delete;

    /// Initialize Vulkan runtime and create pipelines.
    /// Returns false if Vulkan is unavailable.
    bool init(std::string& err);

    /// Check if the projector is ready.
    bool available() const { return ready_; }

    /// Upload a weight matrix to iGPU device memory.
    /// The weights remain in GGUF quantized format (e.g., Q2_0, Q8_0, IQ4_XS).
    /// quant_type: GGML quantization type (e.g., 42=Q2_0, 8=Q8_0, 23=IQ4_XS)
    /// Returns false on allocation or upload failure.
    bool load(const std::string& name, const void* weights, uint64_t bytes,
              int n_in, int n_out, int quant_type, std::string& err);

    /// Execute GEMV: y = W @ x_q8_1
    /// - weights: pointer to the weight matrix (must have been loaded)
    /// - x_q8_1: quantized activation (Q8_1 format, n_in elements)
    /// - y: output buffer (n_out floats)
    /// - n_in, n_out: matrix dimensions
    /// Returns false on dispatch failure.
    bool run(const void* weights, const void* x_q8_1, float* y,
             int n_in, int n_out, std::string& err);

    /// Free all device resources.
    void release();

private:
    bool ready_ = false;

    // Vulkan handles (opaque pointers)
    void* sl_gemv_ = nullptr;      // VkDescriptorSetLayout
    void* pl_gemv_ = nullptr;      // VkPipelineLayout
    void* pipe_q2_0_ = nullptr;      // VkPipeline for Q2_0 weights
    void* pipe_q8_0_ = nullptr;      // VkPipeline for Q8_0 weights
    void* pipe_iq4_xs_ = nullptr;    // VkPipeline for IQ4_XS weights
    void* pipe_iq3_s_ = nullptr;     // VkPipeline for IQ3_S weights

    // Device memory for weight buffers (keyed by weight pointer)
    struct WeightBuf {
        void* d_codes = nullptr;   // device codes buffer
        void* d_scales = nullptr;  // device scales buffer
        uint64_t codes_bytes = 0;
        uint64_t scales_bytes = 0;
    };
    std::vector<WeightBuf> weight_bufs_;
};

/// Global iGPU dense projector instance (lazy-init, thread-safe).
/// Returns nullptr if STRATA_IGPU_DENSE != 1 or init failed.
IgpuDenseProjector* igpu_dense_projector();

/// Check if iGPU dense acceleration is enabled (STRATA_IGPU_DENSE=1).
bool igpu_dense_enabled();

}  // namespace strata::kernels::igpu
