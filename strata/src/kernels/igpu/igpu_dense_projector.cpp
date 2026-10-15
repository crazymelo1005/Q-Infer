// iGPU dense projection accelerator implementation.
// See igpu_dense_projector.hpp for the design.
//
// This accelerates large GDN dense projections (attn_qkv, attn_gate, ssm_out)
// on Intel iGPU. P1 spike measured 3.9x faster than CPU for 10240x2560 matrices.
//
// Supported formats: Q2_0, Q8_0 weights x Q8_1 activation.
// Other formats (IQ4_XS, IQ3_S) require additional shaders.

#include "strata/kernels/igpu/igpu_dense_projector.hpp"
#include "strata/kernels/igpu/vulkan_runtime.hpp"

// Shader SPIR-V data
#include "strata/kernels/igpu/shaders/spv_data.hpp"

#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>

namespace strata::kernels::igpu {

namespace vk = strata::kernels::igpu::vk;

// ---- environment check ----

bool igpu_dense_enabled() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_IGPU_DENSE");
        return v != nullptr && v[0] == '1';
    }();
    return on;
}

// ---- push constants for the GEMV shader ----

struct DenseGemvPc {
    std::uint32_t n_rows;       // number of weight rows
    std::uint32_t n_blocks_w;   // Q2_0 blocks per weight row (n_in / 64)
    std::uint32_t n_blocks_a;   // Q8_1 blocks per activation (n_in / 32)
};

// ---- per-weight device buffers ----

struct DenseWeightBuf {
    void* d_weights = nullptr;  // interleaved quantized weights
    std::uint64_t bytes = 0;
    int n_in = 0;
    int n_out = 0;
    int quant_type = 0;  // GGML quantization type (42=Q2_0, 8=Q8_0, etc.)
};

// ---- global state ----

namespace {
    IgpuDenseProjector* g_projector = nullptr;
    std::once_flag g_init_flag;
    std::mutex g_mutex;

    // Map from host weight pointer to device buffers
    std::unordered_map<const void*, DenseWeightBuf> g_weight_bufs;
}

// ---- IgpuDenseProjector implementation ----

IgpuDenseProjector::~IgpuDenseProjector() {
    release();
}

bool IgpuDenseProjector::init(std::string& err) {
    if (ready_) return true;

    VulkanRuntime& rt = igpu_runtime();
    if (!rt.load(err) || !rt.init_device(err) || !rt.available()) {
        return false;
    }

    // GEMV descriptor set layout: 5 bindings
    //   0: weights (interleaved quantized: Q2_0 18 bytes/block, Q8_0 34 bytes/block)
    //   1: activation codes (SSBO)
    //   2: activation scales (SSBO)
    //   3: activation sums (SSBO)
    //   4: output (SSBO)
    {
        vk::DescriptorSetLayoutBinding bindings[5] = {};
        for (std::uint32_t i = 0; i < 5; ++i) {
            bindings[i].binding = i;
            bindings[i].descriptorType = vk::DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags = vk::SHADER_STAGE_COMPUTE;
        }
        sl_gemv_ = rt.make_set_layout(bindings, 5, err);
        if (sl_gemv_ == nullptr) return false;
        vk::DescriptorSetLayout sl = static_cast<vk::DescriptorSetLayout>(sl_gemv_);
        pl_gemv_ = rt.make_layout(&sl, 1, err);
        if (pl_gemv_ == nullptr) return false;
        
        // Create Q2_0 pipeline
        pipe_q2_0_ = rt.make_pipeline(
            shaders::kStrataDenseQ20GemvSpv,
            shaders::kStrataDenseQ20GemvSpvSize,
            static_cast<vk::PipelineLayout>(pl_gemv_), err);
        if (pipe_q2_0_ == nullptr) return false;
        
        // Create Q8_0 pipeline
        pipe_q8_0_ = rt.make_pipeline(
            shaders::kStrataDenseQ80GemvSpv,
            shaders::kStrataDenseQ80GemvSpvSize,
            static_cast<vk::PipelineLayout>(pl_gemv_), err);
        if (pipe_q8_0_ == nullptr) return false;

        // Create IQ4_XS pipeline
        pipe_iq4_xs_ = rt.make_pipeline(
            shaders::kStrataDenseIq4XsGemvSpv,
            shaders::kStrataDenseIq4XsGemvSpvSize,
            static_cast<vk::PipelineLayout>(pl_gemv_), err);
        if (pipe_iq4_xs_ == nullptr) return false;

        // Create IQ3_S pipeline
        pipe_iq3_s_ = rt.make_pipeline(
            shaders::kStrataDenseIq3sGemvSpv,
            shaders::kStrataDenseIq3sGemvSpvSize,
            static_cast<vk::PipelineLayout>(pl_gemv_), err);
        if (pipe_iq3_s_ == nullptr) return false;
    }

    ready_ = true;
    std::fprintf(stderr, "strata igpu dense: Vulkan projector initialized\n");
    return true;
}

bool IgpuDenseProjector::load(const std::string& name, const void* weights, uint64_t bytes,
                               int n_in, int n_out, int quant_type, std::string& err) {
    if (!ready_) {
        err = "iGPU dense projector not initialized";
        return false;
    }

    // Check if already loaded
    if (g_weight_bufs.count(weights) > 0) {
        return true;
    }

    VulkanRuntime& rt = igpu_runtime();

    // Determine block size and bytes per block based on quantization type
    int QK, block_bytes;
    const char* quant_name;
    if (quant_type == 42) {  // Q2_0
        QK = 64;
        block_bytes = 18;  // 16 codes + 2 scale
        quant_name = "Q2_0";
    } else if (quant_type == 8) {  // Q8_0
        QK = 32;
        block_bytes = 34;  // 32 codes + 2 scale
        quant_name = "Q8_0";
    } else if (quant_type == 23) {  // IQ4_XS
        QK = 256;
        block_bytes = 136;  // 2 d + 2 scales_h + 4 scales_l + 128 qs
        quant_name = "IQ4_XS";
    } else if (quant_type == 21) {  // IQ3_S
        QK = 256;
        block_bytes = 110;  // 2 d + 64 qs + 8 qh + 32 signs + 4 scales
        quant_name = "IQ3_S";
    } else {
        err = "iGPU dense: unsupported quantization type " + std::to_string(quant_type);
        return false;
    }

    if (n_in % QK != 0) {
        err = "iGPU dense: n_in must be multiple of " + std::to_string(QK) + " for " + quant_name;
        return false;
    }

    const int n_blocks = n_in / QK;
    const std::uint64_t total_bytes = (std::uint64_t)n_blocks * block_bytes * n_out;

    DenseWeightBuf buf;
    buf.n_in = n_in;
    buf.n_out = n_out;
    buf.bytes = total_bytes;
    buf.quant_type = quant_type;

    if (!rt.alloc_device(total_bytes, buf.d_weights, err)) {
        return false;
    }

    // Upload weights to device (already in interleaved format from GGUF)
    if (!rt.upload(weights, total_bytes, buf.d_weights, err)) {
        rt.free_buf(buf.d_weights, err);
        return false;
    }

    g_weight_bufs[weights] = buf;
    std::fprintf(stderr, "strata igpu dense: loaded %s %s (%dx%d, %.1f MiB)\n",
                 quant_name, name.c_str(), n_out, n_in,
                 (double)total_bytes / 1048576.0);
    return true;
}

bool IgpuDenseProjector::run(const void* weights, const void* x_q8_1, float* y,
                              int n_in, int n_out, std::string& err) {
    if (!ready_) {
        err = "iGPU dense projector not initialized";
        return false;
    }

    auto it = g_weight_bufs.find(weights);
    if (it == g_weight_bufs.end()) {
        err = "iGPU dense: weights not loaded";
        return false;
    }

    const DenseWeightBuf& wbuf = it->second;
    if (wbuf.n_in != n_in || wbuf.n_out != n_out) {
        err = "iGPU dense: dimension mismatch";
        return false;
    }

    VulkanRuntime& rt = igpu_runtime();

    // Select pipeline and block size based on weight quantization type
    void* pipeline = nullptr;
    int QK_W;
    if (wbuf.quant_type == 42) {  // Q2_0: 64 elements per block
        pipeline = pipe_q2_0_;
        QK_W = 64;
    } else if (wbuf.quant_type == 8) {  // Q8_0: 32 elements per block
        pipeline = pipe_q8_0_;
        QK_W = 32;
    } else if (wbuf.quant_type == 23) {  // IQ4_XS: 256 elements per super-block
        pipeline = pipe_iq4_xs_;
        QK_W = 256;
    } else if (wbuf.quant_type == 21) {  // IQ3_S: 256 elements per super-block
        pipeline = pipe_iq3_s_;
        QK_W = 256;
    } else {
        err = "iGPU dense: unsupported quant type in run: " + std::to_string(wbuf.quant_type);
        return false;
    }
    const int QK_A = 32;  // Q8_1: 32 elements per block
    const int n_blocks_w = n_in / QK_W;
    const int n_blocks_a = n_in / QK_A;

    // Parse Q8_1 activation buffer layout:
    //   codes:  n_blocks_a * 32 bytes (int8 codes)
    //   scales: n_blocks_a * 2 bytes (fp16 scales)
    //   sums:   n_blocks_a * 2 bytes (fp16 sums)
    const std::uint64_t act_codes_bytes = (std::uint64_t)n_blocks_a * 32;
    const std::uint64_t act_scales_bytes = (std::uint64_t)n_blocks_a * 2;
    const std::uint64_t act_sums_bytes = (std::uint64_t)n_blocks_a * 2;

    const std::uint8_t* q8_1_bytes = static_cast<const std::uint8_t*>(x_q8_1);
    const std::uint8_t* act_codes_ptr = q8_1_bytes;
    const std::uint8_t* act_scales_ptr = q8_1_bytes + act_codes_bytes;
    const std::uint8_t* act_sums_ptr = q8_1_bytes + act_codes_bytes + act_scales_bytes;

    // Allocate device buffers for activation
    void* d_act_codes = nullptr;
    void* d_act_scales = nullptr;
    void* d_act_sums = nullptr;
    void* d_out = nullptr;

    if (!rt.alloc_device(act_codes_bytes, d_act_codes, err) ||
        !rt.alloc_device(act_scales_bytes, d_act_scales, err) ||
        !rt.alloc_device(act_sums_bytes, d_act_sums, err) ||
        !rt.alloc_device((std::uint64_t)n_out * sizeof(float), d_out, err)) {
        return false;
    }

    // Upload activation to device
    if (!rt.upload(act_codes_ptr, act_codes_bytes, d_act_codes, err) ||
        !rt.upload(act_scales_ptr, act_scales_bytes, d_act_scales, err) ||
        !rt.upload(act_sums_ptr, act_sums_bytes, d_act_sums, err)) {
        rt.free_buf(d_act_codes, err);
        rt.free_buf(d_act_scales, err);
        rt.free_buf(d_act_sums, err);
        rt.free_buf(d_out, err);
        return false;
    }

    // Create descriptor set
    rt.reset_desc_pool();
    vk::DescriptorSet set = rt.make_set(static_cast<vk::DescriptorSetLayout>(sl_gemv_), nullptr, 0, err);
    if (set == nullptr) {
        rt.free_buf(d_act_codes, err);
        rt.free_buf(d_act_scales, err);
        rt.free_buf(d_act_sums, err);
        rt.free_buf(d_out, err);
        return false;
    }

    vk::Buffer bufs[5] = {
        static_cast<vk::Buffer>(wbuf.d_weights),
        static_cast<vk::Buffer>(d_act_codes),
        static_cast<vk::Buffer>(d_act_scales),
        static_cast<vk::Buffer>(d_act_sums),
        static_cast<vk::Buffer>(d_out)
    };
    rt.update_set(set, bufs, 5);

    // Dispatch
    DenseGemvPc pc{
        static_cast<std::uint32_t>(n_out),
        static_cast<std::uint32_t>(n_blocks_w),
        static_cast<std::uint32_t>(n_blocks_a)
    };

    std::string dispatch_err;
    if (!rt.dispatch(static_cast<vk::Pipeline>(pipeline),
                     static_cast<vk::PipelineLayout>(pl_gemv_),
                     set, &pc, sizeof(pc),
                     (n_out + 63) / 64, 1, 1, dispatch_err)) {
        err = "iGPU dense dispatch: " + dispatch_err;
        rt.free_buf(d_act_codes, err);
        rt.free_buf(d_act_scales, err);
        rt.free_buf(d_act_sums, err);
        rt.free_buf(d_out, err);
        return false;
    }

    // Download result
    if (!rt.download(d_out, (std::uint64_t)n_out * sizeof(float), y, err)) {
        rt.free_buf(d_act_codes, err);
        rt.free_buf(d_act_scales, err);
        rt.free_buf(d_act_sums, err);
        rt.free_buf(d_out, err);
        return false;
    }

    // Free activation buffers
    rt.free_buf(d_act_codes, err);
    rt.free_buf(d_act_scales, err);
    rt.free_buf(d_act_sums, err);
    rt.free_buf(d_out, err);

    return true;
}

void IgpuDenseProjector::release() {
    if (!ready_) return;

    VulkanRuntime& rt = igpu_runtime();
    std::string ignored;

    for (auto& [ptr, buf] : g_weight_bufs) {
        if (buf.d_weights) rt.free_buf(buf.d_weights, ignored);
    }
    g_weight_bufs.clear();

    ready_ = false;
}

// ---- global accessor ----

IgpuDenseProjector* igpu_dense_projector() {
    if (!igpu_dense_enabled()) return nullptr;

    std::call_once(g_init_flag, []() {
        std::lock_guard<std::mutex> lock(g_mutex);
        if (g_projector == nullptr) {
            g_projector = new IgpuDenseProjector();
            std::string err;
            if (!g_projector->init(err)) {
                std::fprintf(stderr, "strata igpu dense: init failed: %s\n", err.c_str());
                delete g_projector;
                g_projector = nullptr;
            }
        }
    });

    return g_projector;
}

}  // namespace strata::kernels::igpu
