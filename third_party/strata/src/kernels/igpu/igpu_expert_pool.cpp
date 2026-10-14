// IgpuExpertPool + hybrid dispatcher implementation.
// See igpu_expert_pool.hpp for the design.
//
// Strata integration (Task 5): the activation quantization calls
// strata::kernels::cpu::act_quant_q8_1 directly (wrapped into PoolActQ).
// The CPU fallback function pointer is set from generate.cpp.
#include "strata/kernels/igpu/igpu_expert_pool.hpp"

#include "strata/kernels/igpu/shaders/spv_data.hpp"
#include "strata/kernels/igpu/strata_pack_adapter.hpp"
#include "strata/kernels/igpu/vulkan_runtime.hpp"

// Strata headers for the real act_quant_q8_1.
#include "strata/kernels/cpu/expert.hpp"

#include <cmath>
#include <cstring>
#include <mutex>
#include <vector>

namespace strata::kernels::igpu {

namespace vk = strata::kernels::igpu::vk;

// ---- geometry constants (matching Strata's expert.hpp) ----
namespace {

constexpr int H  = 2560;   // n_embd
constexpr int FF = 640;    // expert intermediate width
constexpr int QK = 64;     // Q2_0 weights per fp16 scale
constexpr int QKA = 32;    // activation elements per chunk
constexpr int MAXC = H / QKA;  // 80 activation chunks

// ---- Strata act_quant_q8_1 wrapper ----

// Wraps strata::kernels::cpu::act_quant_q8_1 into the PoolActQ contract.
// PoolActQ and strata::kernels::cpu::ActQ share the first five fields
// (q, scale, sum, hx, nchunks) but differ in alignment (alignas(64) vs
// alignas(32)) and ActQ has additional bit-plane fields. We quantize into
// a temporary ActQ and copy the fields PoolActQ needs.
void strata_act_quant(const float* x, int n, PoolActQ& a) {
    strata::kernels::cpu::ActQ tmp{};
    strata::kernels::cpu::act_quant_q8_1(x, n, tmp);
    std::memcpy(a.q, tmp.q, H * sizeof(int8_t));
    std::memcpy(a.scale, tmp.scale, MAXC * sizeof(float));
    std::memcpy(a.sum, tmp.sum, MAXC * sizeof(int32_t));
    std::memcpy(a.hx, tmp.hx, MAXC * sizeof(float));
    a.nchunks = tmp.nchunks;
}

// Global quantization function pointer (defaults to Strata's real implementation).
ActQuantFn g_act_quant = &strata_act_quant;

// Global CPU fallback function pointer (set from generate.cpp).
CpuDispatchFn g_cpu_dispatch = nullptr;

// ---- buffer RAII ----

struct BatchBufs {
    VulkanRuntime* rt = nullptr;
    std::vector<void*> bufs;
    ~BatchBufs() {
        if (rt != nullptr) {
            std::string ignored;
            for (void* b : bufs) rt->free_buf(b, ignored);
        }
    }
    bool make(VulkanRuntime& r, std::uint64_t bytes, void*& out, std::string& err) {
        rt = &r;
        if (!r.alloc_device(bytes, out, err)) return false;
        bufs.push_back(out);
        return true;
    }
};

// Push constants for the GEMV shader.
struct GemvPc {
    std::uint32_t n_rows;
    std::uint32_t n_blocks;
};

// Push constants for the SwiGLU shader.
struct SwigluPc {
    std::uint32_t n_elems;
    std::uint32_t n_chunks;
};

// ActQ buffer size in bytes (matches the shader's layout: 880 uints = 3520 bytes).
constexpr std::uint64_t kActQBufBytes = 880 * sizeof(std::uint32_t);

// SwiGLU output buffer size (same layout as ActQ, 220 uints = 880 bytes for FF=640).
constexpr std::uint64_t kSwigluOutBytes = 220 * sizeof(std::uint32_t);

}  // namespace

// ---- public API ----

void set_act_quant_fn(ActQuantFn fn) {
    g_act_quant = fn;
}

ActQuantFn act_quant_fn() {
    return g_act_quant;
}

void set_cpu_dispatch_fn(CpuDispatchFn fn) {
    g_cpu_dispatch = fn;
}

CpuDispatchFn cpu_dispatch_fn() {
    return g_cpu_dispatch;
}

bool IgpuExpertPool::init(std::string& err) {
    err.clear();
    VulkanRuntime& rt = igpu_runtime();
    if (!rt.load(err) || !rt.init_device(err) || !rt.available()) {
        reason_ = rt.reason().empty() ? err : rt.reason();
        return false;
    }

    // GEMV descriptor set layout: 4 bindings (wcodes, wscales, act, out)
    {
        vk::DescriptorSetLayoutBinding bindings[4] = {};
        for (std::uint32_t i = 0; i < 4; ++i) {
            bindings[i].binding = i;
            bindings[i].descriptorType = vk::DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags = vk::SHADER_STAGE_COMPUTE;
        }
        sl_gemv_ = rt.make_set_layout(bindings, 4, err);
        if (sl_gemv_ == nullptr) return false;
        pl_gemv_ = rt.make_layout(&sl_gemv_, 1, err);
        if (pl_gemv_ == nullptr) return false;
        pipe_gemv_ = rt.make_pipeline(shaders::kStrataQ20GemvSpv,
                                       shaders::kStrataQ20GemvSpvSize, pl_gemv_, err);
        if (pipe_gemv_ == nullptr) return false;
    }

    // SwiGLU descriptor set layout: 3 bindings (gate, up, out)
    {
        vk::DescriptorSetLayoutBinding bindings[3] = {};
        for (std::uint32_t i = 0; i < 3; ++i) {
            bindings[i].binding = i;
            bindings[i].descriptorType = vk::DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags = vk::SHADER_STAGE_COMPUTE;
        }
        sl_swiglu_ = rt.make_set_layout(bindings, 3, err);
        if (sl_swiglu_ == nullptr) return false;
        pl_swiglu_ = rt.make_layout(&sl_swiglu_, 1, err);
        if (pl_swiglu_ == nullptr) return false;
        pipe_swiglu_ = rt.make_pipeline(shaders::kStrataSwigluQuantSpv,
                                         shaders::kStrataSwigluQuantSpvSize, pl_swiglu_, err);
        if (pipe_swiglu_ == nullptr) return false;
    }

    ready_ = true;
    return true;
}

void IgpuExpertPool::run(void* jobs_ptr, int n) {
    if (!ready_ || n <= 0) return;

    auto* jobs = static_cast<PoolExpertJob*>(jobs_ptr);
    VulkanRuntime& rt = igpu_runtime();
    std::string err;

    // Allocate device buffers:
    //   d_act: activation buffer (3520 bytes, reused across experts)
    //   d_gate: gate projection output (FF=640 floats)
    //   d_up: up projection output (FF=640 floats)
    //   d_swiglu_out: SwiGLU output in ActQ format (880 bytes)
    //   d_down: down projection output (H=2560 floats)
    //   d_wcodes: weight codes buffer
    //   d_wscales: weight scales buffer
    void* d_act = nullptr;
    void* d_gate = nullptr;
    void* d_up = nullptr;
    void* d_swiglu_out = nullptr;
    void* d_down = nullptr;
    void* d_wcodes = nullptr;
    void* d_wscales = nullptr;

    BatchBufs bufs;
    // Weight buffers sized for gate/up (larger): FF rows * ROW_GU codes = 640*640 = 409600 bytes
    // and FF rows * SC_GU*2 scales = 640*80 = 51200 bytes
    constexpr std::uint64_t kMaxCodesBytes = 2 * FF * 640;  // gate+up codes region = 1280000
    constexpr std::uint64_t kMaxScalesBytes = 2 * FF * 40 * 2;  // gate+up scales = 102400

    if (!bufs.make(rt, kActQBufBytes, d_act, err) ||
        !bufs.make(rt, FF * sizeof(float), d_gate, err) ||
        !bufs.make(rt, FF * sizeof(float), d_up, err) ||
        !bufs.make(rt, kSwigluOutBytes, d_swiglu_out, err) ||
        !bufs.make(rt, H * sizeof(float), d_down, err) ||
        !bufs.make(rt, kMaxCodesBytes, d_wcodes, err) ||
        !bufs.make(rt, kMaxScalesBytes, d_wscales, err)) {
        return;
    }

    // Descriptor sets: one for GEMV, one for SwiGLU (reused across dispatches).
    vk::DescriptorSet set_gemv = rt.make_set(sl_gemv_, nullptr, 0, err);
    if (set_gemv == nullptr) return;
    vk::DescriptorSet set_swiglu = rt.make_set(sl_swiglu_, nullptr, 0, err);
    if (set_swiglu == nullptr) return;

    // Process each expert
    for (int e = 0; e < n; ++e) {
        const auto& job = jobs[e];
        if (job.blob == nullptr || job.act == nullptr || job.out == nullptr) continue;

        // Upload activation to device
        if (!rt.upload(job.act, kActQBufBytes, d_act, err)) return;

        // Gate GEMV: FF=640 rows, H/QK=40 blocks per row
        {
            StrataPackRow gate_row = gu_row(job.blob, 0, false);
            std::uint64_t gate_codes_bytes = FF * 640;
            std::uint64_t gate_scales_bytes = FF * 40 * 2;
            if (!rt.upload(gate_row.codes, gate_codes_bytes, d_wcodes, err)) return;
            if (!rt.upload(gate_row.scales, gate_scales_bytes, d_wscales, err)) return;

            vk::Buffer b[4] = {
                static_cast<vk::Buffer>(d_wcodes),
                static_cast<vk::Buffer>(d_wscales),
                static_cast<vk::Buffer>(d_act),
                static_cast<vk::Buffer>(d_gate)
            };
            rt.update_set(set_gemv, b, 4);
            GemvPc pc{FF, H / QK};
            std::string e2;
            if (!rt.dispatch(pipe_gemv_, pl_gemv_, set_gemv, &pc, sizeof(pc),
                            (FF + 63) / 64, 1, 1, e2)) {
                err = std::string("pool GEMV gate: ") + e2;
                return;
            }
        }

        // Up GEMV: FF=640 rows, H/QK=40 blocks per row
        {
            StrataPackRow up_row = gu_row(job.blob, 0, true);
            std::uint64_t up_codes_bytes = FF * 640;
            std::uint64_t up_scales_bytes = FF * 40 * 2;
            if (!rt.upload(up_row.codes, up_codes_bytes, d_wcodes, err)) return;
            if (!rt.upload(up_row.scales, up_scales_bytes, d_wscales, err)) return;

            vk::Buffer b[4] = {
                static_cast<vk::Buffer>(d_wcodes),
                static_cast<vk::Buffer>(d_wscales),
                static_cast<vk::Buffer>(d_act),
                static_cast<vk::Buffer>(d_up)
            };
            rt.update_set(set_gemv, b, 4);
            GemvPc pc{FF, H / QK};
            std::string e2;
            if (!rt.dispatch(pipe_gemv_, pl_gemv_, set_gemv, &pc, sizeof(pc),
                            (FF + 63) / 64, 1, 1, e2)) {
                err = std::string("pool GEMV up: ") + e2;
                return;
            }
        }

        // SwiGLU + ActQ quantization: gate, up -> intermediate ActQ
        {
            vk::Buffer b[3] = {
                static_cast<vk::Buffer>(d_gate),
                static_cast<vk::Buffer>(d_up),
                static_cast<vk::Buffer>(d_swiglu_out)
            };
            rt.update_set(set_swiglu, b, 3);
            SwigluPc pc{FF, FF / QKA};
            std::string e2;
            if (!rt.dispatch(pipe_swiglu_, pl_swiglu_, set_swiglu, &pc, sizeof(pc),
                            (FF / QKA + 63) / 64, 1, 1, e2)) {
                err = std::string("pool SwiGLU: ") + e2;
                return;
            }
        }

        // Down GEMV: H=2560 rows, FF/QK=10 blocks per row
        {
            StrataPackRow down_r = down_row(job.blob, 0);
            std::uint64_t down_codes_bytes = H * 160;
            std::uint64_t down_scales_bytes = H * 10 * 2;
            if (!rt.upload(down_r.codes, down_codes_bytes, d_wcodes, err)) return;
            if (!rt.upload(down_r.scales, down_scales_bytes, d_wscales, err)) return;

            vk::Buffer b[4] = {
                static_cast<vk::Buffer>(d_wcodes),
                static_cast<vk::Buffer>(d_wscales),
                static_cast<vk::Buffer>(d_swiglu_out),
                static_cast<vk::Buffer>(d_down)
            };
            rt.update_set(set_gemv, b, 4);
            GemvPc pc{H, FF / QK};
            std::string e2;
            if (!rt.dispatch(pipe_gemv_, pl_gemv_, set_gemv, &pc, sizeof(pc),
                            (H + 63) / 64, 1, 1, e2)) {
                err = std::string("pool GEMV down: ") + e2;
                return;
            }
        }

        // Download result
        if (!rt.download(d_down, H * sizeof(float), job.out, err)) return;
    }
}

// ---- hybrid dispatcher ----

void hybrid_expert_dispatch(void* user, const float* x_f, const int32_t* ids,
                            const float* weights, int64_t n_embd, int64_t k, float* out) {
    // Lazy-init the pool (once, thread-safe).
    static IgpuExpertPool pool;
    static std::once_flag init_flag;
    std::call_once(init_flag, []() {
        std::string err;
        pool.init(err);
    });

    // Route: large batches to iGPU, small to CPU.
    if (pool.available() && k > 4 && user != nullptr) {
        // Quantize the activation once for all experts in this batch.
        PoolActQ act;
        g_act_quant(x_f, static_cast<int>(n_embd), act);

        // Build job array from PoolFn arguments.
        // The user pointer convention: cast to a blob array pointer.
        // In the Strata integration, the blobs come from ExpertSource via
        // the Drive wrapper in generate.cpp.
        auto** blobs = static_cast<const uint8_t**>(user);
        std::vector<PoolExpertJob> jobs(static_cast<std::size_t>(k));
        for (int64_t i = 0; i < k; ++i) {
            jobs[static_cast<std::size_t>(i)].blob = blobs[ids[i]];
            jobs[static_cast<std::size_t>(i)].act = &act;
            jobs[static_cast<std::size_t>(i)].out = out + i * n_embd;
            jobs[static_cast<std::size_t>(i)].weight = weights[i];
            jobs[static_cast<std::size_t>(i)].slot = static_cast<int>(i);
        }
        pool.run(jobs.data(), static_cast<int>(k));
    } else {
        // CPU fallback.
        if (g_cpu_dispatch != nullptr) {
            g_cpu_dispatch(user, x_f, ids, weights, n_embd, k, out);
        }
        // If no CPU fallback is set, this is a no-op.
    }
}

}  // namespace strata::kernels::igpu
