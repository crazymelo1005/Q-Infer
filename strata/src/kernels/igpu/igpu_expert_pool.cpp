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

// ExpertDispatch: the user pointer passed by igpu_drive_wrapper in generate.cpp.
#include "strata/core/expert_source.hpp"

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
    // Use the AVX-2 variant — act_quant_q8_1 has hardcoded AVX-512 intrinsics
    // which crash on CPUs without AVX-512 (like our Intel 285K).
    // The CPU pool path uses act_quant_any which dispatches at runtime;
    // we call the AVX-2 variant directly since we know this CPU has AVX-2.
    strata::kernels::cpu::ActQ tmp{};
    strata::kernels::cpu::act_quant_q8_1_avx2(x, n, tmp);
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

}  // close anonymous namespace

// Multi-token CPU fallback (must be outside anonymous namespace for external linkage).
CpuDispatchMultiFn g_cpu_dispatch_multi = nullptr;
void set_cpu_dispatch_multi_fn(CpuDispatchMultiFn fn) { g_cpu_dispatch_multi = fn; }
CpuDispatchMultiFn cpu_dispatch_multi_fn() { return g_cpu_dispatch_multi; }

namespace {  // reopen for buffer RAII

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
    std::uint32_t row_stride;  // 2 for interleaved GU, 1 for contiguous down
};

// Push constants for the SwiGLU shader.
struct SwigluPc {
    std::uint32_t n_elems;
    std::uint32_t n_chunks;
};

// ActQ buffer size in bytes (matches the shader's layout: 880 uints = 3520 bytes).
constexpr std::uint64_t kActQBufBytes = 880 * sizeof(std::uint32_t);

// SwiGLU output buffer size: must match the full ActQ format (880 uints = 3520 bytes)
// because the down GEMV reads scale at act[640+chunk] and hx at act[800+chunk].
constexpr std::uint64_t kSwigluOutBytes = 880 * sizeof(std::uint32_t);

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

bool IgpuExpertPool::run(void* jobs_ptr, int n, std::string& err) {
    err.clear();
    if (!ready_) {
        err = "iGPU pool not initialized";
        return false;
    }
    if (n <= 0) return true;

    auto* jobs = static_cast<PoolExpertJob*>(jobs_ptr);
    VulkanRuntime& rt = igpu_runtime();

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
    // Weight buffers sized for the full interleaved GU region:
    //   codes: 2*FF rows * ROW_GU(640) bytes = 819200 bytes
    //   scales: 2*FF rows * SC_GU(40) * 2 bytes = 102400 bytes
    // The GU region is interleaved (gate0, up0, gate1, up1, ...); we upload it
    // whole and use row_stride=2 in the GEMV shader to select even rows (gate)
    // or odd rows (up).
    constexpr std::uint64_t kGUCodesBytes = 2 * FF * 640;    // 819200
    constexpr std::uint64_t kGUScalesBytes = 2 * FF * 40 * 2; // 102400

    if (!bufs.make(rt, kActQBufBytes, d_act, err) ||
        !bufs.make(rt, FF * sizeof(float), d_gate, err) ||
        !bufs.make(rt, FF * sizeof(float), d_up, err) ||
        !bufs.make(rt, kSwigluOutBytes, d_swiglu_out, err) ||
        !bufs.make(rt, H * sizeof(float), d_down, err) ||
        !bufs.make(rt, kGUCodesBytes, d_wcodes, err) ||
        !bufs.make(rt, kGUScalesBytes, d_wscales, err)) {
        return false;
    }

    // Reset the descriptor pool to free sets from previous run() calls.
    rt.reset_desc_pool();

    // Descriptor sets: one for GEMV, one for SwiGLU (reused across dispatches).
    vk::DescriptorSet set_gemv = rt.make_set(sl_gemv_, nullptr, 0, err);
    if (set_gemv == nullptr) return false;
    vk::DescriptorSet set_swiglu = rt.make_set(sl_swiglu_, nullptr, 0, err);
    if (set_swiglu == nullptr) return false;

    // Process each expert
    for (int e = 0; e < n; ++e) {
        const auto& job = jobs[e];
        if (job.blob == nullptr || job.act == nullptr || job.out == nullptr) continue;

        // Upload activation to device
        if (!rt.upload(job.act, kActQBufBytes, d_act, err)) return false;

        // Gate GEMV: FF=640 rows, H/QK=40 blocks per row.
        // Upload the full interleaved GU codes/scales region (gate0, up0, gate1, up1, ...).
        // With row_stride=2 the shader reads even-indexed rows = gate rows.
        {
            const uint8_t* gu_codes = job.blob;  // O_GU_CODES = 0
            const uint8_t* gu_scales = job.blob + 1228800;  // O_GU_SCALES
            if (!rt.upload(gu_codes, kGUCodesBytes, d_wcodes, err)) return false;
            if (!rt.upload(gu_scales, kGUScalesBytes, d_wscales, err)) return false;

            vk::Buffer b[4] = {
                static_cast<vk::Buffer>(d_wcodes),
                static_cast<vk::Buffer>(d_wscales),
                static_cast<vk::Buffer>(d_act),
                static_cast<vk::Buffer>(d_gate)
            };
            rt.update_set(set_gemv, b, 4);
            GemvPc pc{FF, H / QK, 2};  // stride=2: even rows = gate
            std::string e2;
            if (!rt.dispatch(pipe_gemv_, pl_gemv_, set_gemv, &pc, sizeof(pc),
                            (FF + 63) / 64, 1, 1, e2)) {
                err = std::string("pool GEMV gate: ") + e2;
                return false;
            }
        }

        // Up GEMV: FF=640 rows, H/QK=40 blocks per row.
        // Same interleaved data, but offset by one row so stride=2 reads odd rows = up rows.
        {
            constexpr int ROW_GU = 640;   // bytes of codes per gate/up row
            constexpr int SC_GU = 40;     // weight blocks per gate/up row
            const uint8_t* up_codes = job.blob + ROW_GU;  // offset to first up row
            const uint8_t* up_scales = job.blob + 1228800 + SC_GU * 2;  // offset to first up row scales
            if (!rt.upload(up_codes, kGUCodesBytes, d_wcodes, err)) return false;
            if (!rt.upload(up_scales, kGUScalesBytes, d_wscales, err)) return false;

            vk::Buffer b[4] = {
                static_cast<vk::Buffer>(d_wcodes),
                static_cast<vk::Buffer>(d_wscales),
                static_cast<vk::Buffer>(d_act),
                static_cast<vk::Buffer>(d_up)
            };
            rt.update_set(set_gemv, b, 4);
            GemvPc pc{FF, H / QK, 2};  // stride=2: odd rows (from offset) = up
            std::string e2;
            if (!rt.dispatch(pipe_gemv_, pl_gemv_, set_gemv, &pc, sizeof(pc),
                            (FF + 63) / 64, 1, 1, e2)) {
                err = std::string("pool GEMV up: ") + e2;
                return false;
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
                return false;
            }
        }

        // Down GEMV: H=2560 rows, FF/QK=10 blocks per row.
        // Down region is contiguous (not interleaved), so row_stride=1.
        {
            StrataPackRow down_r = down_row(job.blob, 0);
            std::uint64_t down_codes_bytes = H * 160;
            std::uint64_t down_scales_bytes = H * 10 * 2;
            if (!rt.upload(down_r.codes, down_codes_bytes, d_wcodes, err)) return false;
            if (!rt.upload(down_r.scales, down_scales_bytes, d_wscales, err)) return false;

            vk::Buffer b[4] = {
                static_cast<vk::Buffer>(d_wcodes),
                static_cast<vk::Buffer>(d_wscales),
                static_cast<vk::Buffer>(d_swiglu_out),
                static_cast<vk::Buffer>(d_down)
            };
            rt.update_set(set_gemv, b, 4);
            GemvPc pc{H, FF / QK, 1};  // stride=1: contiguous rows
            std::string e2;
            if (!rt.dispatch(pipe_gemv_, pl_gemv_, set_gemv, &pc, sizeof(pc),
                            (H + 63) / 64, 1, 1, e2)) {
                err = std::string("pool GEMV down: ") + e2;
                return false;
            }
        }

        // Download result
        if (!rt.download(d_down, H * sizeof(float), job.out, err)) return false;
    }
    return true;
}

// ---- hybrid dispatcher ----

// Thread-local error string for propagating iGPU errors through the const-char* fail field.
static thread_local std::string g_igpu_error;

void hybrid_expert_dispatch(void* user, const float* x_f, const int32_t* ids,
                            const float* weights, int64_t n_embd, int64_t k, float* out) {
    // The user pointer is ExpertDispatch* (passed from igpu_drive_wrapper in generate.cpp).
    auto* d = static_cast<strata::core::ExpertDispatch*>(user);
    if (d == nullptr || d->failed) return;

    // Geometry check: the shaders are compiled for fixed dimensions.
    if (n_embd != H) {
        d->failed = true;
        d->fail = "iGPU pool: activation width mismatch (expected 2560)";
        d->fail_layer = d->layers;
        return;
    }

    // Lazy-init the pool (once, thread-safe).
    static IgpuExpertPool pool;
    static std::once_flag init_flag;
    std::call_once(init_flag, []() {
        std::string err;
        if (pool.init(err)) {
            std::fprintf(stderr, "strata igpu: Vulkan pool initialized OK\n");
        } else {
            std::fprintf(stderr, "strata igpu: Vulkan pool init FAILED: %s\n", err.c_str());
        }
    });

    // Route: large batches to iGPU, small to CPU.
    if (pool.available() && k > 4 && d->src != nullptr) {
        // Quantize the activation once for all experts in this batch.
        PoolActQ act;
        g_act_quant(x_f, static_cast<int>(n_embd), act);

        // Notify the source that a new layer is starting (matches CPU path).
        d->src->begin_layer(d->layers, ids, k);

        // Build job array, extracting blobs from ExpertSource.
        std::vector<PoolExpertJob> jobs(static_cast<std::size_t>(k));
        for (int64_t i = 0; i < k; ++i) {
            const int64_t e = ids[i];
            if (e < 0 || e >= d->n_expert) {
                d->failed = true;
                d->fail = "iGPU pool: routed expert id out of range";
                d->fail_layer = d->layers;
                d->fail_expert = e;
                return;
            }
            const uint8_t* b = d->src->blob(d->layers, e);
            if (b == nullptr) {
                d->failed = true;
                d->fail = "iGPU pool: expert source returned null blob";
                d->fail_layer = d->layers;
                d->fail_expert = e;
                ++d->missing;
                return;
            }
            jobs[static_cast<std::size_t>(i)].blob = b;
            jobs[static_cast<std::size_t>(i)].act = &act;
            jobs[static_cast<std::size_t>(i)].out = out + i * n_embd;
            jobs[static_cast<std::size_t>(i)].weight = weights ? weights[i] : 1.0f;
            jobs[static_cast<std::size_t>(i)].slot = static_cast<int>(i);
        }

        std::string err;
        if (!pool.run(jobs.data(), static_cast<int>(k), err)) {
            g_igpu_error = "iGPU pool run failed: " + err;
            d->failed = true;
            d->fail = g_igpu_error.c_str();
            d->fail_layer = d->layers;
            return;
        }
    } else {
        // CPU fallback.
        if (g_cpu_dispatch != nullptr) {
            g_cpu_dispatch(user, x_f, ids, weights, n_embd, k, out);
        }
    }
}

// ---- Multi-token hybrid dispatcher (for verify window / speculative decode) ----

void hybrid_expert_dispatch_multi(void* user, const float* x_f, const int32_t* ids,
                                  int64_t n_tok, int64_t k, float* out, int64_t layer) {
    auto* d = static_cast<strata::core::ExpertDispatch*>(user);
    if (d == nullptr || d->failed) return;

    d->layers = layer;

    // Lazy-init the pool (same static as single-token path).
    static IgpuExpertPool pool;
    static std::once_flag init_flag;
    std::call_once(init_flag, []() {
        std::string err;
        if (pool.init(err)) {
            std::fprintf(stderr, "strata igpu: Vulkan pool initialized OK\n");
        } else {
            std::fprintf(stderr, "strata igpu: Vulkan pool init FAILED: %s\n", err.c_str());
        }
    });

    if (pool.available() && k > 4 && d->src != nullptr) {
        // iGPU path: iterate over tokens.
        constexpr int64_t H = 2560;
        for (int64_t tok = 0; tok < n_tok; ++tok) {
            const float* tok_x = x_f + tok * H;
            const int32_t* tok_ids = ids + tok * k;
            float* tok_out = out + tok * k * H;

            PoolActQ act;
            g_act_quant(tok_x, static_cast<int>(H), act);

            d->src->begin_layer(layer, tok_ids, k);

            std::vector<PoolExpertJob> jobs(static_cast<std::size_t>(k));
            for (int64_t i = 0; i < k; ++i) {
                const int64_t e = tok_ids[i];
                if (e < 0 || e >= d->n_expert) {
                    d->failed = true; d->fail = "iGPU: expert id OOB"; d->fail_layer = layer; return;
                }
                const uint8_t* b = d->src->blob(layer, e);
                if (!b) { d->failed = true; d->fail = "iGPU: null blob"; d->fail_layer = layer; ++d->missing; return; }
                jobs[static_cast<std::size_t>(i)] = {b, &act, tok_out + i * H, 1.0f, static_cast<int>(i)};
            }
            std::string err;
            if (!pool.run(jobs.data(), static_cast<int>(k), err)) {
                d->failed = true; d->fail = "iGPU pool run failed"; d->fail_layer = layer; return;
            }
        }
    } else {
        // CPU fallback: use multi-token dispatcher directly.
        if (g_cpu_dispatch_multi != nullptr) {
            g_cpu_dispatch_multi(user, x_f, ids, n_tok, k, out, layer);
        }
    }
}

}  // namespace strata::kernels::igpu
