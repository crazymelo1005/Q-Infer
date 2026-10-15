// P1 spike: iGPU dense projection benchmark.
// Compares CPU vs iGPU for IQ4_XS × Q8_K GEMV at attn_qkv shape (10240 × 2560).
//
// Build:
//   g++ -O3 -std=c++20 -Isrc -Istrata/include \
//       measure/igpu_dense_spike/benchmark.cpp \
//       src/experts/igpu/vulkan_runtime.cpp \
//       src/kernels/iq4xs.cpp src/kernels/fp16.cpp \
//       -ldl -lpthread -o build/igpu_dense_spike
//
// Run: ./build/igpu_dense_spike

#include "experts/igpu/vulkan_runtime.hpp"
#include "kernels/iq4xs.hpp"
#include "kernels/fp16.hpp"
#include "kernels/iq4nl_tables.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>

namespace vk = qinfer::experts::igpu::vk;

// Shader SPIR-V data (embedded from compiled .spv)
#include "iq4xs_gemv.spv.inc"

// Generate random IQ4_XS weights for a matrix of shape [n_rows x n_cols]
// Use bounded scales to avoid overflow in accumulation
static void gen_iq4xs_weights(int n_rows, int n_cols, std::vector<uint8_t>& out) {
    const int n_blocks = n_cols / 256;
    out.resize(n_rows * n_blocks * 136);
    srand(42);
    for (int r = 0; r < n_rows; ++r) {
        for (int b = 0; b < n_blocks; ++b) {
            uint8_t* blk = out.data() + (r * n_blocks + b) * 136;
            // d: small fp16 (avoid large values)
            uint16_t d = 0x3800 | (rand() & 0x7F);  // ~0.015-0.03 range
            blk[0] = d & 0xFF;
            blk[1] = d >> 8;
            // scales_h: limit to small values (avoid large scales)
            uint16_t sh = rand() & 0x0003;  // only 2 bits used anyway
            blk[2] = sh & 0xFF;
            blk[3] = sh >> 8;
            // scales_l: limit low 4 bits to small values
            for (int i = 0; i < 4; ++i) blk[4 + i] = rand() & 0x0F;
            // qs: random 128 bytes (codebook values are bounded)
            for (int i = 0; i < 128; ++i) blk[8 + i] = rand() & 0xFF;
        }
    }
}

// Generate random Q8_K activation for a vector of length n
// Q8_K: 292 bytes per 256 elements (float d + int8 qs[256] + int16 bsums[16])
static void gen_q8k_activation(int n, std::vector<uint8_t>& out) {
    const int n_blocks = n / 256;
    out.resize(n_blocks * 292);
    srand(123);
    for (int b = 0; b < n_blocks; ++b) {
        uint8_t* blk = out.data() + b * 292;
        // d: small float
        float d = 0.01f + (rand() % 100) * 0.0001f;
        memcpy(blk, &d, 4);
        // qs: small int8 values (-64 to 63)
        for (int i = 0; i < 256; ++i) blk[4 + i] = (rand() & 0x7F) - 64;
        // bsums: compute sums of each 16-element group
        for (int g = 0; g < 16; ++g) {
            int16_t sum = 0;
            for (int i = 0; i < 16; ++i) {
                sum += blk[4 + g * 16 + i];
            }
            memcpy(blk + 260 + g * 2, &sum, 2);
        }
    }
}

// CPU reference: compute GEMV using the library function
static void cpu_gemv(const uint8_t* w, int n_rows, int n_blocks,
                     const uint8_t* y, std::vector<float>& out) {
    out.resize(n_rows);
    const qinfer::kernels::Q8kBlock* yb = reinterpret_cast<const qinfer::kernels::Q8kBlock*>(y);
    qinfer::kernels::iq4xs_gemv_q8k(w, n_rows, n_blocks, yb, out.data());
}

int main() {
    printf("=== P1 Spike: iGPU Dense Projection Benchmark ===\n\n");

    // attn_qkv shape: 10240 x 2560, IQ4_XS weights, Q8_K activation
    const int n_rows = 10240;
    const int n_cols = 2560;
    const int n_blocks = n_cols / 256;  // 10

    printf("Matrix shape: %d x %d (n_blocks = %d)\n", n_rows, n_cols, n_blocks);
    printf("Weight size: %.2f MiB\n", n_rows * n_blocks * 136.0 / 1024 / 1024);
    printf("Activation size: %.2f KiB\n\n", (n_cols / 256) * 292.0 / 1024);

    // Generate random data
    std::vector<uint8_t> weights, activation;
    gen_iq4xs_weights(n_rows, n_cols, weights);
    gen_q8k_activation(n_cols, activation);

    // CPU reference
    printf("Running CPU reference...\n");
    auto t0 = std::chrono::high_resolution_clock::now();
    std::vector<float> cpu_out;
    const int cpu_iters = 10;
    for (int i = 0; i < cpu_iters; ++i) {
        cpu_gemv(weights.data(), n_rows, n_blocks, activation.data(), cpu_out);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / cpu_iters;
    printf("CPU: %.3f ms/iter (avg of %d)\n\n", cpu_ms, cpu_iters);

    // iGPU
    printf("Initializing iGPU...\n");
    qinfer::experts::igpu::VulkanRuntime rt;
    std::string err;
    if (!rt.load(err)) {
        printf("iGPU load failed: %s\n", err.c_str());
        return 1;
    }
    if (!rt.init_device(err)) {
        printf("iGPU init failed: %s\n", err.c_str());
        return 1;
    }
    printf("iGPU device: %s\n\n", rt.device_name());

    // Create pipeline
    vk::DescriptorSetLayoutBinding bindings[3] = {};
    for (int i = 0; i < 3; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = vk::DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = vk::SHADER_STAGE_COMPUTE;
    }
    vk::DescriptorSetLayout sl = rt.make_set_layout(bindings, 3, err);
    if (!sl) {
        printf("set layout failed: %s\n", err.c_str());
        return 1;
    }
    vk::PipelineLayout pl = rt.make_layout(&sl, 1, err);
    if (!pl) {
        printf("pipeline layout failed: %s\n", err.c_str());
        return 1;
    }
    vk::Pipeline pipeline = rt.make_pipeline(iq4xs_gemv_spv, sizeof(iq4xs_gemv_spv), pl, err);
    if (!pipeline) {
        printf("pipeline failed: %s\n", err.c_str());
        return 1;
    }

    // Allocate buffers
    void *d_w = nullptr, *d_y = nullptr, *d_o = nullptr;
    uint64_t w_bytes = n_rows * n_blocks * 136;
    uint64_t y_bytes = (n_cols / 256) * 292;
    uint64_t o_bytes = n_rows * sizeof(float);
    if (!rt.alloc_device(w_bytes, d_w, err) ||
        !rt.alloc_device(y_bytes, d_y, err) ||
        !rt.alloc_device(o_bytes, d_o, err)) {
        printf("alloc failed: %s\n", err.c_str());
        return 1;
    }

    // Upload weights and activation
    if (!rt.upload(weights.data(), w_bytes, d_w, err) ||
        !rt.upload(activation.data(), y_bytes, d_y, err)) {
        printf("upload failed: %s\n", err.c_str());
        return 1;
    }

    // Create descriptor set
    vk::DescriptorSet set = rt.make_set(sl, nullptr, 0, err);
    if (!set) {
        printf("descriptor set failed: %s\n", err.c_str());
        return 1;
    }
    vk::Buffer bufs[3] = {
        static_cast<vk::Buffer>(d_w),
        static_cast<vk::Buffer>(d_y),
        static_cast<vk::Buffer>(d_o)
    };
    rt.update_set(set, bufs, 3);

    // Run iGPU benchmark
    printf("Running iGPU benchmark...\n");
    struct Pc { uint32_t n_rows; uint32_t n_blocks; } pc = {n_rows, n_blocks};
    uint32_t n_groups = (n_rows + 63) / 64;

    // Warmup
    rt.dispatch(pipeline, pl, set, &pc, sizeof(pc), n_groups, 1, 1, err);

    auto t2 = std::chrono::high_resolution_clock::now();
    const int gpu_iters = 100;
    for (int i = 0; i < gpu_iters; ++i) {
        rt.dispatch(pipeline, pl, set, &pc, sizeof(pc), n_groups, 1, 1, err);
    }
    auto t3 = std::chrono::high_resolution_clock::now();
    double gpu_ms = std::chrono::duration<double, std::milli>(t3 - t2).count() / gpu_iters;
    printf("iGPU: %.3f ms/iter (avg of %d)\n\n", gpu_ms, gpu_iters);

    // Download and compare
    std::vector<float> gpu_out(n_rows);
    if (!rt.download(d_o, o_bytes, gpu_out.data(), err)) {
        printf("download failed: %s\n", err.c_str());
        return 1;
    }

    // Check correctness
    double max_err = 0.0, rel_err = 0.0;
    int n_mismatch = 0;
    int first_bad = -1;
    for (int i = 0; i < n_rows; ++i) {
        double diff = std::abs(cpu_out[i] - gpu_out[i]);
        double ref = std::abs(cpu_out[i]);
        if (diff > max_err) max_err = diff;
        if (ref > 1e-6) {
            double re = diff / ref;
            if (re > rel_err) rel_err = re;
        }
        if (diff > 1e-3 * std::max(1.0, ref)) {
            ++n_mismatch;
            if (first_bad < 0) first_bad = i;
        }
    }

    printf("=== Results ===\n");
    printf("CPU time:  %.3f ms\n", cpu_ms);
    printf("iGPU time: %.3f ms\n", gpu_ms);
    printf("Speedup:   %.2fx\n", cpu_ms / gpu_ms);
    printf("Max abs error: %.6f\n", max_err);
    printf("Max rel error: %.6f\n", rel_err);
    printf("Mismatched rows: %d / %d\n", n_mismatch, n_rows);
    
    if (first_bad >= 0) {
        printf("\nFirst mismatch at row %d:\n", first_bad);
        printf("  CPU: %.6f (isnan=%d, isinf=%d)\n", cpu_out[first_bad], 
               std::isnan(cpu_out[first_bad]), std::isinf(cpu_out[first_bad]));
        printf("  GPU: %.6f (isnan=%d, isinf=%d)\n", gpu_out[first_bad],
               std::isnan(gpu_out[first_bad]), std::isinf(gpu_out[first_bad]));
    }

    // Cleanup
    rt.free_buf(d_w, err);
    rt.free_buf(d_y, err);
    rt.free_buf(d_o, err);

    return 0;
}
