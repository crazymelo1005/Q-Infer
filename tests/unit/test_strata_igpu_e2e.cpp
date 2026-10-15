// End-to-end test for the iGPU expert pool integration (Strata-compatible).
//
// Verifies:
// 1. The iGPU pool library links correctly with the CPU kernels
// 2. The hybrid dispatcher falls back to CPU when Vulkan is unavailable
// 3. The act_quant function produces correct results
// 4. The Strata pack adapter produces correct row pointers
//
// This test does NOT require a Vulkan-capable iGPU. On machines without Vulkan,
// the pool gracefully falls back to CPU via the hybrid dispatcher.
//
// The Strata integration (Task 5) copies this code to strata/ with
// namespace updates (qinfer::experts::igpu -> strata::kernels::igpu). This test
// verifies the logic against Q-Infer's build; the Strata build verifies the
// namespace-migrated version.
//
// Build: part of the Q-Infer test suite (CMakeLists.txt).
// Run: ./test_strata_igpu_e2e

#include "experts/igpu/igpu_expert_pool.hpp"
#include "experts/igpu/strata_pack_adapter.hpp"
#include "experts/igpu/vulkan_runtime.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

// Reference ActQ quantization for comparison (matches Strata's contract).
void reference_act_quant(const float* x, int n, qinfer::experts::igpu::PoolActQ& a) {
    constexpr int QKA = 32;
    int nc = n / QKA;
    a.nchunks = nc;
    for (int c = 0; c < nc; ++c) {
        const float* chunk = x + c * QKA;
        float amax = 0.0f;
        for (int j = 0; j < QKA; ++j) {
            float ax = std::fabs(chunk[j]);
            if (ax > amax) amax = ax;
        }
        float scale = amax / 127.0f;
        float inv = (scale > 0.0f) ? 1.0f / scale : 0.0f;
        a.scale[c] = scale;
        int32_t s = 0;
        for (int j = 0; j < QKA; ++j) {
            float t = chunk[j] * inv;
            float hs = (t >= 0.0f) ? 0.5f : -0.5f;
            int q = static_cast<int>(t + hs);
            if (q > 127) q = 127;
            if (q < -127) q = -127;
            a.q[c * QKA + j] = static_cast<int8_t>(q);
            s += q;
        }
        a.sum[c] = s;
        a.hx[c] = scale * static_cast<float>(s);
    }
}

// Check that the act_quant function produces correct results.
bool test_act_quant() {
    std::printf("test_act_quant: ");

    constexpr int H = 2560;
    constexpr int QKA = 32;
    constexpr int MAXC = H / QKA;  // 80

    // Generate a test input
    std::vector<float> x(H);
    for (int i = 0; i < H; ++i) {
        x[i] = static_cast<float>(i % 100) * 0.01f - 0.5f;
    }

    // Run through the pool's quantization function
    qinfer::experts::igpu::PoolActQ pool_act{};
    qinfer::experts::igpu::ActQuantFn quant_fn = qinfer::experts::igpu::act_quant_fn();
    quant_fn(x.data(), H, pool_act);

    // Run through the reference implementation
    qinfer::experts::igpu::PoolActQ ref_act{};
    reference_act_quant(x.data(), H, ref_act);

    // Compare
    bool ok = true;
    if (pool_act.nchunks != ref_act.nchunks) {
        std::printf("FAIL nchunks %d vs %d\n", pool_act.nchunks, ref_act.nchunks);
        ok = false;
    }
    for (int c = 0; c < MAXC && ok; ++c) {
        if (pool_act.scale[c] != ref_act.scale[c]) {
            std::printf("FAIL scale[%d] %g vs %g\n", c, pool_act.scale[c], ref_act.scale[c]);
            ok = false;
        }
        if (pool_act.sum[c] != ref_act.sum[c]) {
            std::printf("FAIL sum[%d] %d vs %d\n", c, pool_act.sum[c], ref_act.sum[c]);
            ok = false;
        }
        if (pool_act.hx[c] != ref_act.hx[c]) {
            std::printf("FAIL hx[%d] %g vs %g\n", c, pool_act.hx[c], ref_act.hx[c]);
            ok = false;
        }
    }
    for (int i = 0; i < H && ok; ++i) {
        if (pool_act.q[i] != ref_act.q[i]) {
            std::printf("FAIL q[%d] %d vs %d\n", i, pool_act.q[i], ref_act.q[i]);
            ok = false;
        }
    }

    if (ok) std::printf("OK\n");
    return ok;
}

// Check that the Strata pack adapter produces correct row pointers.
bool test_pack_adapter() {
    std::printf("test_pack_adapter: ");

    constexpr int H = 2560;
    constexpr int FF = 640;
    constexpr int BLOB = 1382400;

    // Create a synthetic blob with recognizable patterns
    std::vector<uint8_t> blob(BLOB, 0);
    // Mark gate codes region start
    blob[0] = 0xAA;
    // Mark up codes region start (FF * ROW_GU = 640 * 640 = 409600)
    blob[409600] = 0xBB;
    // Mark down codes region start (O_D_CODES = 819200)
    blob[819200] = 0xCC;

    // Gate row 0
    auto gate = qinfer::experts::igpu::gu_row(blob.data(), 0, false);
    bool ok = (gate.codes[0] == 0xAA);
    if (!ok) {
        std::printf("FAIL gate row 0 codes[0] = 0x%02x (expected 0xAA)\n", gate.codes[0]);
        return false;
    }

    // Up row 0
    auto up = qinfer::experts::igpu::gu_row(blob.data(), 0, true);
    ok = (up.codes[0] == 0xBB);
    if (!ok) {
        std::printf("FAIL up row 0 codes[0] = 0x%02x (expected 0xBB)\n", up.codes[0]);
        return false;
    }

    // Down row 0
    auto down = qinfer::experts::igpu::down_row(blob.data(), 0);
    ok = (down.codes[0] == 0xCC);
    if (!ok) {
        std::printf("FAIL down row 0 codes[0] = 0x%02x (expected 0xCC)\n", down.codes[0]);
        return false;
    }

    // Check n_blocks
    ok = ok && (gate.n_blocks == 40);   // H/QK = 2560/64
    ok = ok && (down.n_blocks == 10);   // FF/QK = 640/64
    if (!ok) {
        std::printf("FAIL n_blocks gate=%d down=%d\n", gate.n_blocks, down.n_blocks);
        return false;
    }

    std::printf("OK\n");
    return true;
}

// Check that the hybrid dispatcher gracefully falls back to CPU when Vulkan
// is unavailable.
bool test_hybrid_fallback() {
    std::printf("test_hybrid_fallback: ");

    // Set a CPU dispatch function that records it was called
    static bool cpu_called = false;
    cpu_called = false;
    qinfer::experts::igpu::set_cpu_dispatch_fn(
        [](void* /*user*/, const float* /*x_f*/, const int32_t* /*ids*/,
           const float* /*weights*/, int64_t /*n_embd*/, int64_t /*k*/, float* out) {
            cpu_called = true;
            if (out != nullptr) out[0] = -1.0f;
        });

    // Call the hybrid dispatcher with k=2 (small batch, should go to CPU)
    float x_f[2560] = {};
    int32_t ids[2] = {0, 1};
    float weights[2] = {1.0f, 1.0f};
    float out[2 * 2560] = {};

    qinfer::experts::igpu::hybrid_expert_dispatch(
        nullptr, x_f, ids, weights, 2560, 2, out);

    if (!cpu_called) {
        std::printf("FAIL CPU fallback was not called\n");
        return false;
    }
    if (out[0] != -1.0f) {
        std::printf("FAIL CPU fallback did not write output\n");
        return false;
    }

    std::printf("OK\n");
    return true;
}

// Check that Vulkan runtime reports unavailability gracefully.
bool test_vulkan_availability() {
    std::printf("test_vulkan_availability: ");

    qinfer::experts::igpu::IgpuExpertPool pool;
    std::string err;
    bool init_ok = pool.init(err);

    if (!init_ok) {
        std::printf("OK (Vulkan unavailable: %s)\n", err.c_str());
        return true;
    }

    if (pool.available()) {
        std::printf("OK (Vulkan available)\n");
        return true;
    }

    std::printf("FAIL init succeeded but pool not available\n");
    return false;
}

}  // namespace

int main() {
    std::printf("=== iGPU expert pool end-to-end test ===\n\n");

    int pass = 0, fail = 0;

    if (test_act_quant()) ++pass; else ++fail;
    if (test_pack_adapter()) ++pass; else ++fail;
    if (test_hybrid_fallback()) ++pass; else ++fail;
    if (test_vulkan_availability()) ++pass; else ++fail;

    std::printf("\n=== Results: %d passed, %d failed ===\n", pass, fail);
    return fail > 0 ? 1 : 0;
}
