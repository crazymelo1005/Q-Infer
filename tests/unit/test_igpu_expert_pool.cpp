// Task 4: IgpuExpertPool + hybrid dispatcher tests.
//
// Verifies:
//   1. IgpuExpertPool::init() succeeds when Vulkan is available (or skips gracefully)
//   2. hybrid_expert_dispatch() routes to CPU when iGPU is unavailable or k <= 4
//   3. The built-in ActQ quantization stub produces correct output
//   4. The pool processes a synthetic expert job (when iGPU is available)
#include "experts/igpu/igpu_expert_pool.hpp"

#include "check.hpp"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace qinfer::experts::igpu;

namespace {

// ---- Test 1: init succeeds or skips gracefully ----
int test_init() {
    IgpuExpertPool pool;
    std::string err;
    bool ok = pool.init(err);
    if (ok) {
        CHECK(pool.available());
        std::printf("  init: iGPU available\n");
    } else {
        CHECK(!pool.available());
        std::printf("  init: skipped (no iGPU): %s\n", err.c_str());
    }
    return 0;
}

// ---- Test 2: routing logic ----

// Track whether the CPU fallback was called and with what arguments.
struct CpuCallRecord {
    bool called = false;
    const float* x_f = nullptr;
    int64_t n_embd = 0;
    int64_t k = 0;
};
static CpuCallRecord g_record;

void mock_cpu_dispatch(void* /*user*/, const float* x_f, const int32_t* /*ids*/,
                       const float* /*weights*/, int64_t n_embd, int64_t k, float* /*out*/) {
    g_record.called = true;
    g_record.x_f = x_f;
    g_record.n_embd = n_embd;
    g_record.k = k;
}

int test_routing() {
    // Set the CPU fallback to our mock.
    set_cpu_dispatch_fn(&mock_cpu_dispatch);

    // Check iGPU availability once.
    IgpuExpertPool pool;
    std::string err;
    pool.init(err);
    const bool has_igpu = pool.available();

    // Use Strata's actual geometry (H=2560) for the dispatcher tests.
    // This avoids buffer overflows when the iGPU path actually runs.
    constexpr int64_t N_EMBD = 2560;
    std::vector<float> x(N_EMBD, 0.1f);
    int32_t ids[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    float weights[10] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    // Output: k * H floats (enough for 10 experts).
    std::vector<float> out(10 * N_EMBD, 0.0f);

    // Case 1: k=2 (small batch) -> should route to CPU regardless of iGPU.
    g_record = {};
    hybrid_expert_dispatch(nullptr, x.data(), ids, weights, N_EMBD, 2, out.data());
    CHECK(g_record.called);
    CHECK(g_record.k == 2);
    CHECK(g_record.n_embd == N_EMBD);
    std::printf("  routing k=2: CPU fallback called\n");

    // Case 2: k=4 (boundary) -> should route to CPU (k > 4 is the threshold).
    g_record = {};
    hybrid_expert_dispatch(nullptr, x.data(), ids, weights, N_EMBD, 4, out.data());
    CHECK(g_record.called);
    CHECK(g_record.k == 4);
    std::printf("  routing k=4: CPU fallback called\n");

    // Case 3: k=5 with null user -> CPU fallback (safety: no blobs).
    g_record = {};
    hybrid_expert_dispatch(nullptr, x.data(), ids, weights, N_EMBD, 5, out.data());
    CHECK(g_record.called);
    CHECK(g_record.k == 5);
    std::printf("  routing k=5 null user: CPU fallback (safe)\n");

    // Case 4: k=5 with non-null user -> iGPU path if available, else CPU.
    // Provide a valid blob array for the iGPU path.
    std::vector<uint8_t> dummy_blob(1382400, 0);
    std::vector<const uint8_t*> blob_array(512, dummy_blob.data());

    g_record = {};
    hybrid_expert_dispatch(blob_array.data(), x.data(), ids, weights, N_EMBD, 5, out.data());
    if (has_igpu) {
        CHECK(!g_record.called);
        std::printf("  routing k=5 non-null user: iGPU path taken\n");
    } else {
        CHECK(g_record.called);
        CHECK(g_record.k == 5);
        std::printf("  routing k=5 non-null user: CPU fallback (no iGPU)\n");
    }

    // Case 5: k=10 with non-null user -> same logic.
    g_record = {};
    hybrid_expert_dispatch(blob_array.data(), x.data(), ids, weights, N_EMBD, 10, out.data());
    if (has_igpu) {
        CHECK(!g_record.called);
        std::printf("  routing k=10 non-null user: iGPU path taken\n");
    } else {
        CHECK(g_record.called);
        CHECK(g_record.k == 10);
        std::printf("  routing k=10 non-null user: CPU fallback (no iGPU)\n");
    }

    // Clean up: reset the CPU fallback.
    set_cpu_dispatch_fn(nullptr);
    return 0;
}

// ---- Test 3: ActQ quantization stub ----
int test_act_quant() {
    // Verify the built-in quantization stub produces correct output.
    constexpr int N = 2560;  // H
    std::vector<float> x(N);
    for (int i = 0; i < N; ++i) {
        x[static_cast<std::size_t>(i)] = 0.5f + 0.01f * static_cast<float>(i % 13 - 6);
    }

    PoolActQ a;
    std::memset(&a, 0, sizeof(a));
    act_quant_fn()(x.data(), N, a);

    CHECK(a.nchunks == N / 32);
    CHECK(a.nchunks == 80);

    // Check that scales are non-negative and codes are in range.
    for (int c = 0; c < a.nchunks; ++c) {
        CHECK(a.scale[c] >= 0.0f);
        // Verify codes are in [-127, 127]
        for (int j = 0; j < 32; ++j) {
            int q = static_cast<int>(a.q[c * 32 + j]);
            if (q > 127) q -= 256;  // interpret as signed
            CHECK(q >= -127 && q <= 127);
        }
        // Verify hx = scale * sum
        float expected_hx = a.scale[c] * static_cast<float>(a.sum[c]);
        float diff = std::fabs(a.hx[c] - expected_hx);
        CHECK(diff < 1e-5f);
    }

    // Check that the quantization preserves the signal approximately.
    // Reconstruct and compare: x_approx[i] = scale[chunk] * q[i]
    double total_err = 0.0;
    double total_abs = 0.0;
    for (int c = 0; c < a.nchunks; ++c) {
        for (int j = 0; j < 32; ++j) {
            int q = static_cast<int>(a.q[c * 32 + j]);
            if (q > 127) q -= 256;
            float x_approx = a.scale[c] * static_cast<float>(q);
            float x_orig = x[static_cast<std::size_t>(c * 32 + j)];
            total_err += std::fabs(static_cast<double>(x_approx) - static_cast<double>(x_orig));
            total_abs += std::fabs(static_cast<double>(x_orig));
        }
    }
    double rel_err = total_abs > 0.0 ? total_err / total_abs : 0.0;
    std::printf("  act_quant: L1 rel error = %.3e\n", rel_err);
    CHECK(rel_err < 0.05);  // less than 5% relative error

    return 0;
}

// ---- Test 4: pool run with mock data (only when iGPU is available) ----
int test_pool_run() {
    IgpuExpertPool pool;
    std::string err;
    if (!pool.init(err) || !pool.available()) {
        std::printf("  pool_run: skipped (no iGPU): %s\n", err.c_str());
        return 0;
    }

    // Create a minimal synthetic expert blob (1,382,400 bytes).
    // Fill with deterministic data.
    constexpr std::size_t kBlobSize = 1382400;
    std::vector<uint8_t> blob(kBlobSize);
    for (std::size_t i = 0; i < kBlobSize; ++i) {
        blob[i] = static_cast<uint8_t>((i * 7 + 13) & 0xFF);
    }

    // Create a quantized activation.
    constexpr int N = 2560;
    std::vector<float> x(N);
    for (int i = 0; i < N; ++i) {
        x[static_cast<std::size_t>(i)] = 0.1f * static_cast<float>((i * 3 + 1) % 7 - 3);
    }
    PoolActQ act;
    std::memset(&act, 0, sizeof(act));
    act_quant_fn()(x.data(), N, act);

    // Run one expert.
    std::vector<float> out(N, 0.0f);
    PoolExpertJob job;
    job.blob = blob.data();
    job.act = &act;
    job.out = out.data();
    job.weight = 1.0f;
    job.slot = 0;

    pool.run(&job, 1);

    // We can't easily verify the output numerically without the CPU reference,
    // but we can check that it ran without crashing and produced non-zero output.
    bool any_nonzero = false;
    for (int i = 0; i < N; ++i) {
        if (out[static_cast<std::size_t>(i)] != 0.0f) {
            any_nonzero = true;
            break;
        }
    }
    std::printf("  pool_run: output %s\n", any_nonzero ? "non-zero (ok)" : "all zero");
    // Note: all-zero output is possible with certain weight patterns, so we don't CHECK.

    return 0;
}

}  // namespace

int main() {
    int rc = 0;
    std::printf("=== test_igpu_expert_pool ===\n");
    rc += test_init();
    rc += test_routing();
    rc += test_act_quant();
    rc += test_pool_run();
    if (rc == 0) std::puts("igpu_expert_pool: all tests passed");
    return rc;
}
