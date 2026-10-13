// Stage B task 1: device KV pool vs the host KvPageCache reference.
//
// A deterministic LCG-style fixture fills both the host page cache and the
// device pool with the same cells; read_pool must return byte-identical pools.
// Also pins the reuse path (release then reserve again) and the guard errors
// (zero geometry fails; null cell pointers fail).
#include "dense/kv_device_cache.hpp"
#include "dense/kv_page_cache.hpp"
#include "dense/qsa_index_attend_cuda.hpp"
#include "dense/qsa_select.hpp"

#include "check.hpp"
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

using namespace qinfer::dense;

namespace {

// Deterministic fp16-ish bit pattern (uint16 words) for a cell value.
std::uint16_t cell_val(std::uint64_t t, std::uint64_t h, std::uint64_t j) {
    return static_cast<std::uint16_t>((t * 1009u + h * 977u + j * 31u) & 0xFFFFu);
}

// Fill host cache and device pool with cells t = 0..n_cells-1 and compare.
int run_fixture(KvDeviceExec& exec, std::uint64_t n_pages, std::uint64_t n_head_kv,
                std::uint64_t page_size, std::uint64_t head_dim, std::uint64_t n_cells) {
    const std::uint64_t elems = n_pages * n_head_kv * page_size * head_dim;

    // Host reference: identity page table (same mapping the device uses here).
    KvPageCache hc;
    hc.n_pages = n_pages;
    hc.n_head_kv = n_head_kv;
    hc.page_size = page_size;
    hc.head_dim = head_dim;
    hc.reset();
    for (std::uint64_t l = 0; l < n_pages; ++l) {
        hc.table[static_cast<std::size_t>(l)] = static_cast<std::int32_t>(l);
    }

    KvDevicePool* dp = nullptr;
    std::string err;
    if (!exec.reserve(dp, n_pages, n_head_kv, page_size, head_dim, err)) {
        std::printf("FAIL device reserve: %s\n", err.c_str());
        return 1;
    }

    for (std::uint64_t t = 0; t < n_cells; ++t) {
        std::vector<std::uint16_t> k(n_head_kv * head_dim), v(n_head_kv * head_dim);
        for (std::uint64_t h = 0; h < n_head_kv; ++h) {
            for (std::uint64_t j = 0; j < head_dim; ++j) {
                k[static_cast<std::size_t>(h * head_dim + j)] =
                    cell_val(t, h, j);
                v[static_cast<std::size_t>(h * head_dim + j)] =
                    static_cast<std::uint16_t>(cell_val(t, h, j) ^ 0x55AAu);
            }
        }
        const KvCell cell{k.data(), v.data()};
        if (!kv_append(hc, t, cell, err)) {
            std::printf("FAIL host kv_append: %s\n", err.c_str());
            exec.release(dp, err);
            return 1;
        }
        if (!exec.append(dp, t, cell, err)) {
            std::printf("FAIL device append: %s\n", err.c_str());
            exec.release(dp, err);
            return 1;
        }
    }

    std::vector<std::uint16_t> dk(elems), dv(elems);
    if (!exec.read_pool(dp, dk.data(), dv.data(), err)) {
        std::printf("FAIL read_pool: %s\n", err.c_str());
        exec.release(dp, err);
        return 1;
    }

    std::uint64_t bad_k = 0, bad_v = 0;
    for (std::uint64_t i = 0; i < elems; ++i) {
        if (hc.k_pool[static_cast<std::size_t>(i)] != dk[static_cast<std::size_t>(i)]) ++bad_k;
        if (hc.v_pool[static_cast<std::size_t>(i)] != dv[static_cast<std::size_t>(i)]) ++bad_v;
    }
    exec.release(dp, err);

    if (bad_k != 0 || bad_v != 0) {
        std::printf("FAIL device vs host pools differ: k %llu / v %llu of %llu\n",
                    (unsigned long long) bad_k, (unsigned long long) bad_v,
                    (unsigned long long) elems);
        return 1;
    }
    std::printf("  byte-identical pools (%llu elems) for %llu cells\n",
                (unsigned long long) elems, (unsigned long long) n_cells);
    return 0;
}

int test_pool_fixtures() {
    int bad = 0;
    KvDeviceExec exec = kv_device_exec();
    // Small fixture: exactly the serve_fixture_gen-style geometry.
    bad += run_fixture(exec, 8, 2, 4, 32, 32);
    // Large fixture (8K cells at page_size 4 = 2048 pages): residency volume.
    bad += run_fixture(exec, 2048, 2, 4, 256, 8192);
    return bad;
}

int test_guards() {
    int bad = 0;
    KvDeviceExec exec = kv_device_exec();
    std::string err;
    KvDevicePool* p = nullptr;
    if (exec.reserve(p, 0, 2, 4, 32, err)) {
        std::printf("FAIL zero n_pages should be rejected\n");
        bad = 1;
    } else if (p != nullptr) {
        std::printf("FAIL reserve must leave out nullptr on error\n");
        bad = 1;
    }
    if (exec.release(nullptr, err) == false) {
        std::printf("FAIL null release should be a no-op success\n");
        bad = 1;
    }
    if (bad == 0) std::printf("  guard checks: zero geometry + null release ok\n");
    return bad;
}

int test_reuse_path() {
    KvDeviceExec exec = kv_device_exec();
    std::string err;
    KvDevicePool* p = nullptr;
    if (!exec.reserve(p, 4, 1, 2, 16, err)) return 1;
    if (!exec.release(p, err)) return 1;
    if (!exec.reserve(p, 4, 1, 2, 16, err)) return 1;
    const std::uint16_t kk[16] = {0};
    const std::uint16_t vv[16] = {0};
    const KvCell cell{kk, vv};
    if (!exec.append(p, 0, cell, err)) return 1;
    exec.release(p, err);
    std::printf("  reuse path: reserve -> release -> reserve -> append ok\n");
    return 0;
}

// Fill both host/device caches with n_cells cells (same fixture as run_fixture).
// Returns the host KvPageCache (identity table) for use by the gather tests.
bool fill_caches(KvDeviceExec& exec, KvPageCache& hc, KvDevicePool*& dp,
                 std::uint64_t n_pages, std::uint64_t n_head_kv, std::uint64_t page_size,
                 std::uint64_t head_dim, std::uint64_t n_cells, std::string& err) {
    hc.n_pages = n_pages;
    hc.n_head_kv = n_head_kv;
    hc.page_size = page_size;
    hc.head_dim = head_dim;
    hc.reset();
    for (std::uint64_t l = 0; l < n_pages; ++l) {
        hc.table[static_cast<std::size_t>(l)] = static_cast<std::int32_t>(l);
    }
    if (!exec.reserve(dp, n_pages, n_head_kv, page_size, head_dim, err)) return false;
    for (std::uint64_t t = 0; t < n_cells; ++t) {
        std::vector<std::uint16_t> k(n_head_kv * head_dim), v(n_head_kv * head_dim);
        for (std::uint64_t h = 0; h < n_head_kv; ++h) {
            for (std::uint64_t j = 0; j < head_dim; ++j) {
                k[static_cast<std::size_t>(h * head_dim + j)] = cell_val(t, h, j);
                v[static_cast<std::size_t>(h * head_dim + j)] =
                    static_cast<std::uint16_t>(cell_val(t, h, j) ^ 0x55AAu);
            }
        }
        const KvCell cell{k.data(), v.data()};
        if (!kv_append(hc, t, cell, err) || !exec.append(dp, t, cell, err)) return false;
    }
    return true;
}

// Stage B task 2: device gather must reproduce host kv_gather byte-for-byte,
// and the attend device-buffer variant must match the host scratch path.
int test_gather_and_attend() {
    const std::uint64_t nk = 2, ps = 4, hd = 32, nh = 4, cells = 32;
    KvDeviceExec exec = kv_device_exec();
    std::string err;
    KvPageCache hc;
    KvDevicePool* dp = nullptr;
    if (!fill_caches(exec, hc, dp, 8, nk, ps, hd, cells, err)) {
        std::printf("FAIL fill: %s\n", err.c_str());
        return 1;
    }
    const std::uint64_t ids_dense[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    const std::uint64_t ids_sparse[] = {0, 3, 4, 9, 17, 30};
    const std::uint64_t* cases[2] = {ids_dense, ids_sparse};
    const std::uint64_t case_n[2] = {12, 6};
    int bad = 0;
    for (int ci = 0; ci < 2; ++ci) {
        const std::uint64_t n_ids = case_n[ci];
        const std::size_t elems = static_cast<std::size_t>(n_ids * nk * hd);
        std::vector<std::uint16_t> hk(elems), hv(elems);
        if (!kv_gather(hc, cases[ci], n_ids, hk.data(), hv.data(), err)) {
            std::printf("FAIL host gather: %s\n", err.c_str());
            bad = 1;
            break;
        }
        std::uint16_t* dk = nullptr;
        std::uint16_t* dv = nullptr;
        if (cudaMalloc(&dk, elems * sizeof(std::uint16_t)) != cudaSuccess ||
            cudaMalloc(&dv, elems * sizeof(std::uint16_t)) != cudaSuccess) {
            std::printf("FAIL device scratch alloc\n");
            bad = 1;
            break;
        }
        if (!exec.gather(dp, cases[ci], n_ids, dk, dv, err)) {
            std::printf("FAIL device gather: %s\n", err.c_str());
            bad = 1;
            cudaFree(dk);
            cudaFree(dv);
            break;
        }
        std::vector<std::uint16_t> gk(elems), gv(elems);
        cudaMemcpy(gk.data(), dk, elems * sizeof(std::uint16_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(gv.data(), dv, elems * sizeof(std::uint16_t), cudaMemcpyDeviceToHost);
        std::uint64_t mism = 0;
        for (std::size_t i = 0; i < elems; ++i) {
            if (hk[i] != gk[i] || hv[i] != gv[i]) ++mism;
        }
        std::printf("  gather case %d (n_ids %llu): %s\n", ci,
                    (unsigned long long) n_ids, mism == 0 ? "byte-identical" : "MISMATCH");
        if (mism != 0) bad = 1;
        cudaFree(dk);
        cudaFree(dv);
    }

    // Attend: same q / q_full through the host-scratch path and the device path.
    const std::uint64_t n_ids = 12;
    const std::size_t nk_elems = static_cast<std::size_t>(n_ids * nk * hd);
    std::vector<std::uint16_t> hk(nk_elems), hv(nk_elems);
    kv_gather(hc, ids_dense, n_ids, hk.data(), hv.data(), err);
    std::vector<float> q(static_cast<std::size_t>(nh * hd));
    std::vector<float> qf(static_cast<std::size_t>(nh * 2 * hd));
    for (std::size_t i = 0; i < q.size(); ++i) q[i] = 0.1f * static_cast<float>(i % 17) - 0.8f;
    for (std::size_t i = 0; i < qf.size(); ++i) qf[i] = 0.05f * static_cast<float>(i % 23) - 0.5f;
    std::vector<float> attn_h(static_cast<std::size_t>(nh * hd)), attn_d(attn_h.size());
    std::vector<float> w_h(static_cast<std::size_t>(nh * n_ids)), w_d(w_h.size());
    QsaShapes sh;
    sh.head_dim = hd;
    sh.n_head = nh;
    sh.n_head_kv = nk;
    if (!qsa_attend_cuda(q.data(), hk.data(), hv.data(), n_ids, sh, qf.data(), attn_h.data(),
                         w_h.data(), nullptr, err)) {
        std::printf("FAIL host attend: %s\n", err.c_str());
        bad = 1;
        exec.release(dp, err);
        return bad;
    }
    std::uint16_t* dk = nullptr;
    std::uint16_t* dv = nullptr;
    float* d_q = nullptr;
    float* d_qf = nullptr;
    float* d_attn = nullptr;
    float* d_w = nullptr;
    const std::size_t qb = q.size() * sizeof(float);
    const std::size_t qfb = qf.size() * sizeof(float);
    const std::size_t kvb = nk_elems * sizeof(std::uint16_t);
    const std::size_t ab = attn_h.size() * sizeof(float);
    const std::size_t wb = w_h.size() * sizeof(float);
    const bool alloc_ok =
        cudaMalloc(&dk, kvb) == cudaSuccess && cudaMalloc(&dv, kvb) == cudaSuccess &&
        cudaMalloc(&d_q, qb) == cudaSuccess && cudaMalloc(&d_qf, qfb) == cudaSuccess &&
        cudaMalloc(&d_attn, ab) == cudaSuccess && cudaMalloc(&d_w, wb) == cudaSuccess;
    if (!alloc_ok) {
        std::printf("FAIL attend device alloc\n");
        bad = 1;
    } else {
        const bool up = cudaMemcpy(d_q, q.data(), qb, cudaMemcpyHostToDevice) == cudaSuccess &&
                        cudaMemcpy(d_qf, qf.data(), qfb, cudaMemcpyHostToDevice) == cudaSuccess;
        if (!up) {
            std::printf("FAIL attend device q upload\n");
            bad = 1;
        } else if (!exec.gather(dp, ids_dense, n_ids, dk, dv, err) ||
                   !qsa_attend_cuda_dev(d_q, d_qf, dk, dv, n_ids, sh, d_attn, d_w, nullptr, err)) {
            std::printf("FAIL device attend chain: %s\n", err.c_str());
            bad = 1;
        } else {
            cudaMemcpy(attn_d.data(), d_attn, ab, cudaMemcpyDeviceToHost);
            cudaMemcpy(w_d.data(), d_w, wb, cudaMemcpyDeviceToHost);
        std::uint64_t same = 0;
        double diff = 0.0, abs = 0.0;
        for (std::size_t i = 0; i < attn_h.size(); ++i) {
            if (std::bit_cast<std::uint32_t>(attn_h[i]) == std::bit_cast<std::uint32_t>(attn_d[i])) {
                ++same;
            }
            diff += std::abs(static_cast<double>(attn_h[i]) - static_cast<double>(attn_d[i]));
            abs += std::abs(static_cast<double>(attn_h[i]));
        }
        const double l1 = abs > 0.0 ? diff / abs : 0.0;
        const double ratio = static_cast<double>(same) / static_cast<double>(attn_h.size());
        std::printf("  attend dev-vs-host: bit-same %.3f, L1 rel %.3e, weights diff %.3e\n",
                    ratio, l1, static_cast<double>(w_d[0] - w_h[0]));
        if (!(ratio >= 0.99 && l1 <= 1e-4)) bad = 1;
        }
    }
    cudaFree(dk);
    cudaFree(dv);
    cudaFree(d_q);
    cudaFree(d_qf);
    cudaFree(d_attn);
    cudaFree(d_w);
    exec.release(dp, err);
    return bad;
}

// Stage B task 2 timing note: 8K-cell fixture (2048 pages), gather width like
// the real selection (min(n_kv, idx_top_k + idx_block - 1) clamp) — report
// host-path vs device-path attn-segment time.
int test_8k_timing_probe() {
    const std::uint64_t nk = 2, ps = 4, hd = 256, nh = 24;
    const std::uint64_t n_pages = 2048, cells = 8192;
    const std::uint64_t n_ids = 2048;  // ~ real selection width
    KvDeviceExec exec = kv_device_exec();
    std::string err;
    KvPageCache hc;
    KvDevicePool* dp = nullptr;
    if (!fill_caches(exec, hc, dp, n_pages, nk, ps, hd, cells, err)) {
        std::printf("FAIL 8K fill: %s\n", err.c_str());
        return 1;
    }
    std::vector<std::uint64_t> ids(n_ids);
    for (std::uint64_t i = 0; i < n_ids; ++i) ids[i] = i;  // ascending prefix
    std::vector<std::uint16_t> hk(n_ids * nk * hd), hv(n_ids * nk * hd);
    std::vector<float> q(nh * hd), qf(nh * 2 * hd);
    for (std::size_t i = 0; i < q.size(); ++i) q[i] = 0.1f * static_cast<float>(i % 17) - 0.8f;
    for (std::size_t i = 0; i < qf.size(); ++i) qf[i] = 0.05f * static_cast<float>(i % 23) - 0.5f;
    std::vector<float> attn(nh * hd), w(nh * n_ids);
    QsaShapes sh;
    sh.head_dim = hd;
    sh.n_head = nh;
    sh.n_head_kv = nk;

    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();
    kv_gather(hc, ids.data(), n_ids, hk.data(), hv.data(), err);
    const auto t1 = clock::now();
    qsa_attend_cuda(q.data(), hk.data(), hv.data(), n_ids, sh, qf.data(), attn.data(), w.data(),
                    nullptr, err);
    const auto t2 = clock::now();
    const double host_ms = std::chrono::duration<double, std::milli>(t2 - t0).count();

    std::uint16_t* dk = nullptr;
    std::uint16_t* dv = nullptr;
    float* d_q = nullptr;
    float* d_qf = nullptr;
    float* d_attn = nullptr;
    float* d_w = nullptr;
    const std::size_t kvd = n_ids * nk * hd * sizeof(std::uint16_t);
    if (cudaMalloc(&dk, kvd) != cudaSuccess || cudaMalloc(&dv, kvd) != cudaSuccess ||
        cudaMalloc(&d_q, q.size() * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&d_qf, qf.size() * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&d_attn, attn.size() * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&d_w, w.size() * sizeof(float)) != cudaSuccess) {
        std::printf("FAIL 8K device alloc\n");
        return 1;
    }
    cudaMemcpy(d_q, q.data(), q.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_qf, qf.data(), qf.size() * sizeof(float), cudaMemcpyHostToDevice);
    const auto t3 = clock::now();
    const bool ok = exec.gather(dp, ids.data(), n_ids, dk, dv, err) &&
                    qsa_attend_cuda_dev(d_q, d_qf, dk, dv, n_ids, sh, d_attn, d_w, nullptr, err);
    const auto t4 = clock::now();
    const double dev_ms = std::chrono::duration<double, std::milli>(t4 - t3).count();
    cudaFree(dk);
    cudaFree(dv);
    cudaFree(d_q);
    cudaFree(d_qf);
    cudaFree(d_attn);
    cudaFree(d_w);
    exec.release(dp, err);
    if (!ok) {
        std::printf("FAIL 8K device path: %s\n", err.c_str());
        return 1;
    }
    std::printf("  8K attn-segment (n_ids %llu): host %.1f ms vs device %.1f ms (%.1fx)\n",
                (unsigned long long) n_ids, host_ms, dev_ms, host_ms / dev_ms);
    return 0;
}

}  // namespace

int main() {
    int rc = 0;
    rc += test_pool_fixtures();
    rc += test_guards();
    rc += test_reuse_path();
    rc += test_gather_and_attend();
    rc += test_8k_timing_probe();
    if (rc == 0) {
        std::puts("kv_device_cuda: device KV pool + gather + attend all match host reference");
    }
    return rc;
}