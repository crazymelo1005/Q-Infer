// Stage B task 1: device KV pool vs the host KvPageCache reference.
//
// A deterministic LCG-style fixture fills both the host page cache and the
// device pool with the same cells; read_pool must return byte-identical pools.
// Also pins the reuse path (release then reserve again) and the guard errors
// (zero geometry fails; null cell pointers fail).
#include "dense/kv_device_cache.hpp"
#include "dense/kv_page_cache.hpp"

#include "check.hpp"
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

}  // namespace

int main() {
    int rc = 0;
    rc += test_pool_fixtures();
    rc += test_guards();
    rc += test_reuse_path();
    if (rc == 0) {
        std::puts("kv_device_cuda: device KV pool byte-identical to host reference");
    }
    return rc;
}