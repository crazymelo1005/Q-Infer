// Device KV pool implementation (stage B task 1).
//
// Offsets mirror host kv_append (kv_page_cache.cpp) exactly, so the device
// path and the host reference produce byte-identical pools; the per-step
// cross-bus traffic this eliminates is only the gathered scratch, which task 2
// removes from the attend path.
#include "dense/kv_device_cache.hpp"

#include <cuda_runtime.h>

#include <vector>

#include "dense/kv_page_cache.hpp"

namespace qinfer::dense {

// Opaque pool: device k/v pools + page table, geometry kept for offset math
// and the host mirror of the table (identity mapping at this stage; eviction
// and readmission arrive with --kv-resident in task 4).
struct KvDevicePool {
    std::uint16_t* k_pool = nullptr;
    std::uint16_t* v_pool = nullptr;
    std::int32_t* d_table = nullptr;
    std::vector<std::int32_t> table;  // host mirror (identity at this stage)
    std::uint64_t n_pages = 0;
    std::uint64_t n_head_kv = 0;
    std::uint64_t page_size = 0;
    std::uint64_t head_dim = 0;
    std::size_t pool_elems = 0;
};

namespace {

// Release helper shared by release and the reserve error path.
bool kv_free(KvDevicePool* p, std::string& err) {
    err.clear();
    if (p == nullptr) return true;  // null pool is a valid no-op release
    cudaError_t bad = cudaSuccess;
    if (p->k_pool != nullptr && cudaFree(p->k_pool) != cudaSuccess) bad = cudaGetLastError();
    if (p->v_pool != nullptr && cudaFree(p->v_pool) != cudaSuccess) bad = cudaGetLastError();
    if (p->d_table != nullptr && cudaFree(p->d_table) != cudaSuccess) bad = cudaGetLastError();
    if (bad != cudaSuccess) {
        err = std::string("device KV release failed: ") + cudaGetErrorString(bad);
        delete p;
        return false;
    }
    delete p;
    return true;
}

// Identity page mapping at this stage: logical page l -> physical page l.
void fill_identity_table(std::vector<std::int32_t>& t, std::uint64_t n_pages) {
    t.resize(static_cast<std::size_t>(n_pages));
    for (std::uint64_t i = 0; i < n_pages; ++i) {
        t[static_cast<std::size_t>(i)] = static_cast<std::int32_t>(i);
    }
}

// One cell's byte span in a pool, per head (same math as host row_in_page).
std::size_t cell_offset(std::uint64_t page, std::uint64_t h, std::uint64_t t,
                        std::uint64_t page_size, std::uint64_t n_head_kv,
                        std::uint64_t head_dim) {
    const std::uint64_t stride = n_head_kv * head_dim;
    return static_cast<std::size_t>(
        page * stride * page_size + (h * page_size + t % page_size) * head_dim);
}

bool dev_reserve(KvDevicePool*& out, std::uint64_t n_pages, std::uint64_t n_head_kv,
                 std::uint64_t page_size, std::uint64_t head_dim, std::string& err) {
    err.clear();
    out = nullptr;
    if (n_pages == 0 || n_head_kv == 0 || page_size == 0 || head_dim == 0) {
        err = "device KV reserve: geometry must be positive";
        return false;
    }
    KvDevicePool* p = new KvDevicePool();
    p->n_pages = n_pages;
    p->n_head_kv = n_head_kv;
    p->page_size = page_size;
    p->head_dim = head_dim;
    p->pool_elems = static_cast<std::size_t>(n_pages * n_head_kv * page_size * head_dim);
    const std::size_t bytes = p->pool_elems * sizeof(std::uint16_t);
    const std::size_t tab_bytes = static_cast<std::size_t>(n_pages) * sizeof(std::int32_t);
    fill_identity_table(p->table, n_pages);
    const cudaError_t st =
        cudaMalloc(&p->k_pool, bytes) != cudaSuccess ? cudaErrorMemoryAllocation
        : cudaMalloc(&p->v_pool, bytes) != cudaSuccess ? cudaErrorMemoryAllocation
        : cudaMalloc(&p->d_table, tab_bytes) != cudaSuccess ? cudaErrorMemoryAllocation
        : cudaSuccess;
    if (st != cudaSuccess) {
        err = std::string("device KV reserve: allocation failed: ") + cudaGetErrorString(st);
        kv_free(p, err);
        return false;
    }
    const cudaError_t ct =
        cudaMemcpy(p->d_table, p->table.data(), tab_bytes, cudaMemcpyHostToDevice);
    if (ct != cudaSuccess) {
        err = std::string("device KV reserve: table upload failed: ") + cudaGetErrorString(ct);
        kv_free(p, err);
        return false;
    }
    out = p;
    return true;
}

bool dev_append(KvDevicePool* p, std::uint64_t t, const KvCell& cell, std::string& err) {
    err.clear();
    if (p == nullptr || cell.k == nullptr || cell.v == nullptr) {
        err = "device KV append: pool or cell pointers incomplete";
        return false;
    }
    const std::uint64_t logical = t / p->page_size;
    if (logical >= p->n_pages) {
        err = "device KV append: logical page " + std::to_string(logical) + " out of range";
        return false;
    }
    const std::uint64_t page = static_cast<std::uint64_t>(p->table[static_cast<std::size_t>(logical)]);
    const std::size_t cell_bytes = static_cast<std::size_t>(p->head_dim) * sizeof(std::uint16_t);
    for (std::uint64_t h = 0; h < p->n_head_kv; ++h) {
        const std::size_t at = cell_offset(page, h, t, p->page_size, p->n_head_kv, p->head_dim);
        const cudaError_t st =
            cudaMemcpy(p->k_pool + at, cell.k + h * p->head_dim, cell_bytes,
                       cudaMemcpyHostToDevice);
        if (st != cudaSuccess) {
            err = std::string("device KV append (k): ") + cudaGetErrorString(st);
            return false;
        }
        const cudaError_t sv =
            cudaMemcpy(p->v_pool + at, cell.v + h * p->head_dim, cell_bytes,
                       cudaMemcpyHostToDevice);
        if (sv != cudaSuccess) {
            err = std::string("device KV append (v): ") + cudaGetErrorString(sv);
            return false;
        }
    }
    return true;
}

bool dev_read_pool(KvDevicePool* p, std::uint16_t* k_out, std::uint16_t* v_out,
                   std::string& err) {
    err.clear();
    if (p == nullptr || k_out == nullptr || v_out == nullptr) {
        err = "device KV read_pool: pointers incomplete";
        return false;
    }
    const std::size_t bytes = p->pool_elems * sizeof(std::uint16_t);
    const cudaError_t sk = cudaMemcpy(k_out, p->k_pool, bytes, cudaMemcpyDeviceToHost);
    if (sk != cudaSuccess) {
        err = std::string("device KV read_pool (k): ") + cudaGetErrorString(sk);
        return false;
    }
    const cudaError_t sv = cudaMemcpy(v_out, p->v_pool, bytes, cudaMemcpyDeviceToHost);
    if (sv != cudaSuccess) {
        err = std::string("device KV read_pool (v): ") + cudaGetErrorString(sv);
        return false;
    }
    return true;
}

bool dev_release(KvDevicePool* p, std::string& err) { return kv_free(p, err); }

}  // namespace

KvDeviceExec kv_device_exec() {
    KvDeviceExec e;
    e.reserve = &dev_reserve;
    e.append = &dev_append;
    e.read_pool = &dev_read_pool;
    e.release = &dev_release;
    return e;
}

}  // namespace qinfer::dense