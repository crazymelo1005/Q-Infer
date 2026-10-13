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

// Page base offset in a pool (elements before the first kv head of the page).
// __host__ __device__: shared by the host-side offset math and the gather kernel.
__host__ __device__ std::size_t page_base(std::uint64_t page, std::uint64_t page_size,
                                          std::uint64_t n_head_kv, std::uint64_t head_dim) {
    return static_cast<std::size_t>(page * page_size * n_head_kv * head_dim);
}

// Device gather: one output element per thread; the source offset recomputes
// the same page/head/cell layout the host kv_gather memcpy's from. The wrapper
// pre-checks the host mirror table, so `page` is never -1 here.
__global__ void kv_gather_dev_kernel(const std::uint16_t* __restrict__ k_pool,
                                     const std::uint16_t* __restrict__ v_pool,
                                     const std::int32_t* __restrict__ table,
                                     const std::uint64_t* __restrict__ ids,
                                     std::uint16_t* __restrict__ d_k_scratch,
                                     std::uint16_t* __restrict__ d_v_scratch,
                                     int n_ids, int n_kv, int page_size, int head_dim) {
    const int total = n_ids * n_kv * head_dim;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const int e = idx % head_dim;
    const int row = idx / head_dim;          // row = id * n_kv + h
    const int id = row / n_kv;
    const int h = row - id * n_kv;
    const std::uint64_t t = ids[id];
    const std::int32_t page = table[static_cast<std::size_t>(t / static_cast<std::uint64_t>(page_size))];
    const std::size_t at =
        page_base(static_cast<std::uint64_t>(page), static_cast<std::uint64_t>(page_size),
                  static_cast<std::uint64_t>(n_kv), static_cast<std::uint64_t>(head_dim)) +
        static_cast<std::size_t>((h * page_size + static_cast<int>(t % static_cast<std::uint64_t>(page_size))) *
                                 head_dim + e);
    d_k_scratch[idx] = k_pool[at];
    d_v_scratch[idx] = v_pool[at];
}

bool dev_gather(KvDevicePool* p, const std::uint64_t* ids, std::uint64_t n_ids,
                std::uint16_t* d_k_scratch, std::uint16_t* d_v_scratch, std::string& err) {
    err.clear();
    if (p == nullptr || (n_ids != 0 && ids == nullptr) || d_k_scratch == nullptr ||
        d_v_scratch == nullptr) {
        err = "device KV gather: pointers incomplete";
        return false;
    }
    if (n_ids == 0) return true;
    // Pre-check the host mirror: every selected page must be resident (-1 never).
    for (std::uint64_t i = 0; i < n_ids; ++i) {
        const std::uint64_t logi = ids[i] / p->page_size;
        if (logi >= p->n_pages) {
            err = "device KV gather: cell " + std::to_string(ids[i]) + " page out of range";
            return false;
        }
        if (p->table[static_cast<std::size_t>(logi)] < 0) {
            err = "device KV gather: page " + std::to_string(logi) +
                  " not resident (host streaming arrives with --kv-resident)";
            return false;
        }
    }
    const std::size_t id_bytes = static_cast<std::size_t>(n_ids) * sizeof(std::uint64_t);
    std::uint64_t* d_ids = nullptr;
    if (cudaMalloc(&d_ids, id_bytes) != cudaSuccess) {
        err = "device KV gather: id buffer allocation failed";
        return false;
    }
    if (cudaMemcpy(d_ids, ids, id_bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        err = "device KV gather: id upload failed";
        cudaFree(d_ids);
        return false;
    }
    const int total = static_cast<int>(n_ids * p->n_head_kv * p->head_dim);
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    kv_gather_dev_kernel<<<blocks, threads>>>(
        p->k_pool, p->v_pool, p->d_table, d_ids, d_k_scratch, d_v_scratch,
        static_cast<int>(n_ids), static_cast<int>(p->n_head_kv),
        static_cast<int>(p->page_size), static_cast<int>(p->head_dim));
    cudaFree(d_ids);
    const cudaError_t st = cudaDeviceSynchronize();
    if (st != cudaSuccess) {
        err = std::string("device KV gather: kernel failed: ") + cudaGetErrorString(st);
        return false;
    }
    return true;
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
    e.gather = &dev_gather;
    e.read_pool = &dev_read_pool;
    e.release = &dev_release;
    return e;
}

}  // namespace qinfer::dense