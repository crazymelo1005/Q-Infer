// Device-side KV pool for QSA layers (stage B: KV on GPU).
//
// Mirrors the host `KvPageCache` (see kv_page_cache.hpp for the layout
// formulas): fp16, paged, `[page][kv_head][page_size][head_dim]` with
// head_dim fastest; logical cell t, head h lives at
//     base + (h * page_size + t % page_size) * head_dim
// where base = page * n_head_kv * page_size * head_dim (identity page mapping
// at this stage; eviction/readmission arrives with --kv-resident in stage B
// task 4). The page table lives on the device (int32, -1 = not resident).
//
// Only compiled under -DQINFER_ENABLE_CUDA=ON (ADR-008). Host builds never see
// these symbols; wiring follows the DenseExec pattern: QsaScratch holds a
// const KvDeviceExec* that the CUDA factory fills, null on host builds.
#pragma once

#include <cstdint>
#include <string>

namespace qinfer::dense {

struct KvCell;              // dense/kv_page_cache.hpp — fwd for the exec signature
struct KvDevicePool;        // opaque CUDA-side pool handle (allocated by reserve)

// Device KV operations behind function pointers so host translation units
// never reference CUDA symbols. Each entry returns false with an English `err`
// on failure; no silent fallback.
struct KvDeviceExec {
    // Allocate device k/v pools and the page table for the given geometry.
    // `out` receives the new pool (nullptr on failure; any partially
    // allocated buffer is freed on the error path).
    bool (*reserve)(KvDevicePool*& out, std::uint64_t n_pages,
                    std::uint64_t n_head_kv, std::uint64_t page_size,
                    std::uint64_t head_dim, std::string& err) = nullptr;
    // Write cell t's K/V into its (identity-mapped) device page. Same offset
    // math as host kv_append; only the cell bytes cross the bus.
    bool (*append)(KvDevicePool*, std::uint64_t t, const KvCell& cell,
                   std::string& err) = nullptr;
    // Copy the whole k/v pools back to host buffers (test/verification hook).
    bool (*read_pool)(KvDevicePool*, std::uint16_t* k_out, std::uint16_t* v_out,
                      std::string& err) = nullptr;
    // Free the device buffers. Safe to call with a null pool (no-op success).
    bool (*release)(KvDevicePool*, std::string& err) = nullptr;
};

// CUDA-side factory (defined in kv_device_cache.cu). Fills all four entries.
KvDeviceExec kv_device_exec();

}  // namespace qinfer::dense