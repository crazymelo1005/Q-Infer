// measure/igpu_gemv_bench.cpp — P3a step 7 (the falsification point).
//
// Times the three gate/up GEMV shaders against the host kernels on the REAL batch shape of the
// deployed pack (gsq-rco-iq2_xs): one expert's gate (or up) matrix is 640 rows x 10 IQ blocks
// (expert_feed_forward_length=640 over a 2560-wide hidden, 256 values per IQ block), and a token
// routes to 10 experts (expert_used_count=10), so a decode step's non-resident work is
// experts x {gate, up} matrices of exactly that shape.
//
// Four arms, all on identical bytes:
//
//   host1   host kernel, one thread                     - the single-job latency
//   hostT   host kernel, rows split over T threads      - what the engine's CPU pool costs
//   gpu     device dispatch, weights already resident   - the compute-only floor
//   gpu_e2e upload the weights per matrix + dispatch + one download at the end
//                                                       - what P3b would actually do
//
// The decision the plan binds to this: if the iGPU is not clearly faster than the CPU pool for the
// same batch, P3b (wiring the device into the engine's PoolFn) is not worth doing, and these
// readings become the record.
//
// Four controls back that verdict up, all reachable from the command line (ADR-019 names them
// G1/C1/S1/O1):
//
//   G1  merge limit    --experts 5 --rows 1280 / --experts 2 --rows 3200 / --experts 1 --rows 6400
//                      Constant 12,800 rows and the same bytes per batch; only the rows per
//                      dispatch changes. One PoolFn call hands over one layer's k experts
//                      (k <= 10), so 6,400 rows is the most a real integration could merge.
//   C1  compute ceiling --rows 25600 --experts 1
//                      Beyond what the seam allows, but it separates dispatch overhead from
//                      compute: one huge resident dispatch is the device's throughput floor.
//   S1  working set     --experts 100
//                      200 matrices = ~100 MiB, far past the 36 MiB L3, so the host arm cannot be
//                      winning on cache residency.
//   O1  per dispatch    --rows 64 --blocks 1 --experts 1
//                      Two nearly-empty dispatches measure the fixed cost of one dispatch.
//
// Usage: igpu_gemv_bench [--format iq2s|iq2xxs|iq1m] [--rows 640] [--blocks 10] [--experts 10]
//                        [--threads 1,8] [--reps 20]

#include "experts/igpu/shaders/spv_data.hpp"
#include "experts/igpu/vulkan_runtime.hpp"

#include "kernels/iq1m.hpp"
#include "kernels/iq1m_tables.hpp"
#include "kernels/iq2s.hpp"
#include "kernels/iq2s_tables.hpp"
#include "kernels/iq2xxs.hpp"
#include "kernels/iq2xxs_tables.hpp"
#include "kernels/iq2xs_tables.hpp"
#include "kernels/q8k.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

using namespace qinfer;
namespace vk = qinfer::experts::igpu::vk;

namespace {

struct Opts {
    std::string format = "iq2s";
    std::uint32_t rows = 640;
    std::uint32_t blocks = 10;
    std::uint32_t experts = 10;
    std::vector<int> threads = {1, 8};
    int reps = 20;
};

struct Format {
    const char* name;
    std::uint32_t block_bytes;
    const std::uint8_t* spv;
    std::size_t spv_size;
    int n_extra_bindings;  // static tables uploaded as storage buffers
    const void* grid;
    std::size_t grid_size;
    const void* signs;
    std::size_t sign_size;
    // One matrix (rows x blocks weight bytes) against the shared activation.
    void (*gemv)(const std::uint8_t*, int, int, const kernels::Q8kBlock*, float*);
};

const Format kFormats[] = {
    {"iq2s", kernels::kIq2sBlockBytes, experts::igpu::shaders::kIq2sGemvSpv,
     experts::igpu::shaders::kIq2sGemvSpvSize, 1, kernels::detail::kIq2sGrid,
     sizeof(kernels::detail::kIq2sGrid), nullptr, 0, kernels::iq2s_gemv_q8k},
    {"iq2xxs", kernels::kIq2xxsBlockBytes, experts::igpu::shaders::kIq2xxsGemvSpv,
     experts::igpu::shaders::kIq2xxsGemvSpvSize, 2, kernels::detail::kIq2xxsGrid,
     sizeof(kernels::detail::kIq2xxsGrid), kernels::detail::kSigns,
     sizeof(kernels::detail::kSigns), kernels::iq2xxs_gemv_q8k},
    {"iq1m", kernels::kIq1mBlockBytes, experts::igpu::shaders::kIq1mGemvSpv,
     experts::igpu::shaders::kIq1mGemvSpvSize, 1, kernels::detail::kIq1sGrid,
     sizeof(kernels::detail::kIq1sGrid), nullptr, 0, kernels::iq1m_gemv_q8k},
};

const Format& pick_format(const std::string& name) {
    for (const Format& f : kFormats) {
        if (name == f.name) return f;
    }
    std::fprintf(stderr, "unknown --format %s (iq2s|iq2xxs|iq1m)\n", name.c_str());
    std::exit(2);
}

std::uint32_t xorshift(std::uint32_t& s) {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return s;
}

// Deterministic weight bytes; the fp16 block scale (or, for IQ1_M, the four nibbles that assemble
// it) is forced to a safe pattern so a random exponent cannot produce an Inf/NaN.
std::vector<std::uint8_t> make_weights(const Format& f, std::uint64_t n_matrices, std::uint32_t rows,
                                       std::uint32_t blocks) {
    const std::uint64_t mat_bytes = static_cast<std::uint64_t>(rows) * blocks * f.block_bytes;
    std::vector<std::uint8_t> w(static_cast<std::size_t>(n_matrices * mat_bytes));
    std::uint32_t s = 0x9E3779B9u;
    for (std::size_t i = 0; i < w.size(); ++i) w[i] = static_cast<std::uint8_t>(xorshift(s) & 0xFFu);
    const std::uint16_t safe[4] = {0x1C00, 0x2400, 0x2C00, 0x3400};
    for (std::uint64_t m = 0; m < n_matrices; ++m) {
        for (std::uint32_t r = 0; r < rows; ++r) {
            for (std::uint32_t b = 0; b < blocks; ++b) {
                std::uint8_t* blk = w.data() + static_cast<std::size_t>(m * mat_bytes) +
                                    static_cast<std::size_t>(r) * blocks * f.block_bytes +
                                    static_cast<std::size_t>(b) * f.block_bytes;
                const std::uint16_t d = safe[(m * 3 + r + b) % 4];
                if (f.block_bytes == static_cast<std::uint32_t>(kernels::kIq1mBlockBytes)) {
                    std::uint8_t* sc = blk + 48;
                    const std::uint8_t nib[4] = {static_cast<std::uint8_t>(d & 0xFu),
                                                 static_cast<std::uint8_t>((d >> 4) & 0xFu),
                                                 static_cast<std::uint8_t>((d >> 8) & 0xFu),
                                                 static_cast<std::uint8_t>((d >> 12) & 0xFu)};
                    for (int i = 0; i < 4; ++i)
                        sc[2 * i + 1] = static_cast<std::uint8_t>((sc[2 * i + 1] & 0x0Fu) |
                                                                  (nib[i] << 4));
                } else {
                    blk[0] = static_cast<std::uint8_t>(d & 0xFFu);
                    blk[1] = static_cast<std::uint8_t>(d >> 8);
                }
            }
        }
    }
    return w;
}

using clock = std::chrono::steady_clock;

double ms_since(clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(clock::now() - t0).count();
}

// Host arm: the whole batch on T threads, rows partitioned in contiguous slices.
double host_batch(const Format& f, const std::vector<std::uint8_t>& w, std::uint64_t n_matrices,
                  std::uint32_t rows, std::uint32_t blocks, const std::vector<kernels::Q8kBlock>& y,
                  std::vector<float>& out, int threads, int reps) {
    const std::uint64_t mat_bytes = static_cast<std::uint64_t>(rows) * blocks * f.block_bytes;
    double best = 1e300;
    for (int rep = 0; rep < reps; ++rep) {
        const auto t0 = clock::now();
        std::vector<std::thread> pool;
        pool.reserve(static_cast<std::size_t>(threads));
        const std::uint64_t total = n_matrices * rows;
        const std::uint64_t slice = (total + threads - 1) / static_cast<std::uint64_t>(threads);
        for (int t = 0; t < threads; ++t) {
            const std::uint64_t lo = std::min(static_cast<std::uint64_t>(t) * slice, total);
            const std::uint64_t hi = std::min(lo + slice, total);
            if (lo >= hi) break;
            pool.emplace_back([&f, &w, &y, &out, rows, blocks, mat_bytes, lo, hi]() {
                // A row belongs to matrix m = row / rows, so each thread may cross matrices.
                for (std::uint64_t g = lo; g < hi;) {
                    const std::uint64_t m = g / rows;
                    const std::uint32_t r0 = static_cast<std::uint32_t>(g - m * rows);
                    const std::uint32_t cnt =
                        static_cast<std::uint32_t>(std::min(hi - g, static_cast<std::uint64_t>(rows - r0)));
                    const std::uint8_t* wp =
                        w.data() + static_cast<std::size_t>(m * mat_bytes) +
                        static_cast<std::size_t>(r0) * blocks * f.block_bytes;
                    f.gemv(wp, static_cast<int>(cnt), static_cast<int>(blocks), y.data(),
                           out.data() + static_cast<std::size_t>(m * rows) + r0);
                    g += cnt;
                }
            });
        }
        for (std::thread& th : pool) th.join();
        best = std::min(best, ms_since(t0));
    }
    return best;
}

struct Device {
    experts::igpu::VulkanRuntime* rt = nullptr;
    vk::DescriptorSetLayout sl = nullptr;
    vk::PipelineLayout pl = nullptr;
    vk::Pipeline pipe = nullptr;
    vk::DescriptorSet set = nullptr;
    void* d_w = nullptr;
    void* d_y = nullptr;
    void* d_out = nullptr;
    std::vector<void*> d_table;
    std::uint32_t rows = 0;
    std::uint32_t blocks = 0;
    std::string err;
    bool ok = false;
};

bool device_init(Device& d, const Format& f, std::uint32_t rows, std::uint32_t blocks,
                 std::uint64_t n_matrices) {
    d.rt = &experts::igpu::igpu_runtime();
    if (!d.rt->load(d.err) || !d.rt->init_device(d.err) || !d.rt->available()) return false;
    d.rows = rows;
    d.blocks = blocks;
    const std::uint32_t n_buffers = 3u + static_cast<std::uint32_t>(f.n_extra_bindings);
    std::vector<vk::DescriptorSetLayoutBinding> bindings(n_buffers);
    for (std::uint32_t i = 0; i < n_buffers; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = vk::DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = vk::SHADER_STAGE_COMPUTE;
    }
    d.sl = d.rt->make_set_layout(bindings.data(), n_buffers, d.err);
    d.pl = d.sl != nullptr ? d.rt->make_layout(&d.sl, 1, d.err) : nullptr;
    d.pipe = d.pl != nullptr ? d.rt->make_pipeline(f.spv, f.spv_size, d.pl, d.err) : nullptr;
    if (d.pipe == nullptr) return false;
    // One weight matrix is reused across dispatches (the pool would cycle matrices through the
    // same slot), plus the shared activation and the batch's output area.
    const std::size_t mat_bytes = static_cast<std::size_t>(rows) * blocks * f.block_bytes;
    if (!d.rt->alloc_device(mat_bytes, d.d_w, d.err) ||
        !d.rt->alloc_device(static_cast<std::size_t>(blocks) * sizeof(kernels::Q8kBlock), d.d_y,
                            d.err) ||
        !d.rt->alloc_device(static_cast<std::size_t>(n_matrices) * rows * sizeof(float), d.d_out,
                            d.err)) {
        return false;
    }
    const void* tables[2] = {f.grid, f.signs};
    const std::size_t sizes[2] = {f.grid_size, f.sign_size};
    for (int i = 0; i < f.n_extra_bindings; ++i) {
        void* p = nullptr;
        if (!d.rt->alloc_device(sizes[i], p, d.err) ||
            !d.rt->upload(tables[i], sizes[i], p, d.err)) {
            return false;
        }
        d.d_table.push_back(p);
    }
    std::vector<vk::Buffer> bufs(n_buffers);
    bufs[0] = static_cast<vk::Buffer>(d.d_w);
    bufs[1] = static_cast<vk::Buffer>(d.d_y);
    bufs[2] = static_cast<vk::Buffer>(d.d_out);
    for (std::size_t i = 0; i < d.d_table.size(); ++i)
        bufs[3 + i] = static_cast<vk::Buffer>(d.d_table[i]);
    d.set = d.rt->make_set(d.sl, bufs.data(), n_buffers, d.err);
    if (d.set == nullptr) return false;
    return true;
}

// One matrix: optionally upload its weights into the reused slot, then dispatch. `upload=false`
// dispatches on the bytes already there - GEMV cost is data-independent, so that is the
// compute-only floor without paying the copy.
bool device_matrix(Device& d, const std::uint8_t* w, std::size_t w_bytes, bool upload) {
    if (upload && !d.rt->upload(w, w_bytes, d.d_w, d.err)) return false;
    struct Pc {
        std::uint32_t n_rows;
        std::uint32_t n_blocks;
    } pc{d.rows, d.blocks};
    // Every matrix writes rows 0..n-1 of d_out; the benchmark sums per-matrix latencies rather
    // than collecting n distinct outputs, which is what a per-job pool would pay.
    return d.rt->dispatch(d.pipe, d.pl, d.set, &pc, sizeof(pc), (d.rows + 63) / 64, 1, 1, d.err);
}

int run(const Opts& o) {
    const Format& f = pick_format(o.format);
    const std::uint64_t n_mat = o.experts * 2;  // gate and up, the two the shaders serve
    const std::uint64_t mat_bytes = static_cast<std::uint64_t>(o.rows) * o.blocks * f.block_bytes;
    std::vector<std::uint8_t> w = make_weights(f, n_mat, o.rows, o.blocks);
    std::vector<float> x(static_cast<std::size_t>(o.blocks) * kernels::kQ8kBlockElems, 0.0f);
    std::uint32_t s = 0x51ED2701u;
    for (float& v : x) v = 0.4f + 0.02f * static_cast<float>(xorshift(s) % 51);
    std::vector<kernels::Q8kBlock> y(o.blocks);
    kernels::q8k_quantize_row(x.data(), y.data(), static_cast<int>(o.blocks));
    std::vector<float> out(static_cast<std::size_t>(n_mat) * o.rows, 0.0f);

    std::printf("format %s: %llu matrices of %u x %u blocks (%llu B each, %.3f MiB total)\n",
                f.name, (unsigned long long) n_mat, o.rows, o.blocks,
                (unsigned long long) mat_bytes,
                static_cast<double>(n_mat * mat_bytes) / (1024.0 * 1024.0));

    for (int t : o.threads) {
        double ms = host_batch(f, w, n_mat, o.rows, o.blocks, y, out, t, o.reps);
        std::printf("  %-7s %2d thread(s): %8.2f ms/batch  %10.1f rows/s\n",
                    t == 1 ? "host1" : "hostT", t, ms,
                    static_cast<double>(n_mat * o.rows) / (ms / 1000.0));
    }

    Device d;
    if (!device_init(d, f, o.rows, o.blocks, n_mat)) {
        std::printf("  gpu     skipped: %s\n", d.err.empty() ? "no iGPU" : d.err.c_str());
        return 0;
    }
    if (!d.rt->upload(y.data(), y.size() * sizeof(kernels::Q8kBlock), d.d_y, d.err)) {
        std::printf("  gpu     skipped: %s\n", d.err.c_str());
        return 0;
    }
    double best_compute = 1e300, best_e2e = 1e300;
    const std::size_t w_bytes = static_cast<std::size_t>(mat_bytes);
    const std::size_t out_bytes = static_cast<std::size_t>(n_mat) * o.rows * sizeof(float);
    std::vector<std::uint8_t> sink(out_bytes, 0);
    for (int rep = 0; rep < o.reps; ++rep) {
        auto t0 = clock::now();
        for (std::uint64_t m = 0; m < n_mat; ++m) {
            if (!device_matrix(d, w.data() + static_cast<std::size_t>(m * mat_bytes), w_bytes,
                               /*upload=*/false)) {
                std::printf("  gpu     dispatch failed: %s\n", d.err.c_str());
                return 1;
            }
        }
        best_compute = std::min(best_compute, ms_since(t0));
        t0 = clock::now();
        for (std::uint64_t m = 0; m < n_mat; ++m) {
            if (!device_matrix(d, w.data() + static_cast<std::size_t>(m * mat_bytes), w_bytes,
                               /*upload=*/true)) {
                std::printf("  gpu_e2e failed: %s\n", d.err.c_str());
                return 1;
            }
        }
        if (!d.rt->download(d.d_out, out_bytes, sink.data(), d.err)) {
            std::printf("  gpu_e2e download failed: %s\n", d.err.c_str());
            return 1;
        }
        best_e2e = std::min(best_e2e, ms_since(t0));
    }
    const double rows = static_cast<double>(n_mat) * o.rows;
    std::printf("  %-7s %2s        : %8.2f ms/batch  %10.1f rows/s  (dispatch only)\n", "gpu", "-",
                best_compute, rows / (best_compute / 1000.0));
    std::printf("  %-7s %2s        : %8.2f ms/batch  %10.1f rows/s  (upload+dispatch+download)\n",
                "gpu_e2e", "-", best_e2e, rows / (best_e2e / 1000.0));
    std::string err;
    d.rt->free_buf(d.d_w, err);
    for (void* p : d.d_table) d.rt->free_buf(p, err);
    d.rt->free_buf(d.d_y, err);
    d.rt->free_buf(d.d_out, err);
    return 0;
}

Opts parse(int argc, char** argv) {
    Opts o;
    auto need = [&](int& i) {
        if (i + 1 >= argc) {
            std::fprintf(stderr, "missing value for %s\n", argv[i]);
            std::exit(2);
        }
        return argv[++i];
    };
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--format") o.format = need(i);
        else if (a == "--rows") o.rows = static_cast<std::uint32_t>(std::atoi(need(i)));
        else if (a == "--blocks") o.blocks = static_cast<std::uint32_t>(std::atoi(need(i)));
        else if (a == "--experts") o.experts = static_cast<std::uint32_t>(std::atoi(need(i)));
        else if (a == "--threads") {
            o.threads.clear();
            std::string v = need(i), tok;
            for (char c : v + ",") {
                if (c == ',') { if (!tok.empty()) o.threads.push_back(std::atoi(tok.c_str())); tok.clear(); }
                else tok += c;
            }
        } else if (a == "--reps") o.reps = std::atoi(need(i));
        else {
            std::fprintf(stderr, "unknown arg %s\n", a.c_str());
            std::exit(2);
        }
    }
    return o;
}

}  // namespace

int main(int argc, char** argv) { return run(parse(argc, argv)); }
