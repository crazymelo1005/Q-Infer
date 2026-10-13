// P3a step 3: the IQ2_XXS x Q8_K GEMV shader against the host reference.
//
// Same shape as tests/test_igpu_iq2s.cpp: a rows x blocks weight matrix, a non-uniform activation
// quantized with kernels::q8k_quantize_row (IQ2_XXS pairs with Q8_K), and kernels::iq2xxs_gemv_q8k
// as the oracle. Acceptance is the repo's iGPU criterion: bit-same ratio >= 0.99 and L1 relative
// error <= 1e-4.
//
// IQ2_XXS needs two static tables at run time: the 256-entry grid codebook (kIq2xxsGrid) and the
// 128-entry sign-pattern table (kSigns, shared with IQ2_XS / IQ2_S). Both are uploaded as storage
// buffers (bindings 3 and 4) instead of being hardcoded in the shader.
//
// The weight bytes are PRNG-filled (fixed seed) with the fp16 block scale drawn from a safe set,
// so random exponents can never produce an Inf/NaN. Every grid index is 8 bits into a 256-entry
// table and every sign index is 7 bits into a 128-entry table, so any byte pattern stays in range.
#include "experts/igpu/shaders/spv_data.hpp"
#include "experts/igpu/vulkan_runtime.hpp"

#include "kernels/iq2xxs.hpp"
#include "kernels/iq2xxs_tables.hpp"
#include "kernels/iq2xs_tables.hpp"
#include "kernels/q8k.hpp"

#include "check.hpp"
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

using namespace qinfer;

namespace {
namespace vk = qinfer::experts::igpu::vk;

constexpr std::uint64_t kRows = 64;
constexpr std::uint64_t kBlocksPerRow = 3;
constexpr std::uint64_t kNb = kRows * kBlocksPerRow;

// A deterministic xorshift32 so the fixture is reproducible across runs.
std::uint32_t xorshift(std::uint32_t& s) {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return s;
}

// Safe fp16 bit patterns (no Inf, no NaN, all normal and positive).
const std::uint16_t kSafeF16[4] = {0x1C00, 0x2400, 0x2C00, 0x3400};

void fill_weights(std::vector<std::uint8_t>& w) {
    std::uint32_t s = 0x7654321u;
    for (std::uint64_t b = 0; b < kNb; ++b) {
        std::uint8_t* blk = w.data() + b * kernels::kIq2xxsBlockBytes;
        for (std::size_t i = 0; i < kernels::kIq2xxsBlockBytes; ++i) {
            blk[i] = static_cast<std::uint8_t>(xorshift(s) & 0xFFu);
        }
        const std::uint16_t d = kSafeF16[b % 4];
        blk[0] = static_cast<std::uint8_t>(d & 0xFFu);
        blk[1] = static_cast<std::uint8_t>(d >> 8);
    }
}

int compare(const std::vector<float>& host, const std::vector<float>& dev, const char* what) {
    std::uint64_t same = 0;
    double diff = 0.0, abs = 0.0;
    for (std::size_t i = 0; i < host.size(); ++i) {
        if (std::bit_cast<std::uint32_t>(host[i]) == std::bit_cast<std::uint32_t>(dev[i])) ++same;
        diff += std::abs(static_cast<double>(host[i]) - static_cast<double>(dev[i]));
        abs += std::abs(static_cast<double>(host[i]));
    }
    const double l1 = abs > 0.0 ? diff / abs : 0.0;
    const double ratio = static_cast<double>(same) / static_cast<double>(host.size());
    std::printf("  %s: bit-same %.3f, L1 rel %.3e\n", what, ratio, l1);
    return (ratio >= 0.99 && l1 <= 1e-4) ? 0 : 1;
}

int test_iq2xxs_gemv() {
    // ---- host side ----
    std::vector<std::uint8_t> w(static_cast<std::size_t>(kNb * kernels::kIq2xxsBlockBytes));
    fill_weights(w);
    std::vector<float> x(static_cast<std::size_t>(kBlocksPerRow * kernels::kIq2xxsValuesPerBlock));
    for (std::size_t i = 0; i < x.size(); ++i) {
        x[i] = 0.5f + 0.01f * static_cast<float>((i * 5 + 3) % 11) *
                          (static_cast<float>(i % 2) * 2.0f - 1.0f);
    }
    std::vector<kernels::Q8kBlock> y(static_cast<std::size_t>(kBlocksPerRow));
    kernels::q8k_quantize_row(x.data(), y.data(), static_cast<int>(kBlocksPerRow));
    std::vector<float> host_out(kRows, 0.0f);
    kernels::iq2xxs_gemv_q8k(w.data(), static_cast<int>(kRows), static_cast<int>(kBlocksPerRow),
                             y.data(), host_out.data());

    // ---- iGPU side ----
    experts::igpu::VulkanRuntime& rt = experts::igpu::igpu_runtime();
    std::string err;
    if (!rt.load(err) || !rt.init_device(err) || !rt.available()) {
        std::printf("  skipped (no iGPU): %s\n", rt.reason().c_str());
        return 0;
    }
    void* d_w = nullptr;
    void* d_y = nullptr;
    void* d_out = nullptr;
    void* d_g = nullptr;
    void* d_s = nullptr;
    const auto* grid_bytes = reinterpret_cast<const std::uint8_t*>(kernels::detail::kIq2xxsGrid);
    const std::size_t grid_size = sizeof(kernels::detail::kIq2xxsGrid);
    const std::size_t sign_size = sizeof(kernels::detail::kSigns);
    if (!rt.alloc_device(w.size(), d_w, err) || !rt.upload(w.data(), w.size(), d_w, err) ||
        !rt.alloc_device(y.size() * sizeof(kernels::Q8kBlock), d_y, err) ||
        !rt.upload(y.data(), y.size() * sizeof(kernels::Q8kBlock), d_y, err) ||
        !rt.alloc_device(kRows * sizeof(float), d_out, err) ||
        !rt.alloc_device(grid_size, d_g, err) || !rt.upload(grid_bytes, grid_size, d_g, err) ||
        !rt.alloc_device(sign_size, d_s, err) ||
        !rt.upload(kernels::detail::kSigns, sign_size, d_s, err)) {
        std::printf("FAIL buffers: %s\n", err.c_str());
        return 1;
    }
    vk::DescriptorSetLayoutBinding bindings[5] = {};
    for (std::uint32_t i = 0; i < 5; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = vk::DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = vk::SHADER_STAGE_COMPUTE;
    }
    vk::DescriptorSetLayout sl = rt.make_set_layout(bindings, 5, err);
    vk::PipelineLayout pl = sl != nullptr ? rt.make_layout(&sl, 1, err) : nullptr;
    vk::Pipeline pipe =
        pl != nullptr ? rt.make_pipeline(experts::igpu::shaders::kIq2xxsGemvSpv,
                                         experts::igpu::shaders::kIq2xxsGemvSpvSize, pl, err)
                      : nullptr;
    vk::Buffer bufs[5] = {static_cast<vk::Buffer>(d_w), static_cast<vk::Buffer>(d_y),
                          static_cast<vk::Buffer>(d_out), static_cast<vk::Buffer>(d_g),
                          static_cast<vk::Buffer>(d_s)};
    vk::DescriptorSet set = pipe != nullptr ? rt.make_set(sl, bufs, 5, err) : nullptr;
    if (set == nullptr) {
        std::printf("FAIL pipeline/set: %s\n", err.c_str());
        return 1;
    }
    struct Pc {
        std::uint32_t n_rows;
        std::uint32_t n_blocks;
    } pc{static_cast<std::uint32_t>(kRows), static_cast<std::uint32_t>(kBlocksPerRow)};
    const std::uint32_t gx = static_cast<std::uint32_t>((kRows + 63) / 64);
    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();
    if (!rt.dispatch(pipe, pl, set, &pc, sizeof(pc), gx, 1, 1, err)) {
        std::printf("FAIL dispatch: %s\n", err.c_str());
        return 1;
    }
    const auto t1 = clock::now();
    std::vector<float> dev_out(kRows);
    if (!rt.download(d_out, kRows * sizeof(float), dev_out.data(), err)) {
        std::printf("FAIL download: %s\n", err.c_str());
        return 1;
    }
    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::printf("  one dispatch (%llu x %llu blocks): %.2f ms\n", (unsigned long long) kRows,
                (unsigned long long) kBlocksPerRow, ms);
    rt.free_buf(d_w, err);
    rt.free_buf(d_y, err);
    rt.free_buf(d_out, err);
    rt.free_buf(d_g, err);
    rt.free_buf(d_s, err);
    return compare(host_out, dev_out, "iq2xxs iGPU vs host");
}

}  // namespace

int main() {
    const int rc = test_iq2xxs_gemv();
    if (rc == 0) std::puts("igpu_iq2xxs: IQ2_XXS x Q8_K matches the host");
    return rc;
}
